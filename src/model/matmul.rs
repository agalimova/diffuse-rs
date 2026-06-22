use super::*;

// =============================================================================
// Quantized matmul
// =============================================================================

// Raw pointer that may cross rayon task boundaries.
// SAFETY: every task writes a disjoint set of indices.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr(pub(crate) *mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    /// SAFETY: idx must be in bounds and not written concurrently.
    pub(crate) unsafe fn write(self, idx: usize, v: f32) {
        *self.0.add(idx) = v;
    }
}

/// Quantized weight matrix. Q4_K/Q6_K/Q4_0 run through our row-parallel
/// matmul over candle's SIMD vec_dot (zero-copy from the mmap); any other
/// dtype falls back to candle's QMatMul.
pub(crate) enum MatWeight {
    Q4K(NativeWeight),
    Q6K(NativeWeight),
    Q40(NativeWeight),
    Candle(QMatMul),
}

/// Activations quantized for vec_dot, cached per activation buffer so e.g.
/// Q/K/V projections quantize their shared input once. Q4_K/Q6_K weights
/// consume Q8_K activations; Q4_0/Q5_0/Q8_0 consume Q8_0.
#[derive(Default)]
pub(crate) struct ActScratch {
    q8k: Vec<BlockQ8K>,
    q8k_key: ActKey,
    q80: Vec<k_quants::BlockQ8_0>,
    q80_key: ActKey,
    /// Bumped whenever a reused activation buffer (b.normed) is rewritten.
    /// Distinguishes different content that lands at the same (ptr, len), so a
    /// cache hit cannot serve a stale quantization on mixed-quant models.
    gen: u64,
}

/// (ptr, len, generation) identity of a quantized activation buffer.
type ActKey = (usize, usize, u64);

impl ActScratch {
    /// Buffers are reused with identical (ptr, len) across passes; callers
    /// must invalidate at every pass boundary.
    pub(crate) fn invalidate(&mut self) {
        self.q8k_key = (0, 0, 0);
        self.q80_key = (0, 0, 0);
    }

    /// Signal that a reused activation buffer was overwritten with new content.
    /// Cheap (one increment); forces the next quantization of a colliding
    /// (ptr, len) instead of serving the stale cache entry.
    pub(crate) fn touch(&mut self) {
        self.gen = self.gen.wrapping_add(1);
    }

    fn q8k(&mut self, act: &[f32], n: usize) -> &[BlockQ8K] {
        let key = (act.as_ptr() as usize, act.len(), self.gen);
        if self.q8k_key != key {
            kernels::quantize_rows_q8_k(act, n, act.len() / n, &mut self.q8k);
            self.q8k_key = key;
        }
        &self.q8k
    }

    fn q80(&mut self, act: &[f32], _n: usize) -> Result<&[k_quants::BlockQ8_0]> {
        let key = (act.as_ptr() as usize, act.len(), self.gen);
        if self.q80_key != key {
            self.q80.clear();
            self.q80.resize(act.len() / 32, k_quants::BlockQ8_0::zeros());
            k_quants::BlockQ8_0::from_float(act, &mut self.q80)?;
            self.q80_key = key;
        }
        Ok(&self.q80)
    }
}

pub(crate) struct NativeWeight {
    pub(crate) bytes: Bytes,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl MatWeight {
    /// out[t*rows + r] = act row t · weight row r.
    pub(crate) fn forward(&self, out: &mut [f32], act: &[f32], sc: &mut ActScratch, n: usize) -> Result<()> {
        match self {
            Self::Q4K(w) => {
                let ys = cast_q8k(sc.q8k(act, n));
                native_matmul::<k_quants::BlockQ4K>(w.bytes.as_slice(), w.rows, w.cols, out, ys, n)
            }
            Self::Q6K(w) => {
                let ys = cast_q8k(sc.q8k(act, n));
                native_matmul::<k_quants::BlockQ6K>(w.bytes.as_slice(), w.rows, w.cols, out, ys, n)
            }
            Self::Q40(w) => {
                let ys = sc.q80(act, n)?;
                native_matmul::<k_quants::BlockQ4_0>(w.bytes.as_slice(), w.rows, w.cols, out, ys, n)
            }
            Self::Candle(w) => project(out, w, act, n),
        }
    }
}

/// Our BlockQ8K is a verbatim layout match of candle's (asserted in tests).
pub(crate) fn cast_q8k(blocks: &[BlockQ8K]) -> &[k_quants::BlockQ8K] {
    unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const k_quants::BlockQ8K, blocks.len())
    }
}

/// Row-parallel quantized matmul over candle's SIMD vec_dot.
/// out[t*rows + r] = act row t · weight row r.
pub(crate) fn native_matmul<B: GgmlType>(
    bytes: &[u8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    act: &[B::VecDotType],
    n: usize,
) -> Result<()> {
    let bpr = cols / B::BLCK_SIZE; // blocks per row
    let row_size = bpr * std::mem::size_of::<B>();
    ensure!(bytes.len() == rows * row_size, "weight size mismatch");
    ensure!(act.len() == n * bpr, "activation block count mismatch");
    ensure!(out.len() == n * rows, "output size mismatch");

    let out_ptr = SendPtr(out.as_mut_ptr());
    (0..rows).into_par_iter().try_for_each(|r| -> Result<()> {
        // SAFETY: r < rows and row_size * rows == bytes.len().
        let row = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr().add(r * row_size) as *const B, bpr)
        };
        for t in 0..n {
            let v = B::vec_dot(cols, row, &act[t * bpr..(t + 1) * bpr])?;
            // SAFETY: each task r writes only indices t*rows + r.
            unsafe { out_ptr.write(t * rows + r, v) };
        }
        Ok(())
    })
}

/// A stack of per-expert weight matrices from a 3D [n_expert, rows, cols]
/// GGUF tensor. Only K-quant dtypes are supported. There is no QMatMul fallback
/// because candle cannot index into expert slices.
pub(crate) enum ExpertMat {
    Q4K(ExpertNative),
    Q6K(ExpertNative),
    Q40(ExpertNative),
    Q80(ExpertNative),
    Q50(ExpertNative),
}

pub(crate) struct ExpertNative {
    pub(crate) bytes: Bytes,
    pub(crate) n_expert: usize,
    /// Rows per expert.
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl ExpertMat {
    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Q4K(w) | Self::Q6K(w) | Self::Q40(w) | Self::Q80(w) | Self::Q50(w) => w.rows,
        }
    }

    /// Matmul against one expert's weight slice.
    pub(crate) fn forward_expert(
        &self,
        expert: usize,
        out: &mut [f32],
        act: &[f32],
        sc: &mut ActScratch,
        n: usize,
    ) -> Result<()> {
        match self {
            Self::Q4K(w) => {
                w.expert_matmul::<k_quants::BlockQ4K>(expert, out, cast_q8k(sc.q8k(act, n)), n)
            }
            Self::Q6K(w) => {
                w.expert_matmul::<k_quants::BlockQ6K>(expert, out, cast_q8k(sc.q8k(act, n)), n)
            }
            Self::Q40(w) => w.expert_matmul::<k_quants::BlockQ4_0>(expert, out, sc.q80(act, n)?, n),
            Self::Q80(w) => w.expert_matmul::<k_quants::BlockQ8_0>(expert, out, sc.q80(act, n)?, n),
            Self::Q50(w) => w.expert_matmul::<k_quants::BlockQ5_0>(expert, out, sc.q80(act, n)?, n),
        }
    }
}

impl ExpertNative {
    fn expert_matmul<B: GgmlType>(
        &self,
        expert: usize,
        out: &mut [f32],
        act: &[B::VecDotType],
        n: usize,
    ) -> Result<()> {
        ensure!(expert < self.n_expert, "expert index out of range");
        let expert_size = self.rows * (self.cols / B::BLCK_SIZE) * std::mem::size_of::<B>();
        let all = self.bytes.as_slice();
        let slice = &all[expert * expert_size..(expert + 1) * expert_size];
        native_matmul::<B>(slice, self.rows, self.cols, out, act, n)
    }
}
// =============================================================================
// Quantized matmul (via candle's QMatMul)
// =============================================================================

/// out[..rows*out_dim] = act x w^T, where act is rows x (act.len()/rows).
pub(crate) fn project(out: &mut [f32], w: &QMatMul, act: &[f32], rows: usize) -> Result<()> {
    let in_dim = act.len() / rows;
    let t = Tensor::from_slice(act, &[rows, in_dim], &Device::Cpu)?;
    let data = w.forward(&t)?.flatten_all()?.to_vec1::<f32>()?;
    out[..data.len()].copy_from_slice(&data);
    Ok(())
}
