use super::*;

// =============================================================================
// Attention
// =============================================================================

/// Attention problem size: nq query rows attending over nk key rows.
/// Q rows have n_head heads; K/V rows have n_head_kv heads. GQA maps each
/// query head to kv head `head / (n_head / n_head_kv)` with no repetition.
#[derive(Clone, Copy)]
pub(crate) struct AttnShape {
    pub(crate) nq: usize,
    pub(crate) nk: usize,
    pub(crate) n_head: usize,
    pub(crate) n_head_kv: usize,
    pub(crate) head_dim: usize,
    /// Symmetric attention band: a key is visible to a query only when their
    /// sequence positions differ by less than `window`. None = full attention.
    pub(crate) sliding_window: Option<usize>,
    /// Dual-mode (Gemma): positions below this are the prompt and attend
    /// causally (encoder); canvas positions attend bidirectionally to all.
    /// None = uniformly bidirectional.
    pub(crate) causal_prefix: Option<usize>,
    /// Score scale (1/sqrt(head_dim) normally; 1.0 for Gemma, which relies on
    /// QK-norm instead).
    pub(crate) scale: f32,
}

/// Whether the key at sequence position `kpos` is visible to the query at
/// `qpos`, under the sliding-window band and causal-prompt boundary. Works on
/// actual sequence positions, so the KV cache may reorder rows freely.
/// `qseq`/`kseq` are the rows' batch-sequence ids; a key is never visible
/// across sequences (block-diagonal attention), keeping batched sequences
/// independent.
#[inline]
fn visible(
    qpos: usize,
    kpos: usize,
    qseq: usize,
    kseq: usize,
    window: Option<usize>,
    causal_prefix: Option<usize>,
) -> bool {
    if qseq != kseq {
        return false;
    }
    if let Some(w) = window {
        if qpos.abs_diff(kpos) >= w {
            return false;
        }
    }
    // Prompt queries (qpos < prefix) may not see future tokens.
    match causal_prefix {
        Some(prefix) if qpos < prefix => kpos <= qpos,
        _ => true,
    }
}

/// Per-row identity for masking: each row's original sequence position and
/// (for batched generation) its sequence id. An empty `seq` means a single
/// sequence. All rows then share id 0, so masking depends only on position.
/// `prefix[i]` is query row i's per-sequence causal-prompt boundary. Gemma
/// batches need the per-row boundary because each sequence has its own prompt
/// length. An empty `prefix` falls back to the shape's scalar `causal_prefix`.
#[derive(Clone, Copy)]
pub(crate) struct Rows<'a> {
    pub(crate) pos: &'a [usize],
    pub(crate) seq: &'a [usize],
    pub(crate) prefix: &'a [usize],
}

impl Rows<'_> {
    #[inline]
    fn seq_at(&self, i: usize) -> usize {
        self.seq.get(i).copied().unwrap_or(0)
    }
}

/// Query and key row metadata for one forward pass. The rows are identical
/// for every layer, so the caller builds them once instead of per layer.
#[derive(Clone, Copy)]
pub(crate) struct PassRows<'a> {
    pub(crate) q: Rows<'a>,
    pub(crate) k: Rows<'a>,
}

/// One attention problem: the Q/K/V buffers, shape, and per-row metadata.
/// `qr`/`kr` carry each query/key row's sequence position and id, so masking
/// stays correct when the KV cache concatenates reordered cached + active rows
/// or when several batched sequences share one forward pass.
#[derive(Clone, Copy)]
pub(crate) struct Attn<'a> {
    pub(crate) q: &'a [f32],
    pub(crate) k: &'a [f32],
    pub(crate) v: &'a [f32],
    pub(crate) s: AttnShape,
    pub(crate) qr: Rows<'a>,
    pub(crate) kr: Rows<'a>,
}

/// Attention: Q[nq, h, hd] x K[nk, hkv, hd] -> out[nq, h*hd].
/// Online-softmax (flash-style): one fused pass over keys per query with a
/// running max/sum and rescaled accumulator. No nq*nk score matrix is ever
/// materialized. (head, query) pairs run in parallel, and each owns a disjoint
/// hd-wide band of `out`.
pub(crate) fn attention(out: &mut [f32], q: &[f32], k: &[f32], v: &[f32], s: AttnShape, qr: Rows, kr: Rows) {
    debug_assert_eq!(out.len(), s.nq * s.n_head * s.head_dim);
    debug_assert_eq!(qr.pos.len(), s.nq);
    debug_assert_eq!(kr.pos.len(), s.nk);
    let out_ptr = SendPtr(out.as_mut_ptr());
    let a = Attn { q, k, v, s, qr, kr };

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2");

    (0..s.n_head * s.nq).into_par_iter().for_each(|task| {
        let (head, i) = (task / s.nq, task % s.nq);
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            unsafe { attention_row_avx2(out_ptr, a, head, i) };
            return;
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is the aarch64 baseline, so the dispatch always uses NEON.
            unsafe { attention_row_neon(out_ptr, a, head, i) };
            return;
        }
        // Portable fallback (non-aarch64, or x86 without AVX2).
        #[cfg(not(target_arch = "aarch64"))]
        attention_row_scalar(out_ptr, a, head, i);
    });
}

/// Head dims up to this use a stack accumulator; larger fall back to heap.
/// The accumulator is the binary's highest-frequency allocation otherwise
/// (one per (head, query) task per layer per step).
const ACC_STACK: usize = 256;

// Portable reference path; unused in the aarch64 bin (NEON is unconditional)
// but still exercised by tests and used on x86/other targets.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn attention_row_scalar(out: SendPtr, a: Attn, head: usize, i: usize) {
    let Attn { q, k, v, s, qr, kr } = a;
    let hd = s.head_dim;
    let scale = s.scale;
    let qo = i * s.n_head * hd + head * hd;
    let kv_stride = s.n_head_kv * hd;
    let kv_off = (head / (s.n_head / s.n_head_kv)) * hd;
    let (qpos, qseq) = (qr.pos[i], qr.seq_at(i));
    let prefix = if qr.prefix.is_empty() { s.causal_prefix } else { Some(qr.prefix[i]) };

    let mut acc_stack = [0.0f32; ACC_STACK];
    let mut acc_heap = Vec::new();
    let acc: &mut [f32] = if hd <= ACC_STACK {
        &mut acc_stack[..hd]
    } else {
        acc_heap.resize(hd, 0.0);
        &mut acc_heap
    };
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    for j in 0..s.nk {
        if !visible(qpos, kr.pos[j], qseq, kr.seq_at(j), s.sliding_window, prefix) {
            continue;
        }
        let ko = j * kv_stride + kv_off;
        let dot: f32 = (0..hd).map(|d| q[qo + d] * k[ko + d]).sum();
        let score = dot * scale;

        let new_max = max.max(score);
        let correction = (max - new_max).exp(); // 1.0 when max unchanged
        let w = (score - new_max).exp();
        sum = sum * correction + w;
        for d in 0..hd {
            acc[d] = acc[d] * correction + w * v[ko + d];
        }
        max = new_max;
    }

    let inv = 1.0 / sum;
    for (d, &a) in acc.iter().enumerate() {
        // SAFETY: task (head, i) exclusively owns this hd-wide band.
        unsafe { out.write(qo + d, a * inv) };
    }
}

/// NEON flash-attention row; width-4 mirror of `attention_row_avx2`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn attention_row_neon(out: SendPtr, a: Attn, head: usize, i: usize) {
    let Attn { q, k, v, s, qr, kr } = a;
    let hd = s.head_dim;
    let scale = s.scale;
    let qo = i * s.n_head * hd + head * hd;
    let kv_stride = s.n_head_kv * hd;
    let kv_off = (head / (s.n_head / s.n_head_kv)) * hd;
    let (qpos, qseq) = (qr.pos[i], qr.seq_at(i));
    let prefix = if qr.prefix.is_empty() { s.causal_prefix } else { Some(qr.prefix[i]) };

    let mut acc_stack = [0.0f32; ACC_STACK];
    let mut acc_heap = Vec::new();
    let acc: &mut [f32] = if hd <= ACC_STACK {
        &mut acc_stack[..hd]
    } else {
        acc_heap.resize(hd, 0.0);
        &mut acc_heap
    };
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    for j in 0..s.nk {
        if !visible(qpos, kr.pos[j], qseq, kr.seq_at(j), s.sliding_window, prefix) {
            continue;
        }
        let ko = j * kv_stride + kv_off;
        let mut dot_vec = vdupq_n_f32(0.0);
        let mut d = 0;
        while d + 4 <= hd {
            let qv = vld1q_f32(q.as_ptr().add(qo + d));
            let kv = vld1q_f32(k.as_ptr().add(ko + d));
            dot_vec = vfmaq_f32(dot_vec, qv, kv);
            d += 4;
        }
        let mut dot = vaddvq_f32(dot_vec);
        while d < hd {
            dot += q[qo + d] * k[ko + d];
            d += 1;
        }
        let score = dot * scale;

        let new_max = max.max(score);
        let correction = (max - new_max).exp();
        let w = (score - new_max).exp();
        sum = sum * correction + w;

        let cv = vdupq_n_f32(correction);
        let wv = vdupq_n_f32(w);
        let mut d = 0;
        while d + 4 <= hd {
            let av = vld1q_f32(acc.as_ptr().add(d));
            let vv = vld1q_f32(v.as_ptr().add(ko + d));
            // av * correction + w * vv
            vst1q_f32(acc.as_mut_ptr().add(d), vfmaq_f32(vmulq_f32(av, cv), wv, vv));
            d += 4;
        }
        while d < hd {
            acc[d] = acc[d] * correction + w * v[ko + d];
            d += 1;
        }
        max = new_max;
    }

    let invv = vdupq_n_f32(1.0 / sum);
    let mut d = 0;
    while d + 4 <= hd {
        let av = vld1q_f32(acc.as_ptr().add(d));
        // SAFETY: task (head, i) exclusively owns this hd-wide band.
        vst1q_f32(out.0.add(qo + d), vmulq_f32(av, invv));
        d += 4;
    }
    while d < hd {
        out.write(qo + d, acc[d] / sum);
        d += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn attention_row_avx2(out: SendPtr, a: Attn, head: usize, i: usize) {
    let Attn { q, k, v, s, qr, kr } = a;
    let hd = s.head_dim;
    let scale = s.scale;
    let qo = i * s.n_head * hd + head * hd;
    let kv_stride = s.n_head_kv * hd;
    let kv_off = (head / (s.n_head / s.n_head_kv)) * hd;
    let (qpos, qseq) = (qr.pos[i], qr.seq_at(i));
    let prefix = if qr.prefix.is_empty() { s.causal_prefix } else { Some(qr.prefix[i]) };

    let mut acc_stack = [0.0f32; ACC_STACK];
    let mut acc_heap = Vec::new();
    let acc: &mut [f32] = if hd <= ACC_STACK {
        &mut acc_stack[..hd]
    } else {
        acc_heap.resize(hd, 0.0);
        &mut acc_heap
    };
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    for j in 0..s.nk {
        if !visible(qpos, kr.pos[j], qseq, kr.seq_at(j), s.sliding_window, prefix) {
            continue;
        }
        let ko = j * kv_stride + kv_off;
        let mut dot_vec = _mm256_setzero_ps();
        let mut d = 0;
        while d + 8 <= hd {
            let qv = _mm256_loadu_ps(q.as_ptr().add(qo + d));
            let kv = _mm256_loadu_ps(k.as_ptr().add(ko + d));
            dot_vec = _mm256_fmadd_ps(qv, kv, dot_vec);
            d += 8;
        }
        let mut dot = hsum_avx2(dot_vec);
        while d < hd {
            dot += q[qo + d] * k[ko + d];
            d += 1;
        }
        let score = dot * scale;

        let new_max = max.max(score);
        let correction = (max - new_max).exp();
        let w = (score - new_max).exp();
        sum = sum * correction + w;

        let cv = _mm256_set1_ps(correction);
        let wv = _mm256_set1_ps(w);
        let mut d = 0;
        while d + 8 <= hd {
            let av = _mm256_loadu_ps(acc.as_ptr().add(d));
            let vv = _mm256_loadu_ps(v.as_ptr().add(ko + d));
            _mm256_storeu_ps(
                acc.as_mut_ptr().add(d),
                _mm256_fmadd_ps(wv, vv, _mm256_mul_ps(av, cv)),
            );
            d += 8;
        }
        while d < hd {
            acc[d] = acc[d] * correction + w * v[ko + d];
            d += 1;
        }
        max = new_max;
    }

    let inv = _mm256_set1_ps(1.0 / sum);
    let mut d = 0;
    while d + 8 <= hd {
        let av = _mm256_loadu_ps(acc.as_ptr().add(d));
        // SAFETY: task (head, i) exclusively owns this hd-wide band.
        _mm256_storeu_ps(out.0.add(qo + d), _mm256_mul_ps(av, inv));
        d += 8;
    }
    while d < hd {
        out.write(qo + d, acc[d] / sum);
        d += 1;
    }
}

