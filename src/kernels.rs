//! Quantized block types and kernels for Q4_K, Q6_K, Q8_K.
//!
//! Block layouts match GGML/candle. Production matmuls run through
//! `model::matmul::native_matmul` over candle's SIMD vec_dot; the scalar
//! kernels here serve the profiler and the cross-validation tests.

// -- Q4_K_M block layout (256 weights -> 144 bytes) --------------------------

// No `packed`. This matches candle's #[repr(C)] BlockQ4K. Fields are naturally aligned.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ4K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

pub const QK_K: usize = 256;

// -- Q6_K block layout (256 weights -> 210 bytes) ----------------------------

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],      // lower 4 bits of quants (128 bytes)
    pub qh: [u8; QK_K / 4],      // upper 2 bits of quants (64 bytes)
    pub scales: [i8; QK_K / 16], // scales (16 bytes)
    pub d: u16,                  // super-block scale, f16 (2 bytes)
}

pub const Q6K_BLOCK_SIZE: usize = std::mem::size_of::<BlockQ6K>(); // 210
pub const Q4K_BLOCK_SIZE: usize = std::mem::size_of::<BlockQ4K>();

#[repr(C)]
#[derive(Clone)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

// -- f16 conversion -----------------------------------------------------------

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: value = mant * 2^-24. Normalize so bit 10 is set; the
        // resulting f32 exponent is -14 - e (biased: 113 - e).
        let mut e = 0u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        return f32::from_bits((sign << 31) | ((113 - e) << 23) | ((m & 0x3FF) << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | if mant == 0 { 0 } else { 0x7FFFFF });
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
}

// -- Quantize f32 -> Q8_K ----------------------------------------------------

#[inline]
pub fn quantize_row_q8_k_into(x: &[f32], blocks: &mut Vec<BlockQ8K>) {
    blocks.clear();
    quantize_row_q8_k_append(x, blocks);
}

/// Quantize n rows of `cols` f32s each into row-major Q8_K blocks.
pub fn quantize_rows_q8_k(src: &[f32], n: usize, cols: usize, dst: &mut Vec<BlockQ8K>) {
    dst.clear();
    dst.reserve(n * cols / QK_K);
    for t in 0..n {
        quantize_row_q8_k_append(&src[t * cols..(t + 1) * cols], dst);
    }
}

fn quantize_row_q8_k_append(x: &[f32], blocks: &mut Vec<BlockQ8K>) {
    let nb = x.len() / QK_K;
    blocks.reserve(nb);

    // Extracted verbatim from candle 0.8.4 k_quants.rs BlockQ8K::from_float
    for i in 0..nb {
        let xs = &x[i * QK_K..(i + 1) * QK_K];
        let mut max = 0f32;
        let mut amax = 0f32;
        for &x in xs.iter() {
            if amax < x.abs() {
                amax = x.abs();
                max = x;
            }
        }
        let mut block = BlockQ8K {
            d: 0.0,
            qs: [0i8; QK_K],
            bsums: [0i16; QK_K / 16],
        };
        if amax == 0f32 {
            block.d = 0f32;
        } else {
            let iscale = -128f32 / max;
            for (j, q) in block.qs.iter_mut().enumerate() {
                let v = (iscale * xs[j]).round();
                *q = v.min(127.) as i8;
            }
            for j in 0..QK_K / 16 {
                let mut sum = 0i32;
                for ii in 0..16 {
                    sum += block.qs[j * 16 + ii] as i32;
                }
                block.bsums[j] = sum as i16;
            }
            block.d = 1.0 / iscale;
        }
        blocks.push(block);
    }
}

/// Dequantize a single Q6_K block row to f32.
pub fn dequantize_q6k_row(blocks: &[BlockQ6K], out: &mut [f32]) {
    let mut pos = 0;
    for block in blocks {
        let d = f16_to_f32(block.d);

        // Reconstruct 6-bit values
        let mut aux = [0i8; QK_K];
        for l in 0..32 {
            aux[l] = ((block.ql[l] & 0xF) | ((block.qh[l] & 3) << 4)) as i8 - 32;
            aux[l + 32] = ((block.ql[l + 32] & 0xF) | (((block.qh[l] >> 2) & 3) << 4)) as i8 - 32;
            aux[l + 64] = ((block.ql[l] >> 4) | (((block.qh[l] >> 4) & 3) << 4)) as i8 - 32;
            aux[l + 96] = ((block.ql[l + 32] >> 4) | (((block.qh[l] >> 6) & 3) << 4)) as i8 - 32;
        }
        for l in 0..32 {
            aux[l + 128] =
                ((block.ql[l + 64] & 0xF) | ((block.qh[l + 32] & 3) << 4)) as i8 - 32;
            aux[l + 160] =
                ((block.ql[l + 64 + 32] & 0xF) | (((block.qh[l + 32] >> 2) & 3) << 4)) as i8 - 32;
            aux[l + 192] =
                ((block.ql[l + 64] >> 4) | (((block.qh[l + 32] >> 4) & 3) << 4)) as i8 - 32;
            aux[l + 224] =
                ((block.ql[l + 64 + 32] >> 4) | (((block.qh[l + 32] >> 6) & 3) << 4)) as i8 - 32;
        }

        for j in 0..(QK_K / 16) {
            let sc = d * block.scales[j] as f32;
            for k in 0..16 {
                out[pos + j * 16 + k] = sc * aux[j * 16 + k] as f32;
            }
        }
        pos += QK_K;
    }
}

/// Dequantize a row of Q4_0 blocks (32 weights -> 18 bytes: f16 scale + 16
/// nibble bytes; low nibbles are elements 0..16, high nibbles 16..32).
pub fn dequantize_q4_0_row(bytes: &[u8], out: &mut [f32]) {
    for (block, chunk) in bytes.chunks_exact(18).zip(out.chunks_exact_mut(32)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (i, &q) in block[2..18].iter().enumerate() {
            chunk[i] = d * ((q & 0x0F) as f32 - 8.0);
            chunk[i + 16] = d * ((q >> 4) as f32 - 8.0);
        }
    }
}

/// Dequantize a Q8_0 row (34-byte blocks: f16 scale + 32 signed int8) to f32.
pub fn dequantize_q8_0_row(bytes: &[u8], out: &mut [f32]) {
    for (block, chunk) in bytes.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (i, &q) in block[2..34].iter().enumerate() {
            chunk[i] = d * (q as i8) as f32;
        }
    }
}

/// Dequantize a single Q4_K block row to f32.
pub fn dequantize_q4k_row(blocks: &[BlockQ4K], out: &mut [f32]) {
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let mut pos = 0;
    for block in blocks {
        let d = f16_to_f32(block.d);
        let dmin = f16_to_f32(block.dmin);
        let mut utmp = [0u32; 4];
        unsafe {
            std::ptr::copy_nonoverlapping(block.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
        }
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;
        let scales = unsafe { std::slice::from_raw_parts(utmp.as_ptr() as *const u8, 16) };
        for j in 0..QK_K / 64 {
            let (sc, mn) = (scales[2 * j] as f32, scales[2 * j + 8] as f32);
            let (sc2, mn2) = (scales[2 * j + 1] as f32, scales[2 * j + 1 + 8] as f32);
            for k in 0..32 {
                let q = block.qs[j * 32 + k];
                out[pos + k] = d * sc * (q & 0x0F) as f32 - dmin * mn;
                out[pos + 32 + k] = d * sc2 * ((q >> 4) & 0x0F) as f32 - dmin * mn2;
            }
            pos += 64;
        }
    }
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent scalar reference for Q4_K x Q8_K, extracted verbatim from
    /// candle 0.8.4 k_quants.rs BlockQ4K::vec_dot_unopt. Test-only: pins
    /// dequantize_q4k_row against a second implementation.
    fn vec_dot_q4k_q8k(xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let mut utmp: [u32; 4] = [0; 4];
        let mut scales: [u8; 8] = [0; 8];
        let mut mins: [u8; 8] = [0; 8];
        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut sums: [f32; 8] = [0.0; 8];
        let mut aux32: [i32; 8] = [0; 8];

        let mut sumf = 0.0f32;
        for (y, x) in ys.iter().zip(xs.iter()) {
            let q4 = &x.qs;
            let q8 = &y.qs;
            aux32.fill(0);

            // Extract nibbles as unsigned 0..15
            let mut a_off = 0;
            let mut q4_off = 0;
            for _ in 0..QK_K / 64 {
                for l in 0..32 {
                    aux8[a_off + l] = (q4[q4_off + l] & 0xF) as i8;
                }
                a_off += 32;
                for l in 0..32 {
                    aux8[a_off + l] = (q4[q4_off + l] >> 4) as i8;
                }
                a_off += 32;
                q4_off += 32;
            }

            // Unpack scales
            unsafe {
                std::ptr::copy_nonoverlapping(x.scales.as_ptr(), utmp.as_mut_ptr() as *mut u8, 12);
            }
            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            // Split into scales[8] and mins[8]
            unsafe {
                std::ptr::copy_nonoverlapping(utmp.as_ptr() as *const u8, scales.as_mut_ptr(), 8);
                std::ptr::copy_nonoverlapping(
                    (utmp.as_ptr() as *const u8).add(8),
                    mins.as_mut_ptr(),
                    8,
                );
            }

            // Mins term: all 16 bsums, each min covers 2 consecutive bsums
            let mut sumi = 0i32;
            for j in 0..QK_K / 16 {
                sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
            }

            // Main term: 8-lane parallel accumulation
            let mut a_off = 0;
            let mut q8_off = 0;
            for &scale in scales.iter() {
                let scale = scale as i32;
                for _ in 0..4 {
                    for l in 0..8 {
                        aux16[l] = q8[q8_off + l] as i16 * aux8[a_off + l] as i16;
                    }
                    for l in 0..8 {
                        aux32[l] += scale * aux16[l] as i32;
                    }
                    q8_off += 8;
                    a_off += 8;
                }
            }

            let d = f16_to_f32(x.d) * y.d;
            for l in 0..8 {
                sums[l] += d * aux32[l] as f32;
            }
            let dmin = f16_to_f32(x.dmin) * y.d;
            sumf -= dmin * sumi as f32;
        }
        sumf + sums.iter().sum::<f32>()
    }

    /// Independent scalar reference for Q6_K x Q8_K, ported from GGML's
    /// ggml_vec_dot_q6_K_q8_K_generic. Test-only: it pins dequantize_q6k_row
    /// against a second implementation.
    fn vec_dot_q6k_q8k(x_blocks: &[BlockQ6K], y_blocks: &[BlockQ8K]) -> f32 {
        let mut sumf = 0.0f32;
        for (x, y) in x_blocks.iter().zip(y_blocks) {
            let d = f16_to_f32(x.d) * y.d;

            // Reconstruct 6-bit values: lower 4 bits from ql, upper 2 from qh.
            let mut aux8 = [0i8; QK_K];
            for l in 0..32 {
                aux8[l] = ((x.ql[l] & 0xF) | ((x.qh[l] & 3) << 4)) as i8 - 32;
                aux8[l + 32] = ((x.ql[l + 32] & 0xF) | (((x.qh[l] >> 2) & 3) << 4)) as i8 - 32;
                aux8[l + 64] = ((x.ql[l] >> 4) | (((x.qh[l] >> 4) & 3) << 4)) as i8 - 32;
                aux8[l + 96] = ((x.ql[l + 32] >> 4) | (((x.qh[l] >> 6) & 3) << 4)) as i8 - 32;
            }
            for l in 0..32 {
                aux8[l + 128] = ((x.ql[l + 64] & 0xF) | ((x.qh[l + 32] & 3) << 4)) as i8 - 32;
                aux8[l + 160] =
                    ((x.ql[l + 64 + 32] & 0xF) | (((x.qh[l + 32] >> 2) & 3) << 4)) as i8 - 32;
                aux8[l + 192] = ((x.ql[l + 64] >> 4) | (((x.qh[l + 32] >> 4) & 3) << 4)) as i8 - 32;
                aux8[l + 224] =
                    ((x.ql[l + 64 + 32] >> 4) | (((x.qh[l + 32] >> 6) & 3) << 4)) as i8 - 32;
            }

            let mut sumi = 0i32;
            for j in 0..(QK_K / 16) {
                let sc = x.scales[j] as i32;
                for k in 0..16 {
                    sumi += sc * aux8[j * 16 + k] as i32 * y.qs[j * 16 + k] as i32;
                }
            }
            sumf += d * sumi as f32;
        }
        sumf
    }

    #[test]
    fn test_f16_to_f32() {
        // Reference values from numpy float16
        let cases: [(u16, f32); 8] = [
            (0x0000, 0.0),
            (0x0001, 5.960_464_5e-8),  // smallest subnormal
            (0x0200, 3.051_757_8e-5),  // mid subnormal
            (0x03FF, 6.097_555e-5),  // largest subnormal
            (0x0400, 6.103_515_6e-5),  // smallest normal
            (0x3C00, 1.0),
            (0xC000, -2.0),
            (0x8001, -5.960_464_5e-8),
        ];
        for (h, want) in cases {
            let got = f16_to_f32(h);
            assert!(
                (got - want).abs() <= want.abs() * 1e-6,
                "f16 {h:#06x}: got {got:e}, want {want:e}"
            );
        }
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_block_sizes() {
        assert_eq!(std::mem::size_of::<BlockQ4K>(), 144);
        assert_eq!(std::mem::size_of::<BlockQ6K>(), 210);
        assert_eq!(std::mem::size_of::<BlockQ8K>(), 292);
    }

    #[test]
    fn test_dequantize_q8_0_row() {
        // Two 34-byte blocks (64 elems): f16 scale 0.5 (0x3800), int8 quants.
        let mut bytes = Vec::new();
        for blk in 0..2i32 {
            bytes.extend_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
            for i in 0..32i32 {
                bytes.push(((blk * 32 + i - 40) as i8) as u8);
            }
        }
        let mut out = vec![0.0f32; 64];
        dequantize_q8_0_row(&bytes, &mut out);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.5 * (i as i32 - 40) as f32);
        }
    }

    #[test]
    fn test_q6k_vs_dequantize() {
        let mut block = BlockQ6K {
            ql: [0u8; QK_K / 2], qh: [0u8; QK_K / 4],
            scales: [0i8; QK_K / 16], d: 0x3C00,
        };
        for i in 0..16 { block.scales[i] = i as i8 + 1; }
        for i in 0..128 { block.ql[i] = ((i * 7 + 3) & 0xFF) as u8; }
        for i in 0..64 { block.qh[i] = ((i * 13 + 5) & 0xFF) as u8; }

        let mut q8_qs = [0i8; QK_K];
        for (i, q) in q8_qs.iter_mut().enumerate() {
            *q = ((i as i32 * 3 - 128).clamp(-128, 127)) as i8;
        }
        let mut q8_bsums = [0i16; QK_K / 16];
        for j in 0..16 { q8_bsums[j] = q8_qs[j*16..(j+1)*16].iter().map(|&v| v as i16).sum(); }
        let q8 = BlockQ8K { d: 1.0 / 127.0, qs: q8_qs, bsums: q8_bsums };

        let dot = vec_dot_q6k_q8k(&[block], &[q8]);
        let mut deq = vec![0.0f32; QK_K];
        dequantize_q6k_row(&[block], &mut deq);
        let dot_ref: f32 = deq.iter().enumerate().map(|(i, &v)| v * q8_qs[i] as f32 / 127.0).sum();

        let diff = (dot - dot_ref).abs();
        assert!(diff / dot.abs().max(1e-6) < 1e-4, "Q6K mismatch: {dot:.4} vs {dot_ref:.4}");
    }

    #[test]
    fn test_q4k_nonzero() {
        let mut block = BlockQ4K { d: 0x3C00, dmin: 0x3800, scales: [0u8; 12], qs: [0u8; 128] };
        for i in 0..12 { block.scales[i] = ((i * 5 + 1) & 0x3F) as u8; }
        for i in 0..128 { block.qs[i] = ((i * 7 + 3) & 0xFF) as u8; }

        let mut q8 = BlockQ8K { d: 1.0 / 127.0, qs: [0i8; QK_K], bsums: [0i16; QK_K / 16] };
        for i in 0..256 { q8.qs[i] = ((i as i32 * 3 - 128).clamp(-128, 127)) as i8; }
        for j in 0..16 { q8.bsums[j] = q8.qs[j*16..(j+1)*16].iter().map(|&v| v as i16).sum(); }

        let result = vec_dot_q4k_q8k(&[block], &[q8]);
        assert!(result.is_finite(), "Q4K dot not finite: {result}");
        assert!(result.abs() > 1e-6, "Q4K dot suspiciously zero: {result}");
    }
}
