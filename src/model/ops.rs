use super::*;

// =============================================================================
// Element-wise ops (RMSNorm, softmax, RoPE, SiLU)
// =============================================================================

pub(crate) fn rms_norm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { rms_norm_avx2(out, x, w, eps) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    return unsafe { rms_norm_neon(out, x, w, eps) };
    #[cfg(not(target_arch = "aarch64"))]
    rms_norm_scalar(out, x, w, eps)
}

/// NEON RMSNorm; width-4 mirror of `rms_norm_avx2`. NEON is baseline on
/// aarch64, so no runtime feature check is needed.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn rms_norm_neon(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f64;
    let mut i = 0;
    while i + 4 <= n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let sq = vmulq_f32(xv, xv);
        let mut tmp = [0.0f32; 4];
        vst1q_f32(tmp.as_mut_ptr(), sq);
        ss += tmp.iter().map(|&t| t as f64).sum::<f64>();
        i += 4;
    }
    while i < n {
        ss += (x[i] * x[i]) as f64;
        i += 1;
    }
    let scale = 1.0 / ((ss / n as f64 + eps as f64).sqrt() as f32);
    let sv = vdupq_n_f32(scale);
    i = 0;
    while i + 4 <= n {
        let xv = vld1q_f32(x.as_ptr().add(i));
        let wv = vld1q_f32(w.as_ptr().add(i));
        vst1q_f32(out.as_mut_ptr().add(i), vmulq_f32(vmulq_f32(xv, sv), wv));
        i += 4;
    }
    while i < n {
        out[i] = x[i] * scale * w[i];
        i += 1;
    }
}

// Portable reference path; unused in the aarch64 bin (NEON is unconditional)
// but still exercised by tests and used on x86/other targets.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn rms_norm_scalar(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    // Match GGML exactly: f32 multiply then cast to f64 for accumulation
    let ss: f64 = x.iter().map(|&v| (v * v) as f64).sum();
    let scale = 1.0 / ((ss / n as f64 + eps as f64).sqrt() as f32);
    for i in 0..n {
        out[i] = x[i] * scale * w[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn rms_norm_avx2(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let n = x.len();
    // Match GGML: accumulate f32 products into f64
    let mut ss = 0.0f64;
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let sq = _mm256_mul_ps(xv, xv);
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), sq);
        ss += tmp.iter().map(|&t| t as f64).sum::<f64>();
        i += 8;
    }
    while i < n {
        ss += (x[i] * x[i]) as f64;
        i += 1;
    }

    let scale = 1.0 / ((ss / n as f64 + eps as f64).sqrt() as f32);
    let sv = _mm256_set1_ps(scale);
    i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        _mm256_storeu_ps(
            out.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_mul_ps(xv, sv), wv),
        );
        i += 8;
    }
    while i < n {
        out[i] = x[i] * scale * w[i];
        i += 1;
    }
}

/// In-place RMSNorm with weight (used for per-head QK-norm).
pub(crate) fn rms_norm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let n = x.len();
    let ss: f64 = x.iter().map(|&v| (v * v) as f64).sum();
    let scale = 1.0 / ((ss / n as f64 + eps as f64).sqrt() as f32);
    for i in 0..n {
        x[i] = x[i] * scale * w[i];
    }
}

/// RMSNorm with no learned weight (Gemma rms_norm_noscale): just normalize.
pub(crate) fn rms_norm_noscale(x: &mut [f32], eps: f32) {
    let n = x.len();
    let ss: f64 = x.iter().map(|&v| (v * v) as f64).sum();
    let scale = 1.0 / ((ss / n as f64 + eps as f64).sqrt() as f32);
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// gelu tanh approximation (gelu_pytorch_tanh), matching ggml LLM_FFN_GELU.
#[inline]
pub(crate) fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044_715 * x * x * x)).tanh())
}

/// GeGLU: gate[i] = gelu_tanh(gate[i]) * up[i].
pub(crate) fn geglu_mul(gate: &mut [f32], up: &[f32]) {
    for i in 0..gate.len() {
        gate[i] = gelu_tanh(gate[i]) * up[i];
    }
}

/// RMSNorm (weighted) each of the first `n` rows of `buf` in place, row width `e`.
pub(crate) fn norm_rows_inplace(buf: &mut [f32], w: &[f32], n: usize, e: usize, eps: f32) {
    for row in buf[..n * e].chunks_mut(e) {
        rms_norm_inplace(row, w, eps);
    }
}

/// Softmax-gate one token over `cfg.n_expert` experts and return the chosen
/// Reusable buffers for MoE routing. Scoring and ranking every expert for
/// every token used to allocate several small Vecs per token per layer per
/// step; the scratch lives in ComputeBuffers and is reused across all of them.
#[derive(Default)]
pub(crate) struct RouteScratch {
    scores: Vec<f32>,
    ranked: Vec<usize>,
    /// One token's chosen (expert, weight) pairs (output of the route fns).
    pub(crate) picks: Vec<(usize, f32)>,
    /// Per-expert routed (token, weight) lists, cleared each pass.
    pub(crate) routes: Vec<Vec<(usize, f32)>>,
}

impl RouteScratch {
    /// Clear per-pass state and size the per-expert lists.
    pub(crate) fn begin(&mut self, n_expert: usize) {
        if self.routes.len() != n_expert {
            self.routes.resize_with(n_expert, Vec::new);
        }
        for r in &mut self.routes {
            r.clear();
        }
    }
}

/// Fill `rs.picks` with (expert, weight) pairs: top-k by score, optionally
/// renormalized to sum 1. `router` is [n_expert, e] row-major; `input` is the
/// e-wide router input.
pub(crate) fn select_top_experts(router: &[f32], input: &[f32], cfg: MoeConfig, rs: &mut RouteScratch) {
    let e = input.len();
    rs.scores.clear();
    rs.scores.extend(
        (0..cfg.n_expert)
            .map(|ex| router[ex * e..(ex + 1) * e].iter().zip(input).map(|(w, x)| w * x).sum::<f32>()),
    );
    softmax_row_scalar(&mut rs.scores);

    rs.ranked.clear();
    rs.ranked.extend(0..cfg.n_expert);
    let scores = &rs.scores;
    rs.ranked.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let chosen = &rs.ranked[..cfg.n_used.min(cfg.n_expert)];
    let norm = if cfg.weights_norm {
        chosen.iter().map(|&ex| scores[ex]).sum::<f32>().max(1e-12)
    } else {
        1.0
    };
    rs.picks.clear();
    rs.picks.extend(chosen.iter().map(|&ex| (ex, scores[ex] / norm)));
}

/// Route one token through the general MoE gate into `rs.picks`: score every
/// expert, apply sigmoid or softmax gating, optional DeepSeek-V3 selection
/// bias and grouped selection, then top-k with optional weight
/// renormalization and scale.
pub(crate) fn route_token_moe(m: &MoeWeights, input: &[f32], rs: &mut RouteScratch) {
    let (cfg, e) = (m.cfg, input.len());
    rs.scores.clear();
    rs.scores.extend(
        (0..cfg.n_expert)
            .map(|ex| m.router[ex * e..(ex + 1) * e].iter().zip(input).map(|(w, x)| w * x).sum::<f32>()),
    );
    if cfg.gating_sigmoid {
        for s in rs.scores.iter_mut() {
            *s = 1.0 / (1.0 + (-*s).exp());
        }
    } else {
        softmax_row_scalar(&mut rs.scores);
    }

    // Selection may be biased (DeepSeek-V3); the bias ranks but does not weight.
    let scores = &rs.scores;
    let sel = |ex: usize| scores[ex] + m.sel_bias.as_ref().map_or(0.0, |b| b[ex]);
    eligible_experts(cfg, &sel, &mut rs.ranked);
    rs.ranked.sort_unstable_by(|&a, &b| sel(b).total_cmp(&sel(a)));
    let chosen = &rs.ranked[..cfg.n_used.min(rs.ranked.len())];
    let norm = if cfg.weights_norm {
        chosen.iter().map(|&ex| scores[ex]).sum::<f32>().max(1e-12)
    } else {
        1.0
    };
    rs.picks.clear();
    rs.picks.extend(chosen.iter().map(|&ex| (ex, scores[ex] / norm * cfg.scale)));
}

/// Fill `ranked` with the experts eligible for selection. Without grouped
/// routing, the set is all experts. Under grouped routing, the set is only
/// those in the top `group_used` groups. Each group is ranked by the sum of
/// its two highest scores (DeepSeek-V3 group selection).
fn eligible_experts(cfg: MoeConfig, sel: &impl Fn(usize) -> f32, ranked: &mut Vec<usize>) {
    ranked.clear();
    if cfg.n_group <= 1 {
        ranked.extend(0..cfg.n_expert);
        return;
    }
    let group_size = cfg.n_expert / cfg.n_group;
    let mut groups: Vec<(f32, usize)> = (0..cfg.n_group)
        .map(|g| {
            let mut top2 = [f32::NEG_INFINITY; 2];
            for ex in g * group_size..(g + 1) * group_size {
                let s = sel(ex);
                if s > top2[0] {
                    top2[1] = top2[0];
                    top2[0] = s;
                } else if s > top2[1] {
                    top2[1] = s;
                }
            }
            (top2[0] + top2[1], g)
        })
        .collect();
    groups.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked.extend(
        groups[..cfg.group_used.min(cfg.n_group)]
            .iter()
            .flat_map(|&(_, g)| g * group_size..(g + 1) * group_size),
    );
}

/// Softmax a row in place (router gating; also the test reference for the
/// fused online-softmax attention).
pub(crate) fn softmax_row_scalar(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in row.iter_mut() {
        *v *= inv;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn silu_mul_avx2(gate: &mut [f32], up: &[f32]) {
    let len = gate.len();
    let ones = _mm256_set1_ps(1.0);
    let mut i = 0;
    while i + 8 <= len {
        let gv = _mm256_loadu_ps(gate.as_ptr().add(i));
        let uv = _mm256_loadu_ps(up.as_ptr().add(i));
        let neg = _mm256_sub_ps(_mm256_setzero_ps(), gv);
        let mut exp_vals = [0.0f32; 8];
        _mm256_storeu_ps(exp_vals.as_mut_ptr(), neg);
        for v in &mut exp_vals {
            *v = v.exp();
        }
        let denom = _mm256_add_ps(ones, _mm256_loadu_ps(exp_vals.as_ptr()));
        _mm256_storeu_ps(
            gate.as_mut_ptr().add(i),
            _mm256_mul_ps(_mm256_div_ps(gv, denom), uv),
        );
        i += 8;
    }
    while i < len {
        gate[i] = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
        i += 1;
    }
}

pub(crate) fn silu_mul(gate: &mut [f32], up: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { silu_mul_avx2(gate, up) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    return unsafe { silu_mul_neon(gate, up) };
    #[cfg(not(target_arch = "aarch64"))]
    for i in 0..gate.len() {
        gate[i] = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
    }
}

/// NEON SwiGLU; width-4 mirror of `silu_mul_avx2`. The exp() is still scalar
/// (no NEON vectorized exp in std), matching the AVX2 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn silu_mul_neon(gate: &mut [f32], up: &[f32]) {
    let len = gate.len();
    let ones = vdupq_n_f32(1.0);
    let mut i = 0;
    while i + 4 <= len {
        let gv = vld1q_f32(gate.as_ptr().add(i));
        let uv = vld1q_f32(up.as_ptr().add(i));
        let neg = vnegq_f32(gv);
        let mut exp_vals = [0.0f32; 4];
        vst1q_f32(exp_vals.as_mut_ptr(), neg);
        for v in &mut exp_vals {
            *v = v.exp();
        }
        let denom = vaddq_f32(ones, vld1q_f32(exp_vals.as_ptr()));
        vst1q_f32(gate.as_mut_ptr().add(i), vmulq_f32(vdivq_f32(gv, denom), uv));
        i += 4;
    }
    while i < len {
        gate[i] = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
        i += 1;
    }
}

// AVX2 reduction helpers
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn hsum_avx2(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s, _mm_shuffle_ps::<0b10_11_00_01>(s, s));
    let s3 = _mm_add_ps(s2, _mm_shuffle_ps::<0b01_00_11_10>(s2, s2));
    _mm_cvtss_f32(s3)
}

/// Element-wise addition: dst[i] += src[i].
pub(crate) fn residual_add(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

/// Add a bias vector to each of n rows. No-op if bias is None.
pub(crate) fn add_bias(data: &mut [f32], bias: Option<&[f32]>, n: usize) {
    if let Some(b) = bias {
        let stride = b.len();
        for t in 0..n {
            for i in 0..stride {
                data[t * stride + i] += b[i];
            }
        }
    }
}

/// RMSNorm every head_dim-wide group of x with the same weight (QK-norm).
pub(crate) fn norm_heads(x: &mut [f32], w: &[f32], eps: f32) {
    for head in x.chunks_mut(w.len()) {
        rms_norm_inplace(head, w, eps);
    }
}

/// RoPE sin/cos table: rope_dim/2 entries per position.
pub(crate) fn build_rope_cache(max_seq_len: usize, rope_dim: usize, theta: f32) -> Vec<(f32, f32)> {
    build_rope_cache_ff(max_seq_len, rope_dim, theta, None)
}

/// RoPE table with optional per-frequency divisors (Gemma rope_freqs /
/// llama.cpp freq_factors): effective freq_i = base_i / freq_factors[i].
pub(crate) fn build_rope_cache_ff(
    max_seq_len: usize,
    rope_dim: usize,
    theta: f32,
    freq_factors: Option<&[f32]>,
) -> Vec<(f32, f32)> {
    let half = rope_dim / 2;
    let mut cache = Vec::with_capacity(max_seq_len * half);
    for t in 0..max_seq_len {
        for i in 0..half {
            let mut freq = 1.0 / theta.powf(2.0 * i as f32 / rope_dim as f32);
            if let Some(ff) = freq_factors {
                freq /= ff[i];
            }
            cache.push((t as f32 * freq).sin_cos());
        }
    }
    cache
}

/// Apply RoPE to Q and K rows using the original sequence position of each
/// row. Only the first `rot` dims of each head rotate (partial rotary).
pub(crate) fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    rope: &[(f32, f32)],
    positions: &[usize],
    hs: HeadShape,
    rot: usize,
) {
    rotate_rows(q, rope, positions, hs.n_head, hs.head_dim, rot);
    rotate_rows(k, rope, positions, hs.n_head_kv, hs.head_dim, rot);
}

pub(crate) fn rotate_rows(
    x: &mut [f32],
    rope: &[(f32, f32)],
    positions: &[usize],
    heads: usize,
    hd: usize,
    rot: usize,
) {
    let half = rot / 2;
    for (t, &pos) in positions.iter().enumerate() {
        for head in 0..heads {
            let off = t * heads * hd + head * hd;
            for i in 0..half {
                let (sin, cos) = rope[pos * half + i];
                let (a, b) = (x[off + i], x[off + half + i]);
                x[off + i] = a * cos - b * sin;
                x[off + half + i] = a * sin + b * cos;
            }
        }
    }
}

