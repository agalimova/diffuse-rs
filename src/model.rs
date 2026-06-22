//! Diffusion LLM model: config, GGUF loading, forward pass, and sampling.
//!
//! One unified forward pass serves both the full (step 0) and KV-cached
//! (steps 1+) paths: callers describe which positions are recomputed via
//! `ActiveSplit`. Quantized matmuls go through candle's QMatMul; element-wise
//! ops use native AVX2 with scalar fallbacks.

// Re-exported with `pub(crate)` so the submodules below (forward, sampler,
// loader, ops, attention, cache, matmul) inherit the full prelude through a
// single `use super::*;`. This preserves the original flat-module namespace.
pub(crate) use anyhow::{bail, ensure, Context, Result};
pub(crate) use rand::rngs::StdRng;
pub(crate) use rand::{Rng, SeedableRng};
#[cfg(target_arch = "x86_64")]
pub(crate) use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
pub(crate) use std::arch::aarch64::*;

pub(crate) use candle_core::quantized::gguf_file;
pub(crate) use candle_core::quantized::k_quants::{self, GgmlType};
pub(crate) use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
pub(crate) use candle_core::{Device, Module, Tensor};
pub(crate) use rayon::prelude::*;
pub(crate) use std::sync::Arc;

pub(crate) use crate::kernels::{
    self, BlockQ4K, BlockQ6K, BlockQ8K, Q4K_BLOCK_SIZE, Q6K_BLOCK_SIZE, QK_K,
};

mod attention;
mod cache;
mod forward;
mod loader;
mod matmul;
mod ops;
mod sampler;

pub(crate) use attention::*;
pub(crate) use cache::*;
pub(crate) use matmul::*;
pub(crate) use ops::*;
pub(crate) use sampler::*;

// =============================================================================
// Config types
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    Cosine,
    Linear,
}

impl std::str::FromStr for Schedule {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "cosine" => Ok(Self::Cosine),
            "linear" => Ok(Self::Linear),
            other => bail!("unknown schedule: {other} (cosine, linear)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Remasking {
    LowConfidence,
    Random,
    EntropyExit,
    /// Unmask positions with the largest top1-top2 logit gap first.
    Margin,
    /// Parallel decoding (Fast-dLLM style): unmask every position whose top
    /// token probability exceeds `confidence_threshold`. The schedule sets a minimum.
    Confidence,
    /// Entropy-bound adaptive decoding (DiffusionGemma): commit every position
    /// whose entropy is below `eb_entropy_bound`. The rest are re-masked, and
    /// decoding stops early once the canvas stabilizes. The number committed per
    /// step is data-dependent rather than schedule-driven.
    EntropyBound,
}

impl std::str::FromStr for Remasking {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "low_confidence" => Ok(Self::LowConfidence),
            "random" => Ok(Self::Random),
            "entropy_exit" => Ok(Self::EntropyExit),
            "margin" => Ok(Self::Margin),
            "confidence" => Ok(Self::Confidence),
            "entropy_bound" => Ok(Self::EntropyBound),
            other => bail!(
                "unknown remasking: {other} (entropy_exit, low_confidence, margin, confidence, entropy_bound, random)"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SamplerParams {
    pub n_steps: usize,
    pub temperature: f32,
    pub schedule: Schedule,
    pub remasking: Remasking,
    pub seed: u64,
    pub entropy_threshold: f32,
    /// Entropy below which a position is committed under Remasking::EntropyBound.
    pub eb_entropy_bound: f32,
    /// Top-token probability needed to unmask under Remasking::Confidence.
    pub confidence_threshold: f32,
    /// Sample only among the k most likely tokens (0 = disabled).
    pub top_k: usize,
    /// Nucleus sampling: smallest token set with cumulative probability >= p
    /// (1.0 = disabled).
    pub top_p: f32,
    /// Classifier-free guidance: blends a prompt-masked unconditional stream,
    /// logits = uncond + (scale + 1) * (cond - uncond). 0 = off. A nonzero scale doubles cost.
    pub cfg_scale: f32,
    pub use_cache: bool,
    /// Semi-autoregressive block decoding: unmask left-to-right in blocks of
    /// this many tokens (LLaDA-style). None = one block over the whole output.
    pub block_length: Option<usize>,
    /// Truncate output at the first occurrence of this token; once it is
    /// unmasked, remaining blocks are skipped.
    pub eos_token_id: Option<i32>,
    /// Print per-step progress to stderr (CLI on, server off).
    pub progress: bool,
    /// dInfer-style credit decoding: commit a masked position once its
    /// argmax token has been stable for this many consecutive steps
    /// (0 = off). Composes with any remasking strategy.
    pub credit_steps: usize,
    /// dInfer-style iteration smoothing (non-Gemma): mix the previous step's
    /// expected token embedding into each still-masked position's input,
    /// `e_mask + alpha * softmax(logits) W_emb`, ramping alpha 0.1 -> this
    /// value across steps. 0 = off. Gemma uses trained self-conditioning
    /// instead. Decodes more tokens per step; changes outputs.
    pub iter_smooth: f32,
    /// dInfer-style vicinity refresh: also recompute committed positions within
    /// this many tokens of a still-masked position (less stale attention near
    /// the decoding frontier). 0 = off. Only affects the cached path.
    pub vicinity: usize,
    /// Token ids forbidden from being committed during generation, besides the
    /// mask token. Turn and end markers, for example, would truncate the canvas
    /// mid-reasoning. Empty = none.
    pub suppress_ids: Vec<i32>,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            n_steps: 16,
            temperature: 0.0,
            schedule: Schedule::Cosine,
            remasking: Remasking::EntropyExit,
            seed: 42,
            entropy_threshold: 1.5,
            eb_entropy_bound: 0.2,
            confidence_threshold: 0.9,
            top_k: 0,
            top_p: 1.0,
            cfg_scale: 0.0,
            use_cache: true,
            block_length: None,
            eos_token_id: None,
            progress: false,
            credit_steps: 0,
            iter_smooth: 0.0,
            vicinity: 0,
            suppress_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Generation completed (EOS found or all positions unmasked).
    Stop,
    /// Step budget exhausted with masked positions remaining.
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

pub struct GenerateOutput {
    pub tokens: Vec<i32>,
    pub finish_reason: FinishReason,
}

/// One forward pass over several independent sequences concatenated into one
/// flat row buffer. `pos[t]`/`seq[t]` are row t's sequence position and id;
/// block-diagonal attention keeps the sequences mutually invisible. For Gemma,
/// `canvas_start[s]` is sequence s's prompt length and `sc_signal` is the
/// per-sequence self-conditioning signal laid out `[n_seq * n_canvas * n_embd]`.
pub(crate) struct Batch {
    pub(crate) tokens: Vec<i32>,
    pub(crate) pos: Vec<usize>,
    pub(crate) seq: Vec<usize>,
    pub(crate) canvas_start: Vec<usize>,
    pub(crate) n_canvas: usize,
    pub(crate) sc_signal: Option<Vec<f32>>,
}

// =============================================================================
// Model structure
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbdType {
    F16,
    F32,
    Q4K,
    Q6K,
    Q40,
    Q80,
}

/// A tensor's raw bytes inside the memory-mapped GGUF. Weights are never
/// copied into RAM; the OS pages them in on demand (llama.cpp-style mmap).
#[derive(Clone)]
pub(crate) struct Bytes {
    pub(crate) mmap: Arc<memmap2::Mmap>,
    pub(crate) range: std::ops::Range<usize>,
}

impl Bytes {
    fn as_slice(&self) -> &[u8] {
        &self.mmap[self.range.clone()]
    }

    #[cfg(test)]
    fn from_vec(data: Vec<u8>) -> Self {
        let mut m = memmap2::MmapMut::map_anon(data.len().max(1)).unwrap();
        m[..data.len()].copy_from_slice(&data);
        Self { mmap: Arc::new(m.make_read_only().unwrap()), range: 0..data.len() }
    }
}

/// Raw embedding table; rows are dequantized on lookup so the full f32
/// table (n_vocab * n_embd * 4 bytes) never materializes.
struct Embedding {
    bytes: Bytes,
    wtype: EmbdType,
}

pub struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    // QKV biases (Dream/Qwen2.5 only)
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    // Per-head QK-norm weights [head_dim] (LLaDA-MoE/Qwen3-style)
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    /// Gemma post-attention RMSNorm, applied before the attention residual add.
    post_attn_norm: Option<Vec<f32>>,
    qkv: Qkv,
    wo: MatWeight,
    ffn: Ffn,
}

/// QKV projections: separate tensors, or one fused matrix whose output rows
/// are laid out [q (h*hd) | k (hkv*hd) | v (hkv*hd)] (BailingMoeV2/LLaDA2).
enum Qkv {
    Split { wq: MatWeight, wk: MatWeight, wv: MatWeight },
    Fused(MatWeight),
}

struct FfnWeights {
    gate: MatWeight,
    up: MatWeight,
    down: MatWeight,
    /// Intermediate (gate/up output) width.
    ff: usize,
}

enum Ffn {
    Dense(FfnWeights),
    Moe(Box<MoeWeights>),
    Gemma(Box<GemmaFfn>),
}

/// DiffusionGemma FFN: a dense shared expert and a fused-expert MoE run in
/// parallel on the same post-attention residual, each with its own pre/post
/// RMSNorms. The two outputs are summed. An outer post-norm, a residual add,
/// and a scalar output scale follow.
struct GemmaFfn {
    dense: FfnWeights,
    dense_post_norm: Vec<f32>, // post_ffw_norm_1
    moe_pre_norm: Vec<f32>,    // pre_ffw_norm_2
    /// Router [n_expert, n_embd] and its per-channel input scale [n_embd].
    router: Vec<f32>,
    router_scale: Vec<f32>,
    /// Fused gate+up experts [n_expert, 2*ff_exp, n_embd]; gate = first ff_exp
    /// rows, up = next ff_exp rows.
    gate_up: ExpertMat,
    down: ExpertMat,
    /// Per-expert down-projection output scale [n_expert].
    down_scale: Vec<f32>,
    moe_post_norm: Vec<f32>, // post_ffw_norm_2
    post_norm: Vec<f32>,     // post_ffw_norm (outer)
    out_scale: f32,          // layer_output_scale
    cfg: MoeConfig,
    ff_exp: usize,
}

/// DiffusionGemma self-conditioning: the previous step's canvas logits are
/// soft-embedded and passed through a gated MLP, then added to the canvas
/// token embeddings so each step refines on the last one's prediction.
struct SelfCond {
    pre_norm: Vec<f32>,
    mlp: FfnWeights, // gate/up -> GeGLU -> down (n_embd -> n_ff -> n_embd)
}

/// Top tokens of the previous-step distribution kept for the soft embedding.
/// The full vocab-wide weighted sum is unaffordable on CPU. The softmax is
/// peaked, and the downstream RMS-norm absorbs the dropped tail mass.
const SC_TOPK: usize = 128;

/// Mixture-of-experts FFN: a router picks `n_used` of `n_expert` experts per
/// token and blends their outputs. Optional shared experts run on every token.
pub(crate) struct MoeWeights {
    /// Router matrix [n_expert, n_embd], dequantized (it is tiny).
    router: Vec<f32>,
    /// DeepSeek-V3-style selection bias: added to gating scores when picking
    /// the top-k experts, but not used in the blend weights.
    sel_bias: Option<Vec<f32>>,
    gate: ExpertMat,
    up: ExpertMat,
    down: ExpertMat,
    shared: Option<FfnWeights>,
    cfg: MoeConfig,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeConfig {
    n_expert: usize,
    n_used: usize,
    /// Renormalize the selected experts' routing weights to sum to 1.
    weights_norm: bool,
    /// llama.cpp expert_gating_func: 1 = softmax (default), 2 = sigmoid.
    gating_sigmoid: bool,
    /// Multiplier on routing weights (llama.cpp expert_weights_scale).
    scale: f32,
    /// DeepSeek-V3-style grouped routing: experts split into n_group groups;
    /// the top group_used groups (by sum of top-2 scores) are eligible.
    n_group: usize,
    group_used: usize,
}


/// Attention head geometry, shared by RoPE/GQA/attention helpers.
#[derive(Clone, Copy)]
pub struct HeadShape {
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
}

/// Per-layer attention configuration. Uniform models repeat one config;
/// Gemma-style models alternate sliding (narrow head, windowed) and global
/// (wide head, full-attention) layers with different RoPE.
#[derive(Clone)]
struct LayerAttn {
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    rope_dim: usize,
    /// Symmetric attention band (bidirectional sliding window); None = full.
    sliding_window: Option<usize>,
    /// RoPE sin/cos table for this layer's (rope_dim, theta); shared across
    /// layers with identical config.
    rope: Arc<Vec<(f32, f32)>>,
}

impl LayerAttn {
    fn q_stride(&self) -> usize {
        self.n_head * self.head_dim
    }
    fn kv_stride(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
    fn fused_stride(&self) -> usize {
        (self.n_head + 2 * self.n_head_kv) * self.head_dim
    }
    fn shape(&self) -> HeadShape {
        HeadShape {
            n_head: self.n_head,
            n_head_kv: self.n_head_kv,
            head_dim: self.head_dim,
        }
    }
}

/// Hard ceiling on prompt + generated positions. Bounds both the RoPE table
/// size at load and per-request allocation, so a GGUF without declared
/// `context_length` (or a hostile `max_tokens`) cannot drive an unbounded
/// allocation. Larger than any diffusion context in practice.
pub(crate) const MAX_TOTAL_LEN: usize = 32768;

pub struct Model {
    /// Arch id: "llada", "llada-moe", "llada2", "dream", "mdlm", "diffusion-gemma".
    /// Read by the server (chat templates, /v1/model); the CLI logs it at load.
    #[cfg_attr(not(feature = "server"), allow(dead_code))]
    pub model_type: String,
    /// Dream predicts the token at position i from the logits at row i-1
    /// (AR-style shift); LLaDA reads row i directly.
    pub logit_shift: bool,
    pub n_vocab: usize,
    pub n_embd: usize,
    /// Read by the server's /v1/model endpoint; per-layer geometry lives in `attn`.
    #[cfg_attr(not(feature = "server"), allow(dead_code))]
    pub n_head: usize,
    pub n_layer: usize,
    pub n_ff: usize,
    pub mask_token_id: u32,
    /// From GGUF metadata if present; used as the server-side default.
    #[cfg_attr(not(feature = "server"), allow(dead_code))]
    pub eos_token_id: Option<i32>,
    pub rms_norm_eps: f32,
    /// Trained context window from GGUF metadata, if declared.
    pub context_length: Option<usize>,
    /// RoPE table capacity. Any absolute position must stay below it, so it is
    /// the true upper bound on prompt + generated length.
    pub max_positions: usize,
    /// Block-diffusion canvas size (Gemma); generations beyond it run in blocks.
    pub canvas_length: Option<usize>,
    /// Per-layer attention geometry (uniform models repeat one entry).
    attn: Vec<LayerAttn>,
    /// DiffusionGemma: per-layer dual FFN, attention scale 1.0, embedding
    /// scaled by sqrt(n_embd) + noscale-norm, and final logit softcap.
    is_gemma: bool,
    /// Embedding multiplier (sqrt(n_embd) for Gemma, else 1.0).
    embed_scale: f32,
    /// Final logit softcap (Gemma 30.0); 0 = disabled.
    logit_softcap: f32,
    /// First generated (canvas) position. Positions below the canvas start are
    /// prompt. The value is set per generation and used only by the Gemma
    /// embedding path.
    canvas_start: usize,
    /// DiffusionGemma self-conditioning weights (None until/unless present).
    self_cond: Option<SelfCond>,
    /// Per-canvas-position signal to add to the embeddings this forward pass
    /// ([n_canvas * n_embd]); set by the sampler each step, None at step 0.
    sc_signal: Option<Vec<f32>>,
    /// IterSmooth signal (non-Gemma): added to still-masked canvas positions
    /// this forward pass ([n_canvas * n_embd]); set by the sampler, None = off.
    smooth_signal: Option<Vec<f32>>,
    /// Widest FFN intermediate across dense/expert/shared paths (buffer sizing;
    /// the profiler's MoE fallback).
    pub(crate) max_ff: usize,
    tok_embd: Embedding,
    output_norm: Vec<f32>,
    output: MatWeight,
    layers: Vec<Layer>,
    bufs: Option<ComputeBuffers>,
}

struct ComputeBuffers {
    cur: Vec<f32>,
    residual: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// Fused QKV projection output before splitting.
    qkv: Vec<f32>,
    k_full: Vec<f32>,
    v_full: Vec<f32>,
    attn_out: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    /// MoE: blended expert outputs per token.
    moe_out: Vec<f32>,
    /// MoE: gathered per-expert activation rows / expert down-proj output.
    moe_act: Vec<f32>,
    /// MoE routing scratch, reused across tokens, layers, and steps.
    route: RouteScratch,
    scratch: ActScratch,
}

impl ComputeBuffers {
    fn new(m: &Model, n: usize) -> Self {
        let (e, ff) = (m.n_embd, m.max_ff);
        // Buffers must hold the widest layer's projections.
        let q_max = m.attn.iter().map(|a| a.q_stride()).max().unwrap();
        let kv_max = m.attn.iter().map(|a| a.kv_stride()).max().unwrap();
        let fused_max = m.attn.iter().map(|a| a.fused_stride()).max().unwrap();
        Self {
            cur: vec![0.0; n * e],
            residual: vec![0.0; n * e],
            normed: vec![0.0; n * e],
            q: vec![0.0; n * q_max],
            k: vec![0.0; n * kv_max],
            v: vec![0.0; n * kv_max],
            qkv: vec![0.0; n * fused_max],
            k_full: vec![0.0; n * kv_max],
            v_full: vec![0.0; n * kv_max],
            attn_out: vec![0.0; n * q_max],
            gate: vec![0.0; n * ff],
            up: vec![0.0; n * ff],
            moe_out: vec![0.0; n * e],
            moe_act: vec![0.0; n * e],
            route: RouteScratch::default(),
            scratch: ActScratch::default(),
        }
    }
}



// Tests live in src/model/tests.rs
#[cfg(test)]
mod tests;
