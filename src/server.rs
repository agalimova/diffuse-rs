//! HTTP API server for diffuse-rs inference.
//!
//! OpenAI-compatible `/v1/completions` (text or token IDs) and
//! `/v1/chat/completions` (messages + model chat template, requires
//! --tokenizer).

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
// parking_lot's Mutex does not poison on panic. A request that panics while
// holding the lock cannot brick every later request. std::sync::Mutex would
// poison the lock instead.
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};

use crate::gguf_tokenizer::GgufTokenizer;
use crate::model::{self, GenerateOutput, Model, Remasking, SamplerParams, Schedule};

/// Tokenizer backend: an HF tokenizer.json file, or the vocab embedded in
/// the GGUF itself (llama.cpp-converted models).
pub enum AnyTokenizer {
    File(Box<Tokenizer>),
    Gguf(Box<GgufTokenizer>),
}

impl AnyTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i32>, String> {
        match self {
            Self::File(t) => t
                .encode(text, false)
                .map(|e| e.get_ids().iter().map(|&i| i as i32).collect())
                .map_err(|e| e.to_string()),
            Self::Gguf(t) => t
                .encode(text)
                .map(|v| v.into_iter().map(|i| i as i32).collect())
                .map_err(|e| e.to_string()),
        }
    }

    fn decode(&self, ids: &[i32], skip_special: bool) -> Option<String> {
        let u: Vec<u32> = ids.iter().map(|&t| t.max(0) as u32).collect();
        match self {
            Self::File(t) => t.decode(&u, skip_special).ok(),
            Self::Gguf(t) => Some(t.decode(&u, skip_special)),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::File(t) => t.token_to_id(token),
            Self::Gguf(t) => t.token_to_id(token),
        }
    }
}

// =============================================================================
// API types
// =============================================================================

/// Sampling fields shared by the completion and chat endpoints.
#[derive(Deserialize)]
pub struct SamplingOptions {
    #[serde(default = "default_n")]
    pub max_tokens: usize,
    /// Sampling temperature (0 = argmax)
    #[serde(default)]
    pub temperature: f32,
    /// RNG seed for temperature/random sampling
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Diffusion steps
    #[serde(default = "default_steps")]
    pub diffuse_steps: usize,
    /// Entropy threshold for entropy_exit
    #[serde(default = "default_entropy")]
    pub entropy_threshold: f32,
    /// Entropy below which a position commits (remasking="entropy_bound")
    #[serde(default = "default_eb_bound")]
    pub eb_entropy_bound: f32,
    /// Top-token probability needed to unmask (remasking="confidence")
    #[serde(default = "default_confidence")]
    pub confidence_threshold: f32,
    /// Sample only among the k most likely tokens (0 = off)
    #[serde(default)]
    pub top_k: usize,
    /// Nucleus sampling threshold (1.0 = off)
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Classifier-free guidance scale (0 = off, doubles compute)
    #[serde(default)]
    pub cfg_scale: f32,
    /// Remasking strategy: "entropy_exit", "low_confidence", "margin",
    /// "confidence", "entropy_bound", "random"
    #[serde(default = "default_remasking")]
    pub remasking: String,
    /// Unmasking schedule: "cosine" or "linear"
    #[serde(default = "default_schedule")]
    pub schedule: String,
    /// Enable inter-step KV cache
    #[serde(default = "default_true")]
    pub use_cache: bool,
    /// Semi-autoregressive block decoding: block size in tokens
    #[serde(default)]
    pub block_length: Option<usize>,
    /// Truncate output at this token ID (default: model metadata, if any)
    #[serde(default)]
    pub eos_token_id: Option<i32>,
    /// Stream per-step denoising snapshots over SSE
    #[serde(default)]
    pub stream: bool,
    /// Commit positions stable for N consecutive steps (dInfer credit, 0 = off)
    #[serde(default)]
    pub credit_steps: usize,
    /// IterSmooth max alpha (non-Gemma soft-embedding carry-forward, 0 = off)
    #[serde(default)]
    pub iter_smooth: f32,
    /// Vicinity refresh: recompute committed positions within N of a mask (0 = off)
    #[serde(default)]
    pub vicinity: usize,
    /// Token ids forbidden from being committed (e.g. turn markers)
    #[serde(default)]
    pub suppress_ids: Vec<i32>,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    /// Prompt as text string (requires --tokenizer)
    pub prompt: Option<String>,
    /// Prompt as token IDs (alternative to text)
    pub token_ids: Option<Vec<i32>>,
    #[serde(flatten)]
    pub opts: SamplingOptions,
}

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(flatten)]
    pub opts: SamplingOptions,
}

fn default_n() -> usize { 128 }
fn default_seed() -> u64 { 42 }
fn default_steps() -> usize { 16 }
fn default_entropy() -> f32 { 1.5 }
fn default_eb_bound() -> f32 { 0.2 }
fn default_confidence() -> f32 { 0.9 }
fn default_top_p() -> f32 { 1.0 }
fn default_remasking() -> String { "entropy_exit".to_string() }
fn default_schedule() -> String { "cosine".to_string() }
fn default_true() -> bool { true }

/// OpenAI-compatible completions response
#[derive(Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    /// Diffusion-specific: performance stats
    pub elapsed_ms: f64,
    pub tok_per_sec: f64,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    /// Decoded text when a tokenizer is loaded; comma-separated IDs otherwise
    pub text: String,
    pub token_ids: Vec<i32>,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    pub elapsed_ms: f64,
    pub tok_per_sec: f64,
}

#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub model_type: String,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub mask_token_id: u32,
}

/// OpenAI-style error envelope: {"error": {"message", "type"}}
#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

type ApiError = (StatusCode, Json<ErrorResponse>);

fn bad_request(msg: impl Into<String>) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: ErrorBody { message: msg.into(), kind: "invalid_request_error" },
        }),
    )
}

fn internal_error(msg: impl Into<String>) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error: ErrorBody { message: msg.into(), kind: "server_error" } }),
    )
}

// =============================================================================
// Chat templates
// =============================================================================

/// Render messages with the model's chat template and name its turn-end
/// token. LLaDA uses the Llama-3 header format; Dream uses ChatML;
/// LLaDA-MoE uses the Ling `<role>` format.
fn apply_chat_template(model_type: &str, messages: &[ChatMessage]) -> (String, &'static str) {
    match model_type {
        // Ling <role> format (LLaDA-MoE and the Ling-2.0-based LLaDA2 family)
        "llada-moe" | "llada2" => {
            let mut text = String::from("<role>SYSTEM</role>");
            let mut rest = messages;
            if let Some(first) = messages.first() {
                if first.role == "system" {
                    text.push_str(&first.content);
                    text.push('\n');
                    rest = &messages[1..];
                }
            }
            text.push_str("detailed thinking off<|role_end|>");
            for m in rest {
                let tag = match m.role.as_str() {
                    "assistant" => "ASSISTANT",
                    "system" => "SYSTEM",
                    _ => "HUMAN",
                };
                text.push_str(&format!("<role>{tag}</role>{}<|role_end|>", m.content));
            }
            text.push_str("<role>ASSISTANT</role>");
            (text, "<|role_end|>")
        }
        // ChatML: Dream (Qwen2.5) and RND1 (Qwen3) both come from Qwen lineages.
        "dream" | "rnd1" => {
            let mut text = String::new();
            for m in messages {
                text.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
            }
            text.push_str("<|im_start|>assistant\n");
            (text, "<|im_end|>")
        }
        // gemma4 turn format; the assistant role is named "model".
        "diffusion-gemma" => {
            let mut text = String::from("<bos>");
            for m in messages {
                let role = if m.role == "assistant" { "model" } else { m.role.as_str() };
                text.push_str(&format!("<|turn>{role}\n{}<turn|>\n", m.content));
            }
            text.push_str("<|turn>model\n");
            (text, "<turn|>")
        }
        _ => {
            let mut text = String::from("<|startoftext|>");
            for m in messages {
                text.push_str(&format!(
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    m.role, m.content
                ));
            }
            text.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            (text, "<|eot_id|>")
        }
    }
}

// =============================================================================
// Server state
// =============================================================================

/// Model facts cached at startup so handlers never lock the model for
/// metadata. Handlers stay responsive while inference holds the lock.
struct ModelMeta {
    model_type: String,
    n_vocab: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    mask_token_id: u32,
    context_length: Option<usize>,
    max_positions: usize,
    eos_token_id: Option<i32>,
}

/// Inference is serialized on the model lock; this bounds how many requests
/// may wait for it before new ones get 503 (llama.cpp-style slot limit).
const MAX_QUEUED_REQUESTS: usize = 8;

pub struct ServerState {
    model: Mutex<Model>,
    tokenizer: Option<AnyTokenizer>,
    meta: ModelMeta,
    queue: Arc<tokio::sync::Semaphore>,
}

fn acquire_slot(state: &ServerState) -> Result<tokio::sync::OwnedSemaphorePermit, ApiError> {
    state.queue.clone().try_acquire_owned().map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: ErrorBody {
                    message: format!("server busy: {MAX_QUEUED_REQUESTS} requests already queued"),
                    kind: "overloaded_error",
                },
            }),
        )
    })
}

// =============================================================================
// Handlers
// =============================================================================

async fn health() -> &'static str {
    "ok"
}

async fn info(State(state): State<Arc<ServerState>>) -> Json<ModelInfo> {
    let m = &state.meta;
    Json(ModelInfo {
        model_type: m.model_type.clone(),
        n_vocab: m.n_vocab,
        n_embd: m.n_embd,
        n_head: m.n_head,
        n_layer: m.n_layer,
        mask_token_id: m.mask_token_id,
    })
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

async fn models(State(state): State<Arc<ServerState>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelEntry {
            id: state.meta.model_type.clone(),
            object: "model",
            owned_by: "diffuse-rs",
        }],
    })
}

fn build_params(state: &ServerState, opts: &SamplingOptions) -> Result<SamplerParams, ApiError> {
    let remasking: Remasking =
        opts.remasking.parse().map_err(|e| bad_request(format!("{e}")))?;
    let schedule: Schedule = opts.schedule.parse().map_err(|e| bad_request(format!("{e}")))?;
    Ok(SamplerParams {
        n_steps: opts.diffuse_steps,
        temperature: opts.temperature,
        schedule,
        remasking,
        seed: opts.seed,
        entropy_threshold: opts.entropy_threshold,
        eb_entropy_bound: opts.eb_entropy_bound,
        confidence_threshold: opts.confidence_threshold,
        top_k: opts.top_k,
        top_p: opts.top_p,
        cfg_scale: opts.cfg_scale,
        use_cache: opts.use_cache,
        block_length: opts.block_length,
        eos_token_id: opts.eos_token_id.or(state.meta.eos_token_id),
        progress: false,
        credit_steps: opts.credit_steps,
        iter_smooth: opts.iter_smooth,
        vicinity: opts.vicinity,
        suppress_ids: opts.suppress_ids.clone(),
    })
}

/// Reject requests the model cannot serve before any compute happens.
fn validate_request(
    meta: &ModelMeta,
    opts: &SamplingOptions,
    prompt_ids: &[i32],
) -> Result<(), ApiError> {
    if opts.max_tokens == 0 {
        return Err(bad_request("max_tokens must be at least 1"));
    }
    if !(1..=4096).contains(&opts.diffuse_steps) {
        return Err(bad_request("diffuse_steps must be in 1..=4096"));
    }
    if !(0.0..=1.0).contains(&opts.top_p) || opts.top_p == 0.0 {
        return Err(bad_request("top_p must be in (0, 1]"));
    }
    if opts.temperature < 0.0 {
        return Err(bad_request("temperature must be >= 0"));
    }
    if opts.block_length == Some(0) {
        return Err(bad_request("block_length must be at least 1"));
    }
    if let Some(&bad) = prompt_ids.iter().find(|&&t| t < 0 || t as usize >= meta.n_vocab) {
        return Err(bad_request(format!("token id {bad} outside vocab (0..{})", meta.n_vocab)));
    }
    // max_positions bounds per-request allocation independent of a declared
    // context_length, so a huge max_tokens cannot force an OOM allocation.
    let limit = meta.context_length.map_or(meta.max_positions, |ctx| ctx.min(meta.max_positions));
    let total = prompt_ids.len().checked_add(opts.max_tokens);
    if total.is_none_or(|t| t > limit) {
        return Err(bad_request(format!(
            "prompt ({}) + max_tokens ({}) exceeds the {limit} position limit",
            prompt_ids.len(),
            opts.max_tokens
        )));
    }
    Ok(())
}

/// Run inference on a blocking thread (it takes seconds to minutes).
async fn run_generation(
    state: Arc<ServerState>,
    prompt_ids: Vec<i32>,
    max_tokens: usize,
    params: SamplerParams,
) -> Result<GenerateOutput, ApiError> {
    tokio::task::spawn_blocking(move || {
        let mut m = state.model.lock();
        model::generate(&mut m, &prompt_ids, max_tokens, &params)
    })
    .await
    .map_err(|e| internal_error(format!("inference task panicked: {e}")))?
    .map_err(|e| internal_error(format!("inference failed: {e}")))
}

fn encode_text(tok: &AnyTokenizer, text: &str) -> Result<Vec<i32>, ApiError> {
    tok.encode(text).map_err(|e| internal_error(format!("tokenization failed: {e}")))
}

fn decode_text(tokenizer: Option<&AnyTokenizer>, token_ids: &[i32]) -> String {
    let fallback = || token_ids.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");
    tokenizer
        .and_then(|tok| tok.decode(token_ids, true))
        .unwrap_or_else(fallback)
}

/// Stream the denoising process as SSE. Per-step chunks carry a full-text
/// `diffusion` snapshot. Denoising is not left-to-right, so OpenAI token
/// deltas do not apply. The final chunk is OpenAI-shaped with finish_reason
/// and usage. A [DONE] line terminates the stream.
fn sse_generation(
    state: Arc<ServerState>,
    prompt_ids: Vec<i32>,
    max_tokens: usize,
    params: SamplerParams,
    chat: bool,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = response_id();
    let object = if chat { "chat.completion.chunk" } else { "text_completion.chunk" };
    let n_prompt = prompt_ids.len();

    tokio::task::spawn_blocking(move || {
        let _permit = permit; // queue slot held until generation finishes
        let model_name = state.meta.model_type.clone();
        let result = {
            let mut m = state.model.lock();
            model::generate_observed(&mut m, &prompt_ids, max_tokens, &params, &mut |snap| {
                let payload = json!({
                    "id": id, "object": object, "model": model_name,
                    "choices": [{"index": 0, "delta": {}}],
                    "diffusion": {
                        "step": snap.step + 1,
                        "planned_steps": snap.planned_steps,
                        "n_masked": snap.n_masked,
                        "snapshot": decode_text(state.tokenizer.as_ref(), snap.tokens),
                    },
                });
                let _ = tx.send(Event::default().data(payload.to_string()));
            })
        };
        let payload = match result {
            Ok(out) => {
                let text = decode_text(state.tokenizer.as_ref(), &out.tokens);
                let choice = if chat {
                    json!({"index": 0, "delta": {"role": "assistant", "content": text},
                           "finish_reason": out.finish_reason.as_str()})
                } else {
                    json!({"index": 0, "text": text, "token_ids": out.tokens,
                           "finish_reason": out.finish_reason.as_str()})
                };
                json!({"id": id, "object": object, "model": model_name, "choices": [choice],
                       "usage": {"prompt_tokens": n_prompt,
                                  "completion_tokens": out.tokens.len(),
                                  "total_tokens": n_prompt + out.tokens.len()}})
            }
            Err(e) => json!({"error": {"message": e.to_string(), "type": "server_error"}}),
        };
        let _ = tx.send(Event::default().data(payload.to_string()));
        let _ = tx.send(Event::default().data("[DONE]"));
    });

    Sse::new(UnboundedReceiverStream::new(rx).map(Ok::<_, std::convert::Infallible>))
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn response_id() -> String {
    format!(
        "diffuse-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    )
}

async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(mut req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    let params = build_params(&state, &req.opts)?;
    let prompt_ids: Vec<i32> = if let Some(ids) = req.token_ids.take() {
        ids
    } else if let Some(text) = req.prompt.take() {
        let tok = state.tokenizer.as_ref().ok_or_else(|| {
            bad_request("text prompts require --tokenizer flag. Pass token_ids instead.")
        })?;
        encode_text(tok, &text)?
    } else {
        return Err(bad_request("provide either 'prompt' (text) or 'token_ids' (array of ints)"));
    };
    validate_request(&state.meta, &req.opts, &prompt_ids)?;
    let permit = acquire_slot(&state)?;

    if req.opts.stream {
        return Ok(sse_generation(state, prompt_ids, req.opts.max_tokens, params, false, permit));
    }

    let t0 = std::time::Instant::now();
    let output =
        run_generation(state.clone(), prompt_ids.clone(), req.opts.max_tokens, params).await?;
    let elapsed = t0.elapsed().as_secs_f64();

    let text = decode_text(state.tokenizer.as_ref(), &output.tokens);
    let model_type = state.meta.model_type.clone();
    let n_gen = output.tokens.len();

    Ok(Json(CompletionResponse {
        id: response_id(),
        object: "text_completion".to_string(),
        model: model_type,
        usage: Usage {
            prompt_tokens: prompt_ids.len(),
            completion_tokens: n_gen,
            total_tokens: prompt_ids.len() + n_gen,
        },
        choices: vec![Choice {
            index: 0,
            text,
            token_ids: output.tokens,
            finish_reason: output.finish_reason.as_str().to_string(),
        }],
        elapsed_ms: elapsed * 1000.0,
        tok_per_sec: if elapsed > 0.0 { n_gen as f64 / elapsed } else { 0.0 },
    })
    .into_response())
}

async fn chat_completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let tok = state
        .tokenizer
        .as_ref()
        .ok_or_else(|| bad_request("chat completions require the --tokenizer flag"))?;
    if req.messages.is_empty() {
        return Err(bad_request("'messages' must not be empty"));
    }

    let model_type = state.meta.model_type.clone();
    let (text, turn_end) = apply_chat_template(&model_type, &req.messages);
    let prompt_ids = encode_text(tok, &text)?;
    validate_request(&state.meta, &req.opts, &prompt_ids)?;

    let mut params = build_params(&state, &req.opts)?;
    // Default EOS for chat: the template's turn-end token.
    if params.eos_token_id.is_none() {
        params.eos_token_id = tok.token_to_id(turn_end).map(|id| id as i32);
    }

    let permit = acquire_slot(&state)?;
    if req.opts.stream {
        return Ok(sse_generation(state, prompt_ids, req.opts.max_tokens, params, true, permit));
    }

    let t0 = std::time::Instant::now();
    let output =
        run_generation(state.clone(), prompt_ids.clone(), req.opts.max_tokens, params).await?;
    let elapsed = t0.elapsed().as_secs_f64();

    let content = decode_text(Some(tok), &output.tokens);
    let n_gen = output.tokens.len();

    Ok(Json(ChatResponse {
        id: response_id(),
        object: "chat.completion".to_string(),
        model: model_type,
        usage: Usage {
            prompt_tokens: prompt_ids.len(),
            completion_tokens: n_gen,
            total_tokens: prompt_ids.len() + n_gen,
        },
        choices: vec![ChatChoice {
            index: 0,
            message: AssistantMessage { role: "assistant", content },
            finish_reason: output.finish_reason.as_str().to_string(),
        }],
        elapsed_ms: elapsed * 1000.0,
        tok_per_sec: if elapsed > 0.0 { n_gen as f64 / elapsed } else { 0.0 },
    })
    .into_response())
}

// =============================================================================
// Server startup
// =============================================================================

pub fn build_router(model: Model, tokenizer: Option<AnyTokenizer>) -> Router {
    let meta = ModelMeta {
        model_type: model.model_type.clone(),
        n_vocab: model.n_vocab,
        n_embd: model.n_embd,
        n_head: model.n_head,
        n_layer: model.n_layer,
        mask_token_id: model.mask_token_id,
        context_length: model.context_length,
        max_positions: model.max_positions,
        eos_token_id: model.eos_token_id,
    };
    let state = Arc::new(ServerState {
        model: Mutex::new(model),
        tokenizer,
        meta,
        queue: Arc::new(tokio::sync::Semaphore::new(MAX_QUEUED_REQUESTS)),
    });

    Router::new()
        .route("/health", get(health))
        .route("/v1/model", get(info))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}
