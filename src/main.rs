//! diffuse-rs: Pure Rust CPU inference for diffusion language models.

mod gguf_tokenizer;
mod kernels;
mod model;
#[cfg(feature = "server")]
mod server;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

use model::{Model, SamplerParams};

#[derive(Parser)]
#[command(name = "diffuse-rs", version, about = "CPU inference engine for diffusion LLMs")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Benchmark inference throughput (matches diffuse-cpp methodology)
    Bench(BenchArgs),
    /// Generate tokens from a prompt
    Generate(GenerateArgs),
    /// Generate for several prompts at once (one batched forward per step)
    GenBatch(GenBatchArgs),
    /// Profile per-op breakdown of a forward pass
    Profile(ProfileArgs),
    /// Encode text with the GGUF-embedded tokenizer (llama.cpp-style)
    Tokenize(TokenizeArgs),
    /// Start HTTP API server
    #[cfg(feature = "server")]
    Serve(ServeArgs),
}

#[derive(Args)]
struct TokenizeArgs {
    #[arg(long)]
    model: String,
    #[arg(long)]
    text: String,
    /// Compare against a tokenizer.json file (requires server feature)
    #[arg(long)]
    tokenizer: Option<String>,
}

#[derive(Args)]
struct BenchArgs {
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "64")]
    batch_size: usize,
    #[arg(long, default_value = "32")]
    prompt_len: usize,
    /// Comma-separated token IDs (overrides --prompt-len with real tokens)
    #[arg(long)]
    tokens: Option<String>,
    #[command(flatten)]
    sampler: SamplerArgs,
    #[arg(long, default_value = "12")]
    threads: usize,
    #[arg(long, default_value = "3")]
    reps: usize,
    #[arg(long, default_value = "1")]
    warmup: usize,
}

#[derive(Args)]
struct GenerateArgs {
    #[arg(long)]
    model: String,
    /// Prompt text, tokenized with the GGUF's embedded tokenizer.
    #[arg(long)]
    prompt: Option<String>,
    /// Prompt as comma-separated token IDs (alternative to --prompt).
    #[arg(long)]
    prompt_ids: Option<String>,
    /// Live canvas view: redraw the decoded canvas after every denoising step
    /// (needs an embedded tokenizer).
    #[arg(long)]
    visual: bool,
    #[arg(short = 'n', long, default_value = "128")]
    n_generate: usize,
    #[command(flatten)]
    sampler: SamplerArgs,
    #[arg(long, default_value = "12")]
    threads: usize,
}

#[derive(Args)]
struct GenBatchArgs {
    #[arg(long)]
    model: String,
    /// Prompts as ';'-separated lists of comma-separated token IDs
    #[arg(long)]
    prompts: String,
    #[arg(short = 'n', long, default_value = "128")]
    n_generate: usize,
    #[command(flatten)]
    sampler: SamplerArgs,
    #[arg(long, default_value = "12")]
    threads: usize,
}

#[derive(Args)]
struct ProfileArgs {
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "12")]
    threads: usize,
    #[arg(long, default_value = "40")]
    seq_len: usize,
}

#[cfg(feature = "server")]
#[derive(Args)]
struct ServeArgs {
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "12")]
    threads: usize,
    /// Bind address. Defaults to loopback; the server is unauthenticated, so
    /// bind a non-loopback address only behind a trusted network or proxy.
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    /// Path to tokenizer.json (enables text prompts and decoded output)
    #[arg(long)]
    tokenizer: Option<String>,
}

/// Sampler flags shared by bench and generate.
#[derive(Args)]
struct SamplerArgs {
    #[arg(long = "n-steps", default_value = "16")]
    n_steps: usize,
    #[arg(long, default_value = "entropy_exit")]
    remasking: String,
    /// Unmasking schedule: cosine or linear
    #[arg(long, default_value = "cosine")]
    schedule: String,
    #[arg(long = "entropy-threshold", default_value = "1.5")]
    entropy_threshold: f32,
    /// Entropy below which a position commits (remasking=entropy_bound)
    #[arg(long = "eb-entropy-bound", default_value = "0.2")]
    eb_entropy_bound: f32,
    /// Top-token probability needed to unmask (remasking=confidence)
    #[arg(long = "confidence-threshold", default_value = "0.9")]
    confidence_threshold: f32,
    #[arg(long, default_value = "0.0")]
    temperature: f32,
    /// Sample only among the k most likely tokens (0 = off)
    #[arg(long = "top-k", default_value = "0")]
    top_k: usize,
    /// Nucleus sampling threshold (1.0 = off)
    #[arg(long = "top-p", default_value = "1.0")]
    top_p: f32,
    /// Classifier-free guidance scale (0 = off, doubles compute)
    #[arg(long = "cfg-scale", default_value = "0.0")]
    cfg_scale: f32,
    /// Commit positions stable for N consecutive steps (dInfer credit, 0 = off)
    #[arg(long = "credit-steps", default_value = "0")]
    credit_steps: usize,
    /// IterSmooth max alpha (non-Gemma soft-embedding carry-forward, 0 = off)
    #[arg(long = "iter-smooth", default_value = "0")]
    iter_smooth: f32,
    /// Vicinity refresh: recompute committed positions within N of a mask (0 = off)
    #[arg(long, default_value = "0")]
    vicinity: usize,
    /// Comma-separated token ids to forbid committing. Turn markers, for
    /// example, truncate reasoning mid-canvas.
    #[arg(long = "suppress-ids", value_delimiter = ',')]
    suppress_ids: Vec<i32>,
    #[arg(long, default_value = "42")]
    seed: u64,
    /// Disable the inter-step KV cache
    #[arg(long)]
    no_cache: bool,
    /// Semi-autoregressive block decoding: block size in tokens
    #[arg(long = "block-length")]
    block_length: Option<usize>,
    /// Truncate output at this token ID (e.g. 126348 for LLaDA <|eot_id|>)
    #[arg(long = "eos-id")]
    eos_id: Option<i32>,
}

impl SamplerArgs {
    fn to_params(&self) -> Result<SamplerParams> {
        Ok(SamplerParams {
            n_steps: self.n_steps,
            temperature: self.temperature,
            schedule: self.schedule.parse()?,
            remasking: self.remasking.parse()?,
            seed: self.seed,
            entropy_threshold: self.entropy_threshold,
            eb_entropy_bound: self.eb_entropy_bound,
            confidence_threshold: self.confidence_threshold,
            top_k: self.top_k,
            top_p: self.top_p,
            cfg_scale: self.cfg_scale,
            use_cache: !self.no_cache,
            block_length: self.block_length,
            eos_token_id: self.eos_id,
            progress: true,
            credit_steps: self.credit_steps,
            iter_smooth: self.iter_smooth,
            vicinity: self.vicinity,
            suppress_ids: self.suppress_ids.clone(),
        })
    }
}

fn parse_token_ids(s: &str) -> Result<Vec<i32>> {
    s.split(',')
        .map(|t| {
            t.trim()
                .parse::<i32>()
                .map_err(|e| anyhow::anyhow!("bad token id {t:?}: {e}"))
        })
        .collect()
}

fn init_threads(threads: usize) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Bench(args) => run_bench(&args),
        Command::Generate(args) => run_generate(&args),
        Command::GenBatch(args) => run_gen_batch(&args),
        Command::Profile(args) => run_profile(&args),
        Command::Tokenize(args) => run_tokenize(&args),
        #[cfg(feature = "server")]
        Command::Serve(args) => run_server(&args),
    }
}

// =============================================================================
// Tokenize
// =============================================================================

fn run_tokenize(args: &TokenizeArgs) -> Result<()> {
    let tok = gguf_tokenizer::GgufTokenizer::from_gguf(&args.model)?
        .ok_or_else(|| anyhow::anyhow!("this GGUF has no embedded tokenizer"))?;
    let ids = tok.encode(&args.text)?;
    println!("gguf:    {ids:?}");
    println!("decoded: {:?}", tok.decode(&ids, false));

    if let Some(path) = &args.tokenizer {
        #[cfg(feature = "server")]
        {
            let file = tokenizers::Tokenizer::from_file(path)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
            let reference: Vec<u32> = file
                .encode(args.text.as_str(), false)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .get_ids()
                .to_vec();
            println!("file:    {reference:?}");
            println!("match:   {}", if reference == ids { "YES" } else { "NO" });
        }
        #[cfg(not(feature = "server"))]
        anyhow::bail!("--tokenizer comparison requires --features server (path: {path})");
    }
    Ok(())
}

// =============================================================================
// Bench
// =============================================================================

fn run_bench(args: &BenchArgs) -> Result<()> {
    init_threads(args.threads)?;
    let params = args.sampler.to_params()?;
    let mut model = Model::from_gguf(&args.model)?;

    let prompt: Vec<i32> = match &args.tokens {
        Some(tok_str) => parse_token_ids(tok_str)?,
        None => vec![1; args.prompt_len],
    };

    for w in 0..args.warmup {
        eprintln!("  warmup {}/{}", w + 1, args.warmup);
        model::generate(&mut model, &prompt, args.batch_size, &params)?;
    }

    let mut tps_list = Vec::new();
    let mut total_tokens = 0usize;
    let mut total_ms = 0.0f64;

    for rep in 0..args.reps {
        let t0 = std::time::Instant::now();
        let output = model::generate(&mut model, &prompt, args.batch_size, &params)?.tokens;
        let elapsed = t0.elapsed();
        let tps = output.len() as f64 / elapsed.as_secs_f64();
        tps_list.push(tps);
        total_tokens += output.len();
        total_ms += elapsed.as_secs_f64() * 1000.0;
        eprintln!("  rep {}/{}: {:.1} tok/s ({:.2}s)", rep + 1, args.reps, tps, elapsed.as_secs_f64());
    }

    let avg = tps_list.iter().sum::<f64>() / tps_list.len() as f64;
    println!("---");
    println!("tok_per_sec: {avg:.2}");
    println!("total_tokens: {total_tokens}");
    println!("total_time_ms: {total_ms:.0}");
    println!("engine: native");
    println!("threads: {}", args.threads);
    println!("remasking: {}", args.sampler.remasking);
    println!("n_steps: {}", args.sampler.n_steps);
    println!("batch_size: {}", args.batch_size);
    println!("prompt_len: {}", args.prompt_len);
    Ok(())
}

// =============================================================================
// Generate
// =============================================================================

/// Render one denoising snapshot: ░ for masked positions, committed tokens
/// decoded as contiguous runs so UTF-8 sequences spanning tokens survive.
fn render_canvas(tokens: &[i32], mask_id: i32, tok: &gguf_tokenizer::GgufTokenizer) -> String {
    let mut canvas = String::new();
    let mut run: Vec<u32> = Vec::new();
    let mut flush = |canvas: &mut String, run: &mut Vec<u32>| {
        if !run.is_empty() {
            canvas.push_str(&tok.decode(run, true));
            run.clear();
        }
    };
    for &t in tokens {
        if t == mask_id {
            flush(&mut canvas, &mut run);
            canvas.push('\u{2591}');
        } else {
            run.push(t.max(0) as u32);
        }
    }
    flush(&mut canvas, &mut run);
    canvas
}

fn run_generate(args: &GenerateArgs) -> Result<()> {
    init_threads(args.threads)?;
    // Validate sampler flags before the multi-second model load.
    let mut params = args.sampler.to_params()?;
    let mut model = Model::from_gguf(&args.model)?;
    let tokenizer = gguf_tokenizer::GgufTokenizer::from_gguf(&args.model)?;
    // No default EOS truncation here, unlike the server: diffusion models can
    // commit a spurious EOS mid-canvas while the real answer is still forming,
    // and a default truncation would then discard it. The server default is
    // safe because chat sets EOS to the template's turn-end token. Pass
    // --eos-id to opt in.

    let prompt: Vec<i32> = match (&args.prompt_ids, &args.prompt) {
        (Some(ids), _) => parse_token_ids(ids)?,
        (None, Some(text)) => {
            let tok = tokenizer
                .as_ref()
                .context("this GGUF has no embedded tokenizer; pass --prompt-ids instead")?;
            tok.encode(text)?.into_iter().map(|i| i as i32).collect()
        }
        (None, None) => bail!("provide --prompt <text> or --prompt-ids <ids>"),
    };

    let t0 = std::time::Instant::now();
    let output = if args.visual {
        let tok = tokenizer
            .as_ref()
            .context("--visual needs an embedded tokenizer; run without it or use a llama.cpp GGUF")?;
        params.progress = false; // the step lines would interleave with the canvas
        let mask_id = model.mask_token_id as i32;
        model::generate_observed(&mut model, &prompt, args.n_generate, &params, &mut |snap| {
            let canvas = render_canvas(snap.tokens, mask_id, tok);
            print!(
                "\x1b[2J\x1b[H-- step {}/{} | {} masked --\n{canvas}\n",
                snap.step + 1,
                snap.planned_steps,
                snap.n_masked
            );
            use std::io::Write;
            let _ = std::io::stdout().flush();
        })?
    } else {
        model::generate(&mut model, &prompt, args.n_generate, &params)?
    };
    let elapsed = t0.elapsed();

    let tps = output.tokens.len() as f64 / elapsed.as_secs_f64();
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s, finish: {})",
        output.tokens.len(),
        elapsed.as_secs_f64(),
        tps,
        output.finish_reason.as_str()
    );
    // Always print the machine-readable token line (harnesses parse it), then
    // the decoded text when the model carries a vocab.
    println!("Tokens: {:?}", output.tokens);
    if let Some(tok) = &tokenizer {
        let ids: Vec<u32> = output.tokens.iter().map(|&t| t.max(0) as u32).collect();
        println!("{}", tok.decode(&ids, true));
    }
    Ok(())
}

fn run_gen_batch(args: &GenBatchArgs) -> Result<()> {
    init_threads(args.threads)?;
    let params = args.sampler.to_params()?;
    // Fail on unsupported flags before the multi-second model load.
    if params.cfg_scale > 0.0 {
        bail!("gen-batch does not support --cfg-scale");
    }
    if params.block_length.is_some() {
        bail!("gen-batch decodes the whole canvas; --block-length is not supported");
    }
    if params.vicinity > 0 {
        eprintln!("[diffuse-rs] note: gen-batch always recomputes fully; --vicinity has no effect");
    }
    let mut model = Model::from_gguf(&args.model)?;
    let prompts: Vec<Vec<i32>> =
        args.prompts.split(';').map(parse_token_ids).collect::<Result<_>>()?;

    let t0 = std::time::Instant::now();
    let outputs = model::generate_batch(&mut model, &prompts, args.n_generate, &params)?;
    let elapsed = t0.elapsed();

    let total: usize = outputs.iter().map(|o| o.tokens.len()).sum();
    println!(
        "Batched {} prompts, {total} tokens in {:.2}s ({:.1} tok/s aggregate)",
        outputs.len(),
        elapsed.as_secs_f64(),
        total as f64 / elapsed.as_secs_f64()
    );
    for (i, o) in outputs.iter().enumerate() {
        println!("[{i}] ({}) Tokens: {:?}", o.finish_reason.as_str(), o.tokens);
    }
    Ok(())
}

// =============================================================================
// Profile
// =============================================================================

fn run_profile(args: &ProfileArgs) -> Result<()> {
    init_threads(args.threads)?;
    let mut model = Model::from_gguf(&args.model)?;

    let seq_len = args.seq_len;
    let tokens: Vec<i32> = vec![1; seq_len];
    let e = model.n_embd;
    // MoE GGUFs may omit feed_forward_length; fall back to the widest
    // intermediate so the dense-equivalent estimate stays finite.
    let ff = if model.n_ff > 0 { model.n_ff } else { model.max_ff };

    // Warmup
    let _ = model.forward(&tokens)?;

    // Full forward
    let t0 = std::time::Instant::now();
    let _ = model.forward(&tokens)?;
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Total forward: {total_ms:.1}ms for {seq_len} tokens\n");

    // Q8K quantization cost. Reuse one buffer so the loop times the kernel,
    // not the allocator.
    let dummy = vec![0.1f32; e];
    let mut q8 = Vec::new();
    kernels::quantize_row_q8_k_into(&dummy, &mut q8);
    let t0 = std::time::Instant::now();
    for _ in 0..1000 {
        kernels::quantize_row_q8_k_into(&dummy, &mut q8);
    }
    let q8k_us = t0.elapsed().as_secs_f64() * 1000.0;
    println!("quantize_q8k({e}): {:.3}ms/call", q8k_us / 1000.0);

    // Single dot product through the production path: candle's SIMD vec_dot,
    // the same kernel native_matmul dispatches per weight row.
    let nb = e / 256;
    let zero_block = kernels::BlockQ4K { d: 0, dmin: 0, scales: [0; 12], qs: [0; 128] };
    let q4_rows = vec![zero_block; nb];
    // SAFETY: our BlockQ4K is a field-for-field layout match of candle's
    // (asserted by test_q8k_layout_matches_candle and the block-size tests).
    let blocks = unsafe {
        std::slice::from_raw_parts(q4_rows.as_ptr() as *const model::k_quants::BlockQ4K, nb)
    };
    let ys = model::cast_q8k(&q8);
    let t0 = std::time::Instant::now();
    for _ in 0..10000 {
        let _ = <model::k_quants::BlockQ4K as model::GgmlType>::vec_dot(e, blocks, ys)?;
    }
    let dot_us = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("vec_dot_q4k({e}, SIMD): {dot_us:.3}us/call");

    // Dense-equivalent matmul estimate from the measured per-row dot cost.
    let dots_per_layer = seq_len * (5 * e + 2 * ff);
    let matmul_total =
        dots_per_layer as f64 * dot_us / 1000.0 * model.n_layer as f64 / args.threads as f64;
    println!("\nMatmul total ({}t): {matmul_total:.1}ms ({:.0}% of total)", args.threads, matmul_total / total_ms * 100.0);
    println!("Other: {:.1}ms ({:.0}%)", total_ms - matmul_total, (total_ms - matmul_total) / total_ms * 100.0);

    Ok(())
}

// =============================================================================
// Server
// =============================================================================

#[cfg(feature = "server")]
fn run_server(args: &ServeArgs) -> Result<()> {
    init_threads(args.threads)?;
    let model = Model::from_gguf(&args.model)?;

    let tokenizer = match &args.tokenizer {
        Some(path) => {
            eprintln!("[diffuse-rs] loading tokenizer from {path}");
            Some(server::AnyTokenizer::File(Box::new(
                tokenizers::Tokenizer::from_file(path)
                    .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?,
            )))
        }
        // Fall back to the vocab embedded in the GGUF (llama.cpp models).
        None => gguf_tokenizer::GgufTokenizer::from_gguf(&args.model)?
            .map(|t| server::AnyTokenizer::Gguf(Box::new(t))),
    };

    let bind = args.bind.clone();
    eprintln!("[diffuse-rs] starting server on {bind} ({} threads)", args.threads);

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let app = server::build_router(model, tokenizer);
        let listener = tokio::net::TcpListener::bind(&bind)
            .await
            .with_context(|| format!("failed to bind {bind}"))?;
        eprintln!("[diffuse-rs] listening on {bind}");
        eprintln!("[diffuse-rs] POST /v1/completions       — generate tokens");
        eprintln!("[diffuse-rs] POST /v1/chat/completions  — chat");
        eprintln!("[diffuse-rs] GET  /v1/model             — model info");
        eprintln!("[diffuse-rs] GET  /v1/models            — model list (OpenAI)");
        eprintln!("[diffuse-rs] GET  /health               — health check");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
                eprintln!("[diffuse-rs] shutting down");
            })
            .await
            .context("server error")
    })?;

    Ok(())
}
