# diffuse-rs

diffuse-rs is a Rust inference engine for diffusion language models. It loads LLaDA, LLaDA2.0, Dream, MDLM, and DiffusionGemma straight from GGUF files.

## Models

| Model | arch | Notes |
|-------|------|-------|
| LLaDA-8B (Base/Instruct) | `llada` | Works out of the box. |
| LLaDA-MoE-7B-A1B | `llada-moe` | 64 experts, top-8, QK norm. ~1B active, about 3x faster than dense LLaDA-8B. Ling `<role>` chat. |
| LLaDA2.0 | `llada2` | Grouped expert MoE: group routing, sigmoid gating, renormalized top-k. Verified on `llada2-mini-q4`. |
| Dream-7B (Instruct) | `dream` | Autoregressive logit shift (token i from row i-1). Needs ChatML via `/v1/chat/completions`. Qwen2.5 tokenizer. |
| Dream-Coder / DiffuCoder-7B | `dream` | Code models on the Dream path. No extra config. |
| MDLM | `mdlm` | Masked discrete diffusion. |
| RND1-Base-30B-A3B | `rnd1` | Qwen3 MoE converted to diffusion (~3B active). Experimental: loads and runs, but output quality needs the timestep schedule from the reference implementation. |
| DiffusionGemma 26B-A4B | `diffusion-gemma` | 128 expert top-8 MoE (~4B active) plus a dense shared expert, self conditioning, sliding and global attention. Use `--remasking entropy_bound`. |

DiffusionGemma 26B-A4B at Q6_K scores 282/300 (94.0%) on GSM8K, zero shot, on CPU.

GGUF compatibility: diffuse-rs reads both our `diffuse.*` metadata and llama.cpp's `{arch}.*` and `tokenizer.ggml.*` keys. llama.cpp converted GGUFs load without reconversion. `diffusion.shift_logits` is honored when present; set `DIFFUSE_SHIFT=1` or `DIFFUSE_SHIFT=0` to override the metadata for mislabeled conversions.

## Performance

Throughput is reported as tokens per second across the full canvas. The numbers below come from an AMD Ryzen AI 9 HX 370 (12 threads, LLaDA-8B Q4_K_M, 64 token canvas, `entropy_exit`, 16 steps).

| Matmul path | Throughput |
|-------------|-----------|
| Native (default) | ~29 tok/s |
| candle QMatMul (`DIFFUSE_MATMUL=candle`) | ~12 tok/s |

```bash
./target/release/diffuse-rs bench --model models/llada-8b-q4km.gguf --n-steps 16 --threads 12
```

The native path quantizes activations to Q8_K and runs rows in parallel with rayon over candle's SIMD dot product. Here it is about 2.5x faster than QMatMul.

SIMD covers x86 (AVX2, FMA) and aarch64 (NEON). Apple Silicon and Grace Blackwell (GB10) run accelerated attention, RMSNorm, and SwiGLU. NEON output matches the scalar reference and the x86 build to within float tolerance. A fixed `--seed` reproduces exactly on the same architecture; float summation order differs between x86 and NEON, so outputs can diverge across architectures.

On MoE models, throughput tracks system RAM bandwidth. Enable your RAM's rated speed profile (XMP or EXPO) in BIOS. Boards often default to a slower JEDEC speed, which can cut MoE throughput by 2 to 3 times.

## Sampling

- `--remasking`: `entropy_exit` (default), `low_confidence`, `margin` (top1 minus top2 gap), `confidence` (Fast-dLLM, `--confidence-threshold`), `entropy_bound` (DiffusionGemma adaptive, `--eb-entropy-bound`), `random`.
- `--schedule`: `cosine` (default) or `linear` unmasking schedule.
- `--block-length`: semi-autoregressive blocks, left to right. Default is the model's `diffusion.canvas_length`.
- `--visual`: redraw the decoded canvas after every denoising step, with `░` marking still-masked positions.
- `--temperature` (Gumbel-max), `--top-k`, `--top-p`, `--seed`.
- `--cfg-scale`: classifier-free guidance. It doubles compute and sharpens logits. Pair it with `low_confidence` or `margin`, not `entropy_exit`.
- `--eos-id`: truncates output, skips remaining blocks, sets `finish_reason` to `stop` or `length`.

## Install

Pick one:

```bash
# 1. Prebuilt binary: download for your platform from the Releases page.
#    https://github.com/agalimova/diffuse-rs/releases

# 2. cargo install (needs Rust; no manual clone)
cargo install --git https://github.com/agalimova/diffuse-rs --features server

# 3. Build from source
git clone https://github.com/agalimova/diffuse-rs && cd diffuse-rs
cargo build --release
```

## Quick Start

```bash
pip install huggingface-hub
huggingface-cli download mradermacher/LLaDA-8B-Instruct-GGUF \
  LLaDA-8B-Instruct.Q4_K_M.gguf --local-dir models/

diffuse-rs generate \
  --model models/LLaDA-8B-Instruct.Q4_K_M.gguf \
  --prompt "What is the capital of France?" \
  -n 32 --n-steps 16 --threads 12
```

`--prompt` uses the GGUF's embedded tokenizer, so text goes in and text comes
out. llama.cpp-converted GGUFs carry a vocab; for a GGUF without one, pass
`--prompt-ids` (comma-separated token IDs). Size `-n` to the answer length,
since a diffusion pass costs `-n` times `--n-steps` regardless of output length.

## HTTP Server

The server needs the `server` feature (`cargo build --release --features server`,
or `cargo install --features server`). It uses the GGUF's embedded tokenizer,
so `--tokenizer` is only needed for GGUFs without a vocab.

The server is unauthenticated and has no rate limiting. It binds `127.0.0.1`
by default. Bind a non-loopback address (`--bind 0.0.0.0:8080`) only behind a
trusted network or a reverse proxy that adds authentication.

```bash
diffuse-rs serve \
  --model models/LLaDA-8B-Instruct.Q4_K_M.gguf --threads 12

# Text prompt
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the capital of France?","max_tokens":128}'

# Token IDs (no tokenizer needed)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"token_ids":[2372,341,268,7706,300,11406,30],"max_tokens":128}'

# Chat (applies the model's template)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":64}'
```

Chat templates are per model. LLaDA uses Llama-3 headers, Dream uses ChatML, LLaDA-MoE uses Ling roles. The template's turn end token becomes the default EOS.

Set `"stream": true` for SSE. Each denoising step emits a chunk with a `diffusion` object: `step`, `n_masked`, and a full text `snapshot`. Denoising is not left to right, so the snapshot is the full text rather than a token delta. The final chunk is OpenAI shaped. Standard OpenAI clients work unmodified.

llama.cpp converted GGUFs carry their vocab. diffuse-rs loads it automatically, using byte-level BPE or SentencePiece unigram depending on the model. `--tokenizer` is only needed for GGUFs without a vocab. Run `diffuse-rs tokenize --model m.gguf --text "..."` to check parity.

## Architecture

```
src/
├── model.rs            # shared types and module wiring
├── model/
│   ├── loader.rs       # GGUF parsing, weight loading, arch detection
│   ├── forward.rs      # forward pass, embeddings, FFN/MoE, logits
│   ├── attention.rs    # block-diagonal attention (scalar, AVX2, NEON)
│   ├── sampler.rs      # remasking, denoising loop, batched generation
│   ├── cache.rs        # inter-step KV cache
│   ├── ops.rs          # RMSNorm, SwiGLU, RoPE, softmax (scalar, AVX2, NEON)
│   └── matmul.rs       # native quantized matmul
├── kernels.rs          # Q4_K/Q6_K/Q8_0 block types, dequant, dot products
├── gguf_tokenizer.rs   # embedded tokenizer
├── server.rs           # HTTP API (axum, optional)
└── main.rs             # CLI: bench, generate, serve, profile
```

Weights stay in the memory mapped GGUF and page in on demand. The full f32 tables never materialize. Quantized matmuls use the native path (Q4_K, Q6_K, Q4_0, Q5_0, Q8_0) or candle's QMatMul. Embedding rows dequantize on lookup. Rayon drives dispatch.

## How It Works

1. Initialize the output positions as `[MASK]`.
2. Run a full forward pass through the bidirectional transformer. There is no causal mask.
3. Score each masked position by entropy, margin, or confidence.
4. Commit the most confident positions. Leave the rest masked.
5. Reuse cached K/V for unchanged positions on later steps.
6. Repeat until every position is filled or the step budget runs out.

## Acknowledgements

- [diffuse-cpp](https://github.com/iafiscal1212/diffuse-cpp): reference C++ implementation
- [LLaDA](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct): masked diffusion language model
- [candle](https://github.com/huggingface/candle) (MIT or Apache-2.0): GGUF parsing and quantized matmul

## License

AGPL-3.0. See [LICENSE](LICENSE). For commercial licensing, contact aygul.galimova@duke.edu.
