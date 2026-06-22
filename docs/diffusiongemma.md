# DiffusionGemma 26B-A4B — implementation blueprint

Source: GGUF header (unsloth/DevQuasar Q4_K_M, byte-identical conversions)
+ llama.cpp PR #24423 (`src/models/diffusion-gemma.cpp`, `gemma4-common.h`),
still an open draft as of 2026-06-13 (not merged).

## Config (from GGUF metadata, arch `diffusion-gemma`)
- 30 layers, n_embd 2816, n_head 16, vocab 262144, ctx 262144 (cap ours at 8192)
- head_count_kv: per-layer array [8×5, 2, ...] (sliding=8, global=2)
- sliding_window_pattern: [T,T,T,T,T,F,...] — true = sliding/local layer
- global layers (pattern false): idx 5,11,17,23,29 — head_dim 512 (key_length),
  rope_dim 512, theta 1e6, full attention, 2 KV heads
- sliding layers: head_dim 256 (key_length_swa), rope_dim 256, theta 1e4 (freq_base_swa),
  window 1024, 8 KV heads
- expert_count 128, expert_used 8, expert_ff 704 (fused gate_up = 1408)
- final_logit_softcapping 30, rms_eps 1e-6, mask_token_id 4, eos 1, bos 2
- tokenizer.ggml.model = gemma4 (byte-BPE; pre may be "gemma"/unknown → llama-bpe ok-ish)

## DONE (committed)
- Per-layer heterogeneous attention (LayerAttn): geometry, per-layer RoPE,
  sliding-window band mask, per-layer KV-cache strides. Regression-clean.
- Gemma attn-array parsing (head_count_kv / sliding_window_pattern / key_length(_swa)
  / rope dims+thetas / window) gated on arch == diffusion-gemma.

## TODO — Gemma forward path (the remaining unit)

### Per-layer tensors → roles
- `attn_norm` input RMSNorm (before attention)
- `attn_q/k/v`, `attn_q_norm`/`attn_k_norm` (per-head RMSNorm on Q,K before RoPE),
  `attn_output`
- `post_attention_norm` = post-attn norm BEFORE residual add
- `ffn_norm` = dense (shared expert) pre-norm
- `ffn_gate/up/down` = dense shared expert (GeGLU, gelu-tanh)
- `post_ffw_norm_1` = dense path post-norm
- `pre_ffw_norm_2` = MoE path pre-norm
- `ffn_gate_inp.weight` router [n_embd,128] + `ffn_gate_inp.scale` [n_embd]
- `ffn_gate_up_exps.weight` [n_embd, 1408, 128] fused (gate=first 704, up=last 704)
- `ffn_down_exps.weight` [704, n_embd, 128] + `ffn_down_exps.scale` [128] per-expert
- `post_ffw_norm_2` = MoE path post-norm
- `post_ffw_norm` = outer post-norm (after dense+moe sum, before residual)
- `layer_output_scale` [1] (canvas), `enc_layer_output_scale` [1] (prompt) scalar mul

### Forward (per layer), residual structure
```
cur      = rms_norm(inpL, attn_norm)
cur      = attention(cur)                       # scale = 1.0 (NOT 1/sqrt(hd))
cur      = rms_norm(cur, post_attention_norm)   # post-norm inside residual
attn_out = cur + inpL

# dense + MoE run in PARALLEL on the SAME attn_out, then sum:
dense = rms_norm(attn_out, ffn_norm)
dense = geglu_ffn(dense, ffn_gate, ffn_up, ffn_down)   # gelu_tanh(gate)*up -> down
dense = rms_norm(dense, post_ffw_norm_1)

# MoE router uses raw attn_out, NOT pre_ffw_norm_2:
r     = rms_norm_noscale(attn_out) * (1/sqrt(n_embd)) * ffn_gate_inp.scale
logits= ffn_gate_inp · r ; softmax; top-8; renormalize weights to sum 1
moe   = rms_norm(attn_out, pre_ffw_norm_2)
moe   = moe_ffn(moe, fused gate_up split, down, geglu, down_scale per expert)
moe   = rms_norm(moe, post_ffw_norm_2)

cur   = dense + moe
cur   = rms_norm(cur, post_ffw_norm)
cur   = cur + attn_out                          # FFN residual
cur  *= layer_output_scale                      # (enc_layer_output_scale for prompt rows)
inpL  = cur
```
Note: `rms_norm_noscale` = RMSNorm with no learned weight (just normalize).

### Embedding
`inpL = embed * sqrt(n_embd)`. Canvas (generated) positions then get an extra
`rms_norm_noscale` (after self-cond added). Prompt positions: sqrt scale only.

### Attention specifics
- scale = 1.0 (no query_pre_attn_scalar, QK-norm handles it)
- QK-norm: per-head RMSNorm on Q and K (after reshape, before RoPE); V no norm
- rope_freqs.weight [256] used as freq_factors on GLOBAL layers only (nullptr on sliding)
- prompt = causal + KV-cached (encoder); canvas = bidirectional (decoder).
  First cut: treat whole sequence bidirectional (our engine default) and validate.

### Final logits
`x = output_norm(x); logits = output · x` (output tied to token_embd);
`logits = 30 * tanh(logits / 30)` softcap before sampling.

### Self-conditioning (model-level: self_cond_pre_norm/gate/up/down)
Input = previous step's canvas logits [n_vocab, C]:
```
probs = softmax(prev_logits * sc_temp_inv)
soft  = (probs · token_embd) * sqrt(n_embd)        # soft embedding
n     = rms_norm(soft, self_cond_pre_norm)
sig   = self_cond_down( gelu_tanh(self_cond_gate·n) * (self_cond_up·n) )
sig  *= sc_use                                     # 0 on step 0
canvas = canvas + sig ; canvas = rms_norm_noscale(canvas)   # before layer 0
```
Step 0: sc_use=0 → reduces to rms_norm_noscale(embed*sqrt(n_embd)).

### Uncertain / verify against live model
- gate/up fused split order (assume gate=first half) — confirm from output
- exact per-expert down_scale application point
- whether the prompt/canvas dual-mask is needed for correct output or whether
  uniform bidirectional suffices for short prompts
- rope_freqs freq_factors semantics
