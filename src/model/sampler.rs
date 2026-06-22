use super::*;

// =============================================================================
// Sampler: iterative unmasking diffusion loop
// =============================================================================

pub fn compute_entropy(logits: &[f32]) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_sum = sum_exp.ln();
    logits.iter().fold(0.0f32, |acc, &l| {
        let lp = (l - max) - log_sum;
        let p = lp.exp();
        if p > 1e-10 {
            acc - p * lp
        } else {
            acc
        }
    })
}

pub(crate) fn log_sum_exp(row: &[f32]) -> f32 {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    max + row.iter().map(|&l| (l - max).exp()).sum::<f32>().ln()
}

/// Indices and logits of the k largest entries, sorted descending, written
/// into `out` (reused: the full-vocab staging is ~1 MB per call otherwise).
pub(crate) fn top_k_logits_into(row: &[f32], k: usize, out: &mut Vec<(u32, f32)>) {
    out.clear();
    out.extend(row.iter().enumerate().map(|(i, &l)| (i as u32, l)));
    let k = k.min(out.len());
    if k < out.len() {
        out.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        out.truncate(k);
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
}

/// Indices and logits of the k largest entries, sorted descending.
/// Test convenience over `top_k_logits_into`.
#[cfg(test)]
pub(crate) fn top_k_logits(row: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut entries = Vec::new();
    top_k_logits_into(row, k, &mut entries);
    entries
}

/// Shortest descending-sorted prefix with cumulative probability >= top_p.
/// Probabilities are relative to the full distribution (lse over all logits).
pub(crate) fn nucleus_prefix(sorted: &[(u32, f32)], lse: f32, top_p: f32) -> &[(u32, f32)] {
    let mut cum = 0.0f32;
    for (i, &(_, logit)) in sorted.iter().enumerate() {
        cum += (logit - lse).exp();
        if cum >= top_p {
            return &sorted[..=i];
        }
    }
    sorted
}

/// Two largest values of a row (row must be non-empty).
pub(crate) fn top2_logits(row: &[f32]) -> (f32, f32) {
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    for &l in row {
        if l > top1 {
            top2 = top1;
            top1 = l;
        } else if l > top2 {
            top2 = l;
        }
    }
    (top1, top2)
}

/// Set the mask token's logit to -inf at every position so it is never sampled.
pub(crate) fn suppress_mask_token(logits: &mut [f32], n_vocab: usize, mask_id: i32) {
    let mask = mask_id as usize;
    if mask < n_vocab {
        for row in logits.chunks_mut(n_vocab) {
            row[mask] = f32::NEG_INFINITY;
        }
    }
}

/// Forbid the given token ids from ever being committed (e.g. turn/end markers
/// that, if committed mid-canvas, truncate the model's own reasoning).
pub(crate) fn suppress_tokens(logits: &mut [f32], n_vocab: usize, ids: &[i32]) {
    if ids.is_empty() {
        return;
    }
    // Walk each row once; the ids list is tiny.
    for row in logits.chunks_mut(n_vocab) {
        for &id in ids {
            if id >= 0 && (id as usize) < n_vocab {
                row[id as usize] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Fraction of the remaining masked tokens to commit at `step` of
/// `total_steps`. Callers iterate `step` in `0..block_steps`, so `step` is
/// always `< total_steps`; the guard keeps the divisors finite regardless.
pub(crate) fn tokens_to_unmask(step: usize, total_steps: usize, masked: usize, schedule: Schedule) -> usize {
    let steps_left = total_steps.saturating_sub(step).max(1);
    let t0 = step as f32 / total_steps as f32;
    let t1 = (step + 1) as f32 / total_steps as f32;
    let frac = match schedule {
        Schedule::Cosine => {
            let c0 = (t0 * std::f32::consts::FRAC_PI_2).cos().max(1e-6);
            let c1 = (t1 * std::f32::consts::FRAC_PI_2).cos();
            (c0 - c1) / c0
        }
        Schedule::Linear => 1.0 / steps_left as f32,
    };
    let n = (frac * masked as f32).round() as usize;
    n.max(if masked > 0 { 1 } else { 0 }).min(masked)
}

struct Candidate {
    pos: usize,
    token: i32,
    entropy: f32,
    margin: f32,
    /// Probability of the most likely token (Confidence remasking only).
    prob: f32,
}

/// Reusable per-step logits buffers. Allocating and zero-filling
/// total_len * n_vocab floats every denoising step dominated per-step
/// overhead, and the cleared rows were never read.
#[derive(Default)]
struct LogitsBufs {
    /// Full-canvas logits [total_len * n_vocab] for the conditional stream.
    full: Vec<f32>,
    /// Active-row logits from a partial (cached) forward.
    active: Vec<f32>,
    /// Full-canvas logits for the CFG unconditional stream.
    uncond: Vec<f32>,
}

/// One stream's forward pass into `full`: a full recompute when the cache is
/// cold, otherwise only active positions (via `active`), scattered into
/// `full`. Rows for cached positions keep stale values; they are never
/// inspected (masked rows and their shift neighbors are always active).
fn stream_logits(
    model: &mut Model,
    seq: &[i32],
    is_masked: &[bool],
    cache: &mut Option<StepCache>,
    step: usize,
    active: &mut Vec<f32>,
    full: &mut Vec<f32>,
) -> Result<()> {
    let total_len = seq.len();
    let force_full = cache.as_ref().is_none_or(|c| !c.initialized);
    if force_full {
        model.forward_pass(seq, &ActiveSplit::all(total_len), cache.as_mut(), full)?;
        if let Some(c) = cache {
            c.update_seq(seq);
        }
        return Ok(());
    }

    let c = cache.as_mut().expect("force_full handles None");
    let split = c.split_active(seq, is_masked, step);
    if split.active.len() >= total_len || split.active.is_empty() {
        model.forward_pass(seq, &ActiveSplit::all(total_len), Some(c), full)?;
    } else {
        model.forward_pass(seq, &split, Some(c), active)?;
        let n_vocab = model.n_vocab;
        if full.len() != total_len * n_vocab {
            full.resize(total_len * n_vocab, 0.0);
        }
        for (row, &pos) in split.active.iter().enumerate() {
            full[pos * n_vocab..(pos + 1) * n_vocab]
                .copy_from_slice(&active[row * n_vocab..(row + 1) * n_vocab]);
        }
    }
    c.update_seq(seq);
    Ok(())
}

/// Split `n_steps` across blocks as evenly as possible, at least 1 each.
pub(crate) fn distribute_steps(n_steps: usize, n_blocks: usize) -> Vec<usize> {
    let base = n_steps / n_blocks;
    let rem = n_steps % n_blocks;
    (0..n_blocks)
        .map(|i| (base + usize::from(i < rem)).max(1))
        .collect()
}

/// Truncate at the first EOS (if configured). Returns true if EOS was found.
pub(crate) fn truncate_at_eos(tokens: &mut Vec<i32>, eos: Option<i32>) -> bool {
    match eos.and_then(|e| tokens.iter().position(|&t| t == e)) {
        Some(idx) => {
            tokens.truncate(idx);
            true
        }
        None => false,
    }
}

/// One denoising step's coordinates: where we are globally (for the KV
/// cache), within the current block's schedule, and which positions the
/// block covers.
struct StepCtx {
    global_step: usize,
    local_step: usize,
    block_steps: usize,
    range: std::ops::Range<usize>,
}

/// One denoising step's outcome, surfaced to streaming observers
/// (the server's SSE stream and the CLI's --visual view).
pub struct StepSnapshot<'a> {
    /// 0-based step just completed.
    pub step: usize,
    pub planned_steps: usize,
    /// Current generated region (mask tokens still present).
    pub tokens: &'a [i32],
    pub n_masked: usize,
}

/// Generate `n_generate` tokens after `prompt` via iterative unmasking.
pub fn generate(
    model: &mut Model,
    prompt: &[i32],
    n_generate: usize,
    params: &SamplerParams,
) -> Result<GenerateOutput> {
    generate_observed(model, prompt, n_generate, params, &mut |_| {})
}

/// Like `generate`, invoking `on_step` after every denoising step.
pub fn generate_observed(
    model: &mut Model,
    prompt: &[i32],
    n_generate: usize,
    params: &SamplerParams,
    on_step: &mut dyn FnMut(StepSnapshot),
) -> Result<GenerateOutput> {
    ensure!(params.n_steps >= 1, "n_steps must be at least 1");
    ensure!(n_generate >= 1, "nothing to generate");
    // max_positions bounds the RoPE table and per-request allocation, so it is
    // the hard cap even when the GGUF declares no context_length.
    let total_len = prompt
        .len()
        .checked_add(n_generate)
        .filter(|&t| t <= model.max_positions)
        .with_context(|| {
            format!(
                "prompt ({}) + max_tokens ({n_generate}) exceeds the {} position limit",
                prompt.len(),
                model.max_positions
            )
        })?;
    for &t in prompt {
        ensure!(
            t >= 0 && (t as usize) < model.n_vocab,
            "prompt token {t} outside vocab (0..{})",
            model.n_vocab
        );
    }

    model.canvas_start = prompt.len();
    let mut denoiser = Denoiser::new(model, prompt, n_generate, params);

    // Default the block size to the model's canvas length (Gemma), so long
    // generations are produced in successive canvas-sized blocks.
    let default_block = model.canvas_length.unwrap_or(n_generate);
    let block_len = params.block_length.unwrap_or(default_block).max(1);
    let block_starts: Vec<usize> = (prompt.len()..total_len).step_by(block_len).collect();
    let steps = distribute_steps(params.n_steps, block_starts.len().max(1));

    let planned_steps: usize = steps.iter().sum();
    let mut global_step = 0;
    let mut lb = LogitsBufs::default();
    for (&start, &block_steps) in block_starts.iter().zip(&steps) {
        let range = start..(start + block_len).min(total_len);
        for local_step in 0..block_steps {
            if denoiser.masked_in(&range) == 0 {
                break;
            }
            denoiser.step(
                model,
                &StepCtx { global_step, local_step, block_steps, range: range.clone() },
                &mut lb,
            )?;
            on_step(StepSnapshot {
                step: global_step,
                planned_steps,
                tokens: &denoiser.seq[prompt.len()..],
                n_masked: denoiser.n_masked,
            });
            global_step += 1;
        }
        // EOS already unmasked: later blocks would be truncated anyway.
        if params.eos_token_id.is_some_and(|eos| denoiser.contains_unmasked(eos)) {
            break;
        }
    }
    Ok(denoiser.finish())
}

/// Generate for a batch of prompts. The batch shares one forward pass per
/// denoising step, so the 13 GB of weights are read once for the whole batch
/// instead of once per sequence. The path uses no inter-step cache and
/// recomputes fully with block-diagonal attention. CFG is not supported. Each
/// prompt gets one output, in input order.
pub fn generate_batch(
    model: &mut Model,
    prompts: &[Vec<i32>],
    n_generate: usize,
    params: &SamplerParams,
) -> Result<Vec<GenerateOutput>> {
    ensure!(!prompts.is_empty(), "empty batch");
    ensure!(n_generate >= 1, "nothing to generate");
    ensure!(params.n_steps >= 1, "n_steps must be at least 1");
    ensure!(params.cfg_scale <= 0.0, "generate_batch does not support CFG");
    for p in prompts {
        let total_len = p.len().checked_add(n_generate).filter(|&t| t <= model.max_positions);
        ensure!(
            total_len.is_some(),
            "prompt ({}) + max_tokens ({n_generate}) exceeds the {} position limit",
            p.len(),
            model.max_positions
        );
        for &t in p {
            ensure!(t >= 0 && (t as usize) < model.n_vocab, "prompt token {t} outside vocab");
        }
    }

    let e = model.n_embd;
    // No-cache denoisers: the batched forward recomputes every position.
    let bp = SamplerParams { use_cache: false, progress: false, ..params.clone() };
    let mut denoisers: Vec<Denoiser> =
        prompts.iter().map(|p| Denoiser::new(model, p, n_generate, &bp)).collect();

    // Row layout (positions, sequence ids, offsets) is fixed across steps;
    // only the tokens and the self-conditioning signal change. Build the
    // batch once and refill the changing fields per step.
    let mut offset = Vec::with_capacity(denoisers.len());
    let (mut pos, mut seq_ids) = (Vec::new(), Vec::new());
    for (s, d) in denoisers.iter().enumerate() {
        offset.push(pos.len());
        pos.extend(0..d.seq.len());
        seq_ids.extend(std::iter::repeat_n(s, d.seq.len()));
    }
    let mut batch = Batch {
        tokens: Vec::new(),
        pos,
        seq: seq_ids,
        canvas_start: prompts.iter().map(|p| p.len()).collect(),
        n_canvas: n_generate,
        sc_signal: None,
    };
    let mut logits: Vec<f32> = Vec::new();

    for step in 0..params.n_steps {
        if denoisers.iter().all(|d| d.n_masked == 0) {
            break;
        }

        // Per-sequence self-conditioning signal (Gemma), concatenated as
        // [n_seq * n_canvas * n_embd]; None before any sequence has a prior step.
        batch.sc_signal = if model.is_gemma
            && model.self_cond.is_some()
            && denoisers.iter().any(|d| d.prev_canvas_logits.is_some())
        {
            let mut sig = vec![0.0f32; denoisers.len() * n_generate * e];
            for (s, d) in denoisers.iter().enumerate() {
                if let Some(prev) = &d.prev_canvas_logits {
                    let seq_sig = model.self_cond_signal(prev, n_generate)?;
                    sig[s * n_generate * e..(s + 1) * n_generate * e].copy_from_slice(&seq_sig);
                }
            }
            Some(sig)
        } else {
            None
        };

        batch.tokens.clear();
        for d in &denoisers {
            batch.tokens.extend_from_slice(&d.seq);
        }
        model.forward_batch(&batch, &mut logits)?;
        let nv = model.n_vocab;

        for (s, d) in denoisers.iter_mut().enumerate() {
            if d.n_masked == 0 {
                continue;
            }
            let len = d.seq.len();
            let rows = &mut logits[offset[s] * nv..(offset[s] + len) * nv];
            let ctx = StepCtx {
                global_step: step,
                local_step: step,
                block_steps: params.n_steps,
                range: batch.canvas_start[s]..len,
            };
            d.process_logits(model, rows, &ctx);
        }
    }

    Ok(denoisers.into_iter().map(|d| d.finish()).collect())
}

/// State of one denoising run: the partially unmasked sequence plus the
/// inter-step KV cache and RNG.
struct Denoiser<'a> {
    params: &'a SamplerParams,
    prompt_len: usize,
    mask_id: i32,
    seq: Vec<i32>,
    is_masked: Vec<bool>,
    n_masked: usize,
    cache: Option<StepCache>,
    /// Separate cache for the unconditional CFG stream.
    cache_uncond: Option<StepCache>,
    /// Credit decoding: last step's pick and its stability streak, per position.
    prev_pick: Vec<i32>,
    credit: Vec<u32>,
    /// Previous step's canvas logits [n_canvas * n_vocab], for Gemma
    /// self-conditioning (None at step 0).
    prev_canvas_logits: Option<Vec<f32>>,
    /// Reused top-k staging for sampled picks (full-vocab sized).
    topk: Vec<(u32, f32)>,
    rng: StdRng,
}

impl<'a> Denoiser<'a> {
    fn new(model: &Model, prompt: &[i32], n_generate: usize, params: &'a SamplerParams) -> Self {
        let total_len = prompt.len() + n_generate;
        let mut seq = Vec::with_capacity(total_len);
        seq.extend_from_slice(prompt);
        seq.resize(total_len, model.mask_token_id as i32);

        let mut is_masked = vec![false; total_len];
        for m in is_masked[prompt.len()..].iter_mut() {
            *m = true;
        }

        let strides: Vec<usize> = model.attn.iter().map(|a| a.kv_stride()).collect();
        let vicinity = params.vicinity;
        let new_cache = || {
            let mut c = StepCache::new(total_len, strides.clone());
            c.vicinity = vicinity;
            c
        };
        // Attention masks by original sequence position (see attention::visible),
        // so the position-reordering cache is correct for windowed models too.
        let use_cache = params.use_cache;

        Self {
            params,
            prompt_len: prompt.len(),
            mask_id: model.mask_token_id as i32,
            seq,
            is_masked,
            n_masked: n_generate,
            cache: use_cache.then(new_cache),
            cache_uncond: (use_cache && params.cfg_scale > 0.0).then(new_cache),
            prev_pick: vec![-1; total_len],
            prev_canvas_logits: None,
            credit: vec![0; total_len],
            topk: Vec::new(),
            rng: StdRng::seed_from_u64(params.seed),
        }
    }

    fn masked_in(&self, range: &std::ops::Range<usize>) -> usize {
        self.is_masked[range.clone()].iter().filter(|&&m| m).count()
    }

    /// Positions whose logits rows this step must produce: every masked
    /// position, plus its left neighbor for logit-shifted models (Dream).
    fn positions_needing_logits(&self, shift: bool) -> Vec<bool> {
        let mut needs = self.is_masked.clone();
        if shift {
            for i in 1..needs.len() {
                if self.is_masked[i] {
                    needs[i - 1] = true;
                }
            }
        }
        needs
    }

    /// Is `token` present and unmasked in the generated region?
    fn contains_unmasked(&self, token: i32) -> bool {
        self.seq[self.prompt_len..]
            .iter()
            .zip(&self.is_masked[self.prompt_len..])
            .any(|(&t, &masked)| !masked && t == token)
    }

    /// Final output: generated tokens truncated at EOS, plus how we finished.
    fn finish(self) -> GenerateOutput {
        let mut tokens = self.seq[self.prompt_len..].to_vec();
        let found_eos = truncate_at_eos(&mut tokens, self.params.eos_token_id);
        let finish_reason = if found_eos || self.n_masked == 0 {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };
        GenerateOutput { tokens, finish_reason }
    }

    /// Compute the self-conditioning signal from the previous step's canvas
    /// logits and hand it to the model for this step's forward (None at step 0).
    fn set_self_cond(&self, model: &mut Model) -> Result<()> {
        let signal = match &self.prev_canvas_logits {
            Some(prev) if model.self_cond.is_some() => {
                let n_canvas = self.seq.len() - self.prompt_len;
                Some(model.self_cond_signal(prev, n_canvas)?)
            }
            _ => None,
        };
        model.sc_signal = signal;
        Ok(())
    }

    /// Stash this step's canvas logit rows (raw/softcapped, pre mask-suppression)
    /// for the next step's self-conditioning.
    fn cache_canvas_logits(&mut self, model: &Model, logits: &[f32]) {
        let smoothing = self.params.iter_smooth > 0.0 && !model.is_gemma;
        if model.self_cond.is_some() || smoothing {
            let n_vocab = logits.len() / self.seq.len();
            self.prev_canvas_logits = Some(logits[self.prompt_len * n_vocab..].to_vec());
        }
    }

    /// IterSmooth (non-Gemma): set the model's smooth signal from the previous
    /// step's canvas logits, with alpha ramping 0.1 -> params.iter_smooth.
    fn set_iter_smooth(&self, model: &mut Model, step: usize, total: usize) {
        let alpha_max = self.params.iter_smooth;
        let signal = if alpha_max > 0.0 && !model.is_gemma {
            self.prev_canvas_logits.as_ref().map(|prev| {
                let n_canvas = self.seq.len() - self.prompt_len;
                let frac = if total > 1 { step as f32 / (total - 1) as f32 } else { 1.0 };
                let alpha = 0.1 + (alpha_max - 0.1) * frac;
                model.iter_smooth_signal(prev, n_canvas, alpha)
            })
        } else {
            None
        };
        model.smooth_signal = signal;
    }

    fn step(&mut self, model: &mut Model, ctx: &StepCtx, lb: &mut LogitsBufs) -> Result<()> {
        self.set_self_cond(model)?;
        self.set_iter_smooth(model, ctx.global_step, ctx.block_steps);
        self.step_logits(model, ctx.global_step, lb)?;
        let committed = self.process_logits(model, &mut lb.full, ctx);
        if self.params.progress {
            eprintln!(
                "  step {} (block {}..{}, {}/{}): unmasked {} tokens, {} remaining",
                ctx.global_step + 1, ctx.range.start, ctx.range.end,
                ctx.local_step + 1, ctx.block_steps, committed, self.n_masked
            );
        }
        Ok(())
    }

    /// Everything after the forward: stash logits for self-conditioning,
    /// suppress the mask token, rank the masked candidates, and commit. Split
    /// out so a batched driver can feed in externally-computed logits.
    fn process_logits(&mut self, model: &Model, logits: &mut [f32], ctx: &StepCtx) -> usize {
        self.cache_canvas_logits(model, logits);
        let n_vocab = logits.len() / self.seq.len();
        suppress_mask_token(logits, n_vocab, self.mask_id);
        suppress_tokens(logits, n_vocab, &self.params.suppress_ids);
        let shift = usize::from(model.logit_shift);
        let mut cands = self.collect_candidates(logits, &ctx.range, shift);
        self.update_credit(&cands);
        let n_unmask = self.rank_candidates(&mut cands, ctx);
        self.commit(&cands, n_unmask)
    }

    /// Credit decoding: track how many consecutive steps each position's argmax
    /// pick has been stable (reset on change).
    fn update_credit(&mut self, cands: &[Candidate]) {
        if self.params.credit_steps == 0 {
            return;
        }
        for c in cands {
            if c.token == self.prev_pick[c.pos] {
                self.credit[c.pos] += 1;
            } else {
                self.credit[c.pos] = 0;
                self.prev_pick[c.pos] = c.token;
            }
        }
    }

    /// Order `cands` by the remasking strategy (best-to-commit first) and return
    /// how many of the leading candidates to unmask this step.
    fn rank_candidates(&mut self, cands: &mut [Candidate], ctx: &StepCtx) -> usize {
        let schedule_n =
            tokens_to_unmask(ctx.local_step, ctx.block_steps, cands.len(), self.params.schedule);
        let (entropy_threshold, confidence_threshold, eb_bound) = (
            self.params.entropy_threshold,
            self.params.confidence_threshold,
            self.params.eb_entropy_bound,
        );
        let by_entropy = |a: &Candidate, b: &Candidate| a.entropy.total_cmp(&b.entropy);
        match self.params.remasking {
            Remasking::EntropyExit => {
                cands.sort_by(by_entropy);
                let easy = cands.iter().take_while(|c| c.entropy < entropy_threshold).count();
                schedule_n.max(easy).min(cands.len())
            }
            Remasking::LowConfidence => {
                cands.sort_by(by_entropy);
                schedule_n
            }
            Remasking::Margin => {
                cands.sort_by(|a, b| b.margin.total_cmp(&a.margin));
                schedule_n
            }
            Remasking::Confidence => {
                cands.sort_by(|a, b| b.prob.total_cmp(&a.prob));
                let confident = cands.iter().take_while(|c| c.prob >= confidence_threshold).count();
                schedule_n.max(confident).min(cands.len())
            }
            Remasking::EntropyBound => {
                cands.sort_by(by_entropy);
                // Adaptive: commit only the confident positions (>=1 for progress).
                let confident = cands.iter().take_while(|c| c.entropy < eb_bound).count();
                confident.max(1).min(cands.len())
            }
            Remasking::Random => {
                for i in (1..cands.len()).rev() {
                    cands.swap(i, self.rng.gen_range(0..=i));
                }
                schedule_n
            }
        }
    }

    /// Unmask the first `n_unmask` ranked candidates plus any that are
    /// credit-stable; returns the number committed.
    fn commit(&mut self, cands: &[Candidate], n_unmask: usize) -> usize {
        let credit_steps = self.params.credit_steps;
        let mut committed = 0;
        for (rank, c) in cands.iter().enumerate() {
            let credit_ok = credit_steps > 0 && self.credit[c.pos] >= credit_steps as u32;
            if rank < n_unmask || credit_ok {
                self.seq[c.pos] = c.token;
                self.is_masked[c.pos] = false;
                self.n_masked -= 1;
                committed += 1;
            }
        }
        committed
    }

    /// Logits for one diffusion step into `lb.full`, with classifier-free
    /// guidance applied when enabled. Rows served from the cache are stale;
    /// they are never masked and so never inspected.
    fn step_logits(&mut self, model: &mut Model, step: usize, lb: &mut LogitsBufs) -> Result<()> {
        let needs = self.positions_needing_logits(model.logit_shift);
        stream_logits(model, &self.seq, &needs, &mut self.cache, step, &mut lb.active, &mut lb.full)?;
        let scale = self.params.cfg_scale;
        if scale <= 0.0 {
            return Ok(());
        }

        // Unconditional stream: prompt replaced by masks (LLaDA CFG).
        let mut seq_un = self.seq.clone();
        let mut masked_un = needs;
        for i in 0..self.prompt_len {
            seq_un[i] = self.mask_id;
            masked_un[i] = true;
        }
        stream_logits(
            model,
            &seq_un,
            &masked_un,
            &mut self.cache_uncond,
            step,
            &mut lb.active,
            &mut lb.uncond,
        )?;

        // Blend the rows that collect_candidates actually consumes. For
        // logit_shift models a masked position at `pos` reads its source row
        // `pos - shift`, so blending row `pos` would leave the consumed row
        // unguided whenever the left neighbor is already committed.
        let shift = usize::from(model.logit_shift);
        let n_vocab = lb.full.len() / self.seq.len();
        for pos in 0..self.seq.len() {
            if !self.is_masked[pos] {
                continue;
            }
            let src = pos.saturating_sub(shift);
            let row = src * n_vocab..(src + 1) * n_vocab;
            for (c, &u) in lb.full[row.clone()].iter_mut().zip(&lb.uncond[row]) {
                *c = u + (scale + 1.0) * (*c - u);
            }
        }
        Ok(())
    }

    fn collect_candidates(
        &mut self,
        logits: &[f32],
        range: &std::ops::Range<usize>,
        shift: usize,
    ) -> Vec<Candidate> {
        let n_vocab = logits.len() / self.seq.len();
        let needs_entropy = matches!(
            self.params.remasking,
            Remasking::EntropyExit | Remasking::LowConfidence | Remasking::EntropyBound
        );
        let needs_prob = self.params.remasking == Remasking::Confidence;
        let mut cands = Vec::new();
        for pos in range.clone() {
            if !self.is_masked[pos] {
                continue;
            }
            let src = pos.saturating_sub(shift);
            let row = &logits[src * n_vocab..(src + 1) * n_vocab];
            let (top1, top2) = top2_logits(row);
            cands.push(Candidate {
                pos,
                token: self.pick_token(row),
                entropy: if needs_entropy { compute_entropy(row) } else { 0.0 },
                margin: top1 - top2,
                prob: if needs_prob { (top1 - log_sum_exp(row)).exp() } else { 0.0 },
            });
        }
        cands
    }

    /// Argmax at temperature 0; otherwise Gumbel-max sampling (equivalent to
    /// sampling from softmax(logits / T)), optionally restricted to the
    /// top-k / nucleus token set.
    fn pick_token(&mut self, row: &[f32]) -> i32 {
        let p = self.params;
        if p.temperature <= 0.0 {
            let mut best = (0usize, f32::NEG_INFINITY);
            for (j, &logit) in row.iter().enumerate() {
                if logit > best.1 {
                    best = (j, logit);
                }
            }
            return best.0 as i32;
        }

        if p.top_k == 0 && p.top_p >= 1.0 {
            let mut best = (0u32, f32::NEG_INFINITY);
            for (j, &logit) in row.iter().enumerate() {
                let key = logit / p.temperature + self.gumbel();
                if key > best.1 {
                    best = (j as u32, key);
                }
            }
            return best.0 as i32;
        }

        // Nucleus mass beyond the top NUCLEUS_CAP logits is negligible.
        const NUCLEUS_CAP: usize = 1024;
        let k = if p.top_k > 0 { p.top_k } else { NUCLEUS_CAP };
        // Take the scratch out so the gumbel calls below can borrow self.
        let mut sorted = std::mem::take(&mut self.topk);
        top_k_logits_into(row, k, &mut sorted);
        let kept = if p.top_p < 1.0 {
            nucleus_prefix(&sorted, log_sum_exp(row), p.top_p)
        } else {
            &sorted[..]
        };
        let mut best = (kept[0].0, f32::NEG_INFINITY);
        for &(idx, logit) in kept {
            let key = logit / p.temperature + self.gumbel();
            if key > best.1 {
                best = (idx, key);
            }
        }
        self.topk = sorted;
        best.0 as i32
    }

    fn gumbel(&mut self) -> f32 {
        let u: f32 = self.rng.gen_range(1e-12..1.0);
        -(-u.ln()).ln()
    }
}
