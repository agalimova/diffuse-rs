use super::*;

// =============================================================================
// Forward pass
// =============================================================================

impl Model {

    fn ensure_bufs(&mut self, n: usize) {
        let need_new = match &self.bufs {
            None => true,
            Some(b) => b.cur.len() < n * self.n_embd,
        };
        if need_new {
            let bufs = ComputeBuffers::new(self, n);
            self.bufs = Some(bufs);
        }
    }

    /// Full forward pass over all positions (no cache).
    pub fn forward(&mut self, tokens: &[i32]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        self.forward_pass(tokens, &ActiveSplit::all(tokens.len()), None, &mut out)?;
        Ok(out)
    }

    /// Unified forward pass. Computes embeddings/QKV/FFN only for active
    /// positions; attention runs over cached + active K/V. Writes logits for
    /// active positions only (active.len() * n_vocab, in active order) into
    /// `out`, reusing its allocation.
    pub fn forward_pass(
        &mut self,
        seq: &[i32],
        split: &ActiveSplit,
        mut cache: Option<&mut StepCache>,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let (na, nc) = (split.active.len(), split.cached.len());
        ensure!(na + nc == seq.len(), "active+cached must cover the sequence");
        ensure!(nc == 0 || cache.is_some(), "cached positions require a cache");
        let max_pos = split.active.iter().chain(&split.cached).max().copied().unwrap_or(0);
        ensure!(max_pos < seq.len(), "position out of range");

        self.ensure_bufs(seq.len());
        let mut b = self.bufs.take().unwrap();
        b.scratch.invalidate();

        let active_tokens: Vec<i32> = split.active.iter().map(|&p| seq[p]).collect();
        self.embed_tokens(&active_tokens, &mut b.cur);

        // Gemma: scale embeddings by sqrt(n_embd). Canvas (generated) positions
        // additionally get the self-conditioning signal added (when set) then a
        // noscale-RMSNorm; prompt positions get only the scaling.
        if self.is_gemma {
            let e = self.n_embd;
            for v in b.cur[..na * e].iter_mut() {
                *v *= self.embed_scale;
            }
            for (t, &pos) in split.active.iter().enumerate() {
                if pos < self.canvas_start {
                    continue;
                }
                let row = &mut b.cur[t * e..(t + 1) * e];
                if let Some(sig) = &self.sc_signal {
                    let c = pos - self.canvas_start;
                    for (x, &s) in row.iter_mut().zip(&sig[c * e..(c + 1) * e]) {
                        *x += s;
                    }
                }
                rms_norm_noscale(row, self.rms_norm_eps);
            }
        } else if let Some(sig) = &self.smooth_signal {
            // IterSmooth (non-Gemma): add the previous step's expected embedding
            // to still-masked canvas positions (e_mask + alpha * E[emb]).
            let e = self.n_embd;
            let mask_id = self.mask_token_id as i32;
            for (t, &pos) in split.active.iter().enumerate() {
                if pos >= self.canvas_start && seq[pos] == mask_id {
                    let c = pos - self.canvas_start;
                    for (x, &s) in b.cur[t * e..(t + 1) * e].iter_mut().zip(&sig[c * e..(c + 1) * e]) {
                        *x += s;
                    }
                }
            }
        }

        // Key rows are [cached positions | active rows]; queries are the
        // active rows. Masking uses these original sequence positions; the
        // cache is single-sequence only, so cached rows carry sequence id 0.
        // The metadata is identical for every layer, so build it once.
        let qr = Rows { pos: &split.active, seq: &[], prefix: &[] };
        let k_pos: Vec<usize> = if nc > 0 {
            split.cached.iter().chain(&split.active).copied().collect()
        } else {
            Vec::new()
        };
        let rows = if nc > 0 {
            PassRows { q: qr, k: Rows { pos: &k_pos, seq: &[], prefix: &[] } }
        } else {
            PassRows { q: qr, k: qr }
        };

        for il in 0..self.n_layer {
            self.attention_block(il, &mut b, split, rows, cache.as_deref_mut())?;
            self.ffn_block(&self.layers[il], &mut b, na)?;
        }
        self.compute_logits(&mut b, na, out)?;

        self.bufs = Some(b);
        Ok(())
    }

    /// One forward pass over a batch of independent sequences (no inter-step
    /// cache): rows are concatenated, attention is block-diagonal (each row
    /// carries its sequence position and id), so the shared weight reads are
    /// amortized across the batch. Writes logits `[total_rows * n_vocab]`
    /// into `out`, reusing its allocation.
    pub fn forward_batch(&mut self, batch: &Batch, out: &mut Vec<f32>) -> Result<()> {
        let total = batch.tokens.len();
        ensure!(batch.pos.len() == total && batch.seq.len() == total, "batch arrays misaligned");
        let e = self.n_embd;

        self.ensure_bufs(total);
        let mut b = self.bufs.take().unwrap();
        b.scratch.invalidate();

        self.embed_tokens(&batch.tokens, &mut b.cur);
        // Gemma: scale embeddings; canvas positions also get the per-sequence
        // self-conditioning signal then a noscale-RMSNorm (see forward_pass).
        if self.is_gemma {
            for v in b.cur[..total * e].iter_mut() {
                *v *= self.embed_scale;
            }
            for t in 0..total {
                let (s, p) = (batch.seq[t], batch.pos[t]);
                if p < batch.canvas_start[s] {
                    continue;
                }
                let row = &mut b.cur[t * e..(t + 1) * e];
                if let Some(sig) = &batch.sc_signal {
                    let c = (s * batch.n_canvas + (p - batch.canvas_start[s])) * e;
                    for (x, &sg) in row.iter_mut().zip(&sig[c..c + e]) {
                        *x += sg;
                    }
                }
                rms_norm_noscale(row, self.rms_norm_eps);
            }
        }

        // Gemma encodes each sequence's prompt causally up to its own
        // canvas_start; supply that per-row so block-diagonal masking is
        // per-sequence. Other models have no causal prefix.
        let prefix: Vec<usize> = if self.is_gemma {
            batch.seq.iter().map(|&s| batch.canvas_start[s]).collect()
        } else {
            Vec::new()
        };
        let split = ActiveSplit { active: (0..total).collect(), cached: Vec::new() };
        let qr = Rows { pos: &batch.pos, seq: &batch.seq, prefix: &prefix };
        let rows = PassRows { q: qr, k: qr };
        for il in 0..self.n_layer {
            self.attention_block(il, &mut b, &split, rows, None)?;
            self.ffn_block(&self.layers[il], &mut b, total)?;
        }
        self.compute_logits(&mut b, total, out)?;

        self.bufs = Some(b);
        Ok(())
    }

    /// Look up `tokens` in the embedding table, writing one n_embd row each
    /// into `cur`. Token ids must be valid (0..n_vocab). generate() and the
    /// server validate inputs, and soft_embed only passes in-vocab ids.
    fn embed_tokens(&self, tokens: &[i32], cur: &mut [f32]) {
        debug_assert!(
            tokens.iter().all(|&t| t >= 0 && (t as usize) < self.n_vocab),
            "embed_tokens called with out-of-vocab id"
        );
        let e = self.n_embd;
        let bytes = self.tok_embd.bytes.as_slice();
        match self.tok_embd.wtype {
            EmbdType::F16 => {
                for (t, &tid) in tokens.iter().enumerate() {
                    let row = &bytes[tid as usize * e * 2..];
                    for j in 0..e {
                        let h = u16::from_le_bytes([row[j * 2], row[j * 2 + 1]]);
                        cur[t * e + j] = kernels::f16_to_f32(h);
                    }
                }
            }
            EmbdType::F32 => {
                for (t, &tid) in tokens.iter().enumerate() {
                    let row = &bytes[tid as usize * e * 4..];
                    for j in 0..e {
                        let raw = [row[j * 4], row[j * 4 + 1], row[j * 4 + 2], row[j * 4 + 3]];
                        cur[t * e + j] = f32::from_le_bytes(raw);
                    }
                }
            }
            EmbdType::Q4K => {
                let nb = e / QK_K;
                let rb = nb * Q4K_BLOCK_SIZE;
                for (t, &tid) in tokens.iter().enumerate() {
                    // SAFETY: bytes come from a heap allocation (>= align 8);
                    // BlockQ4K is align 2 and the row offset is a multiple of 144.
                    let blocks = unsafe {
                        let ptr = bytes.as_ptr().add(tid as usize * rb);
                        std::slice::from_raw_parts(ptr as *const BlockQ4K, nb)
                    };
                    kernels::dequantize_q4k_row(blocks, &mut cur[t * e..(t + 1) * e]);
                }
            }
            EmbdType::Q6K => {
                let nb = e / QK_K;
                let rb = nb * Q6K_BLOCK_SIZE;
                for (t, &tid) in tokens.iter().enumerate() {
                    // SAFETY: as above; BlockQ6K is align 2, row stride 210 * nb.
                    let blocks = unsafe {
                        let ptr = bytes.as_ptr().add(tid as usize * rb);
                        std::slice::from_raw_parts(ptr as *const BlockQ6K, nb)
                    };
                    kernels::dequantize_q6k_row(blocks, &mut cur[t * e..(t + 1) * e]);
                }
            }
            EmbdType::Q40 => {
                let rb = e / 32 * 18; // bytes per row
                for (t, &tid) in tokens.iter().enumerate() {
                    let row = &bytes[tid as usize * rb..(tid as usize + 1) * rb];
                    kernels::dequantize_q4_0_row(row, &mut cur[t * e..(t + 1) * e]);
                }
            }
            EmbdType::Q80 => {
                let rb = e / 32 * 34; // bytes per row
                for (t, &tid) in tokens.iter().enumerate() {
                    let row = &bytes[tid as usize * rb..(tid as usize + 1) * rb];
                    kernels::dequantize_q8_0_row(row, &mut cur[t * e..(t + 1) * e]);
                }
            }
        }
    }

    /// RMSNorm each of the first n rows of src into out.
    fn norm_rows(&self, out: &mut [f32], src: &[f32], w: &[f32], n: usize) {
        let e = self.n_embd;
        for t in 0..n {
            rms_norm(&mut out[t * e..(t + 1) * e], &src[t * e..(t + 1) * e], w, self.rms_norm_eps);
        }
    }

    /// `pos`/`seq` give each active row's sequence position and batch-sequence
    /// id (empty seq = single sequence). The single-forward path passes
    /// `split.active` + `&[]`; batched generation passes per-sequence positions
    /// and ids so several sequences share the pass via block-diagonal attention.
    fn attention_block(
        &self,
        il: usize,
        b: &mut ComputeBuffers,
        split: &ActiveSplit,
        rows: PassRows,
        cache: Option<&mut StepCache>,
    ) -> Result<()> {
        let layer = &self.layers[il];
        let la = &self.attn[il];
        let e = self.n_embd;
        let (hkv, hd) = (la.n_head_kv, la.head_dim);
        let ks = la.q_stride(); // Q row stride = n_head * head_dim
        let kvs = la.kv_stride(); // K/V row stride (un-repeated heads)
        let (na, nc) = (split.active.len(), split.cached.len());
        let nt = na + nc;

        b.residual[..na * e].copy_from_slice(&b.cur[..na * e]);
        self.norm_rows(&mut b.normed, &b.residual, &layer.attn_norm, na);
        b.scratch.touch();

        match &layer.qkv {
            Qkv::Split { wq, wk, wv } => {
                wq.forward(&mut b.q[..na * ks], &b.normed[..na * e], &mut b.scratch, na)?;
                wk.forward(&mut b.k[..na * kvs], &b.normed[..na * e], &mut b.scratch, na)?;
                wv.forward(&mut b.v[..na * kvs], &b.normed[..na * e], &mut b.scratch, na)?;
            }
            Qkv::Fused(w) => {
                let fused = la.fused_stride();
                w.forward(&mut b.qkv[..na * fused], &b.normed[..na * e], &mut b.scratch, na)?;
                for t in 0..na {
                    let row = &b.qkv[t * fused..(t + 1) * fused];
                    b.q[t * ks..(t + 1) * ks].copy_from_slice(&row[..ks]);
                    b.k[t * kvs..(t + 1) * kvs].copy_from_slice(&row[ks..ks + kvs]);
                    b.v[t * kvs..(t + 1) * kvs].copy_from_slice(&row[ks + kvs..]);
                }
            }
        }
        add_bias(&mut b.q, layer.bq.as_deref(), na);
        add_bias(&mut b.k, layer.bk.as_deref(), na);
        add_bias(&mut b.v, layer.bv.as_deref(), na);

        // Per-head QK-norm (LLaDA-MoE/Qwen3, Gemma), before RoPE.
        if let Some(w) = &layer.q_norm {
            norm_heads(&mut b.q[..na * ks], w, self.rms_norm_eps);
        }
        if let Some(w) = &layer.k_norm {
            norm_heads(&mut b.k[..na * kvs], w, self.rms_norm_eps);
        }
        // Gemma applies a plain (unweighted) RMSNorm to each V head; V is not RoPE'd.
        if self.is_gemma {
            for head in b.v[..na * kvs].chunks_mut(hd) {
                rms_norm_noscale(head, self.rms_norm_eps);
            }
        }

        apply_rope(&mut b.q, &mut b.k, &la.rope, rows.q.pos, la.shape(), la.rope_dim);

        if nc > 0 {
            let cache = cache.as_ref().expect("checked in forward_pass");
            cache.gather(il, &split.cached, &mut b.k_full[..nc * kvs], &mut b.v_full[..nc * kvs]);
            b.k_full[nc * kvs..nt * kvs].copy_from_slice(&b.k[..na * kvs]);
            b.v_full[nc * kvs..nt * kvs].copy_from_slice(&b.v[..na * kvs]);
        }
        if let Some(cache) = cache {
            cache.store(il, &split.active, &b.k[..na * kvs], &b.v[..na * kvs]);
        }

        let (k, v) = if nc > 0 {
            (&b.k_full[..nt * kvs], &b.v_full[..nt * kvs])
        } else {
            (&b.k[..na * kvs], &b.v[..na * kvs])
        };
        let shape = AttnShape {
            nq: na,
            nk: nt,
            n_head: la.n_head,
            n_head_kv: hkv,
            head_dim: hd,
            sliding_window: la.sliding_window,
            // Gemma encodes the prompt causally (encoder) and the canvas
            // bidirectionally. Position-aware masking makes this valid even
            // when the cache concatenates reordered cached + active rows.
            causal_prefix: self.is_gemma.then_some(self.canvas_start),
            // Gemma uses no softmax scaling (kq_scale = 1.0); QK-norm with
            // learned weights controls Q/K magnitude. Verified empirically:
            // 1/sqrt(hd) collapses the output, 1.0 does not.
            scale: if self.is_gemma { 1.0 } else { 1.0 / (hd as f32).sqrt() },
        };
        attention(&mut b.attn_out[..na * ks], &b.q[..na * ks], k, v, shape, rows.q, rows.k);

        layer.wo.forward(&mut b.cur[..na * e], &b.attn_out[..na * ks], &mut b.scratch, na)?;
        // Gemma: post-attention RMSNorm before the residual add.
        if let Some(pn) = &layer.post_attn_norm {
            norm_rows_inplace(&mut b.cur, pn, na, e, self.rms_norm_eps);
        }
        residual_add(&mut b.cur[..na * e], &b.residual[..na * e]);
        Ok(())
    }

    fn ffn_block(&self, layer: &Layer, b: &mut ComputeBuffers, na: usize) -> Result<()> {
        if let Ffn::Gemma(gf) = &layer.ffn {
            return self.gemma_ffn_block(layer, gf, b, na);
        }
        let e = self.n_embd;
        b.residual[..na * e].copy_from_slice(&b.cur[..na * e]);
        self.norm_rows(&mut b.normed, &b.residual, &layer.ffn_norm, na);
        b.scratch.touch();

        match &layer.ffn {
            Ffn::Dense(w) => self.dense_ffn(w, b, na)?,
            Ffn::Moe(m) => self.moe_ffn(m, b, na)?,
            Ffn::Gemma(_) => unreachable!("handled above"),
        }
        residual_add(&mut b.cur[..na * e], &b.residual[..na * e]);
        Ok(())
    }

    /// SwiGLU FFN over the first `na` rows of b.normed, written to b.cur.
    fn dense_ffn(&self, w: &FfnWeights, b: &mut ComputeBuffers, na: usize) -> Result<()> {
        let (e, ff) = (self.n_embd, w.ff);
        w.gate.forward(&mut b.gate[..na * ff], &b.normed[..na * e], &mut b.scratch, na)?;
        w.up.forward(&mut b.up[..na * ff], &b.normed[..na * e], &mut b.scratch, na)?;
        silu_mul(&mut b.gate[..na * ff], &b.up[..na * ff]);
        w.down.forward(&mut b.cur[..na * e], &b.gate[..na * ff], &mut b.scratch, na)?;
        Ok(())
    }

    /// MoE FFN over the first `na` rows of b.normed, written to b.cur. Tokens
    /// are batched per expert so each active expert runs one matmul; an optional
    /// shared expert runs on every token and is summed in.
    fn moe_ffn(&self, m: &MoeWeights, b: &mut ComputeBuffers, na: usize) -> Result<()> {
        let e = self.n_embd;
        // Take the scratch out of `b` so the expert loop can borrow both.
        let mut rs = std::mem::take(&mut b.route);
        rs.begin(m.cfg.n_expert);
        for t in 0..na {
            route_token_moe(m, &b.normed[t * e..(t + 1) * e], &mut rs);
            for &(expert, weight) in &rs.picks {
                rs.routes[expert].push((t, weight));
            }
        }

        b.moe_out[..na * e].fill(0.0);
        for (expert, tokens) in rs.routes.iter().enumerate() {
            if !tokens.is_empty() {
                self.run_moe_expert(m, b, expert, tokens)?;
            }
        }
        b.route = rs;

        match &m.shared {
            Some(shared) => {
                self.dense_ffn(shared, b, na)?; // -> b.cur
                residual_add(&mut b.cur[..na * e], &b.moe_out[..na * e]);
            }
            None => b.cur[..na * e].copy_from_slice(&b.moe_out[..na * e]),
        }
        Ok(())
    }

    /// One general-MoE expert (SwiGLU) over its routed tokens, accumulating the
    /// routing-weighted outputs into `b.moe_out`. Experts consume `b.normed`.
    fn run_moe_expert(&self, m: &MoeWeights, b: &mut ComputeBuffers, expert: usize, tokens: &[(usize, f32)]) -> Result<()> {
        let (e, ff) = (self.n_embd, m.gate.rows());
        let rows = tokens.len();
        for (row, &(token, _)) in tokens.iter().enumerate() {
            b.moe_act[row * e..(row + 1) * e].copy_from_slice(&b.normed[token * e..(token + 1) * e]);
        }
        m.gate.forward_expert(expert, &mut b.gate[..rows * ff], &b.moe_act[..rows * e], &mut b.scratch, rows)?;
        m.up.forward_expert(expert, &mut b.up[..rows * ff], &b.moe_act[..rows * e], &mut b.scratch, rows)?;
        silu_mul(&mut b.gate[..rows * ff], &b.up[..rows * ff]);
        m.down.forward_expert(expert, &mut b.moe_act[..rows * e], &b.gate[..rows * ff], &mut b.scratch, rows)?;
        for (row, &(token, weight)) in tokens.iter().enumerate() {
            for d in 0..e {
                b.moe_out[token * e + d] += weight * b.moe_act[row * e + d];
            }
        }
        Ok(())
    }

    /// DiffusionGemma FFN over the first `na` rows. The post-attention residual
    /// arrives in `b.cur`; the dense shared expert and the fused-expert MoE both
    /// run on it (each with its own norms), are summed, outer-normed, residual-
    /// added, and scaled back into `b.cur`. See docs/diffusiongemma.md.
    fn gemma_ffn_block(&self, layer: &Layer, gf: &GemmaFfn, b: &mut ComputeBuffers, na: usize) -> Result<()> {
        let (e, eps) = (self.n_embd, self.rms_norm_eps);
        b.residual[..na * e].copy_from_slice(&b.cur[..na * e]); // attn_out

        self.gemma_dense_expert(layer, gf, b, na)?; // -> b.attn_out
        let mut rs = std::mem::take(&mut b.route);
        self.gemma_route(gf, b, na, &mut rs);
        self.gemma_run_experts(gf, b, na, &rs.routes)?; // -> b.moe_out
        b.route = rs;

        // Combine the two paths, outer-norm, add the residual, scale.
        for i in 0..na * e {
            b.cur[i] = b.attn_out[i] + b.moe_out[i];
        }
        norm_rows_inplace(&mut b.cur, &gf.post_norm, na, e, eps);
        residual_add(&mut b.cur[..na * e], &b.residual[..na * e]);
        if gf.out_scale != 1.0 {
            for v in b.cur[..na * e].iter_mut() {
                *v *= gf.out_scale;
            }
        }
        Ok(())
    }

    /// Dense shared expert: GeGLU FFN on rms_norm(attn_out, ffn_norm), then a
    /// post-norm, written into `b.attn_out`. Reads the residual from `b.residual`.
    fn gemma_dense_expert(&self, layer: &Layer, gf: &GemmaFfn, b: &mut ComputeBuffers, na: usize) -> Result<()> {
        let (e, eps, ff) = (self.n_embd, self.rms_norm_eps, gf.dense.ff);
        self.norm_rows(&mut b.normed, &b.residual, &layer.ffn_norm, na);
        b.scratch.touch();
        gf.dense.gate.forward(&mut b.gate[..na * ff], &b.normed[..na * e], &mut b.scratch, na)?;
        gf.dense.up.forward(&mut b.up[..na * ff], &b.normed[..na * e], &mut b.scratch, na)?;
        geglu_mul(&mut b.gate[..na * ff], &b.up[..na * ff]);
        gf.dense.down.forward(&mut b.attn_out[..na * e], &b.gate[..na * ff], &mut b.scratch, na)?;
        norm_rows_inplace(&mut b.attn_out, &gf.dense_post_norm, na, e, eps);
        Ok(())
    }

    /// Assign each token to its top-k experts, filling `rs.routes` with the
    /// (token, weight) pairs routed to each expert. The router input is
    /// noscale-norm(attn_out) * (1/sqrt(e)) * router_scale. Reads the residual
    /// from `b.residual`.
    fn gemma_route(&self, gf: &GemmaFfn, b: &ComputeBuffers, na: usize, rs: &mut RouteScratch) {
        let (e, eps, cfg) = (self.n_embd, self.rms_norm_eps, gf.cfg);
        let inv = 1.0 / (e as f32).sqrt();
        rs.begin(cfg.n_expert);
        let mut input = vec![0.0f32; e];
        for token in 0..na {
            input.copy_from_slice(&b.residual[token * e..(token + 1) * e]);
            rms_norm_noscale(&mut input, eps);
            for (x, &s) in input.iter_mut().zip(&gf.router_scale) {
                *x *= inv * s;
            }
            select_top_experts(&gf.router, &input, cfg, rs);
            for &(expert, weight) in &rs.picks {
                rs.routes[expert].push((token, weight));
            }
        }
    }

    /// Run each routed expert over its tokens and accumulate the weighted,
    /// per-expert-scaled results into `b.moe_out`. Experts consume
    /// rms_norm(attn_out, moe_pre_norm); the result is post-normed.
    fn gemma_run_experts(&self, gf: &GemmaFfn, b: &mut ComputeBuffers, na: usize, routes: &[Vec<(usize, f32)>]) -> Result<()> {
        let (e, eps) = (self.n_embd, self.rms_norm_eps);
        self.norm_rows(&mut b.normed, &b.residual, &gf.moe_pre_norm, na);
        b.scratch.touch();
        b.moe_out[..na * e].fill(0.0);
        for (expert, tokens) in routes.iter().enumerate() {
            if !tokens.is_empty() {
                self.gemma_one_expert(gf, b, expert, tokens)?;
            }
        }
        norm_rows_inplace(&mut b.moe_out, &gf.moe_post_norm, na, e, eps);
        Ok(())
    }

    /// One expert over its routed tokens: gather inputs, fused gate_up -> GeGLU
    /// -> down, then scatter weight*down_scale*output into `b.moe_out`.
    fn gemma_one_expert(&self, gf: &GemmaFfn, b: &mut ComputeBuffers, expert: usize, tokens: &[(usize, f32)]) -> Result<()> {
        let e = self.n_embd;
        let (ff_exp, fused) = (gf.ff_exp, gf.ff_exp * 2);
        let rows = tokens.len();
        for (row, &(token, _)) in tokens.iter().enumerate() {
            b.moe_act[row * e..(row + 1) * e].copy_from_slice(&b.normed[token * e..(token + 1) * e]);
        }
        gf.gate_up.forward_expert(expert, &mut b.gate[..rows * fused], &b.moe_act[..rows * e], &mut b.scratch, rows)?;
        for row in 0..rows {
            for k in 0..ff_exp {
                let gate = b.gate[row * fused + k];
                let up = b.gate[row * fused + ff_exp + k];
                b.up[row * ff_exp + k] = gelu_tanh(gate) * up;
            }
        }
        gf.down.forward_expert(expert, &mut b.moe_act[..rows * e], &b.up[..rows * ff_exp], &mut b.scratch, rows)?;
        let down_scale = gf.down_scale.get(expert).copied().unwrap_or(1.0);
        for (row, &(token, weight)) in tokens.iter().enumerate() {
            let scaled = weight * down_scale;
            for d in 0..e {
                b.moe_out[token * e + d] += scaled * b.moe_act[row * e + d];
            }
        }
        Ok(())
    }

    /// Project the normed hidden states into `out` [na * n_vocab]. The matmul
    /// overwrites every element, so a reused buffer needs no clearing; resize
    /// zero-fills only a newly grown region.
    fn compute_logits(&self, b: &mut ComputeBuffers, na: usize, out: &mut Vec<f32>) -> Result<()> {
        let e = self.n_embd;
        self.norm_rows(&mut b.normed, &b.cur, &self.output_norm, na);
        b.scratch.touch();
        let need = na * self.n_vocab;
        if out.len() != need {
            out.resize(need, 0.0);
        }
        self.output.forward(out, &b.normed[..na * e], &mut b.scratch, na)?;
        // Gemma final logit softcap: c * tanh(x / c).
        if self.logit_softcap > 0.0 {
            let c = self.logit_softcap;
            for v in out.iter_mut() {
                *v = c * (*v / c).tanh();
            }
        }
        Ok(())
    }

    /// Self-conditioning signal [n_canvas * n_embd] from the previous step's
    /// canvas logits [n_canvas * n_vocab]: soft-embed each distribution, then
    /// pass through the gated MLP (added to the canvas embeddings next forward).
    pub(crate) fn self_cond_signal(&self, prev_logits: &[f32], n_canvas: usize) -> Result<Vec<f32>> {
        let sc = self.self_cond.as_ref().expect("self_cond present");
        let (e, ff, eps) = (self.n_embd, sc.mlp.ff, self.rms_norm_eps);

        // Soft embedding -> pre-norm, row by row into `normed`.
        let mut normed = vec![0.0f32; n_canvas * e];
        let mut emb = vec![0.0f32; e];
        let mut topk = Vec::with_capacity(SC_TOPK);
        for c in 0..n_canvas {
            let logits = &prev_logits[c * self.n_vocab..(c + 1) * self.n_vocab];
            let soft = &mut normed[c * e..(c + 1) * e];
            self.soft_embed(logits, &mut emb, soft, &mut topk);
            rms_norm_inplace(soft, &sc.pre_norm, eps);
        }

        // Gated MLP: GeGLU(gate, up) -> down.
        let mut scratch = ActScratch::default();
        let mut gate = vec![0.0f32; n_canvas * ff];
        let mut up = vec![0.0f32; n_canvas * ff];
        let mut out = vec![0.0f32; n_canvas * e];
        sc.mlp.gate.forward(&mut gate, &normed, &mut scratch, n_canvas)?;
        sc.mlp.up.forward(&mut up, &normed, &mut scratch, n_canvas)?;
        geglu_mul(&mut gate, &up);
        scratch.invalidate();
        sc.mlp.down.forward(&mut out, &gate, &mut scratch, n_canvas)?;
        Ok(out)
    }

    /// IterSmooth signal [n_canvas * n_embd]: each canvas position's expected
    /// token embedding from its previous-step logits, scaled by `alpha`. Unlike
    /// Gemma self-conditioning, IterSmooth applies no learned MLP and emits only
    /// the soft embedding.
    pub(crate) fn iter_smooth_signal(&self, prev_logits: &[f32], n_canvas: usize, alpha: f32) -> Vec<f32> {
        let e = self.n_embd;
        let mut out = vec![0.0f32; n_canvas * e];
        let mut emb = vec![0.0f32; e];
        let mut topk = Vec::with_capacity(SC_TOPK);
        for c in 0..n_canvas {
            let logits = &prev_logits[c * self.n_vocab..(c + 1) * self.n_vocab];
            let row = &mut out[c * e..(c + 1) * e];
            self.soft_embed(logits, &mut emb, row, &mut topk);
            for x in row.iter_mut() {
                *x *= alpha;
            }
        }
        out
    }

    /// Probability-weighted embedding of one token distribution, scaled by the
    /// embedding scale. Approximated over the top-SC_TOPK tokens (renormalized);
    /// `emb` and `top` are reused scratch. Writes the n_embd result to `out`.
    fn soft_embed(&self, logits: &[f32], emb: &mut [f32], out: &mut [f32], top: &mut Vec<(u32, f32)>) {
        top_k_logits_into(logits, SC_TOPK, top);
        let max = top[0].1;
        let sum: f32 = top.iter().map(|&(_, l)| (l - max).exp()).sum();
        out.fill(0.0);
        for &(token, logit) in top.iter() {
            let p = (logit - max).exp() / sum * self.embed_scale;
            self.embed_tokens(&[token as i32], emb);
            for (o, &x) in out.iter_mut().zip(emb.iter()) {
                *o += p * x;
            }
        }
    }
}

