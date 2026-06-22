use super::*;

// =============================================================================
// GGUF loading
// =============================================================================

/// Tensor access over a memory-mapped GGUF. K-quant weights are used in
/// place (zero-copy); small or fallback tensors are read via candle.
struct Loader {
    gguf: candle_core::quantized::gguf_file::Content,
    file: std::fs::File,
    mmap: Arc<memmap2::Mmap>,
    force_candle: bool,
}

impl Loader {
    fn has(&self, name: &str) -> bool {
        self.gguf.tensor_infos.contains_key(name)
    }

    fn shape_of(&self, name: &str) -> Result<(Vec<usize>, GgmlDType)> {
        let info = self
            .gguf
            .tensor_infos
            .get(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        Ok((info.shape.dims().to_vec(), info.ggml_dtype))
    }

    fn bytes(&self, name: &str) -> Result<Bytes> {
        let info = self
            .gguf
            .tensor_infos
            .get(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        let dtype = info.ggml_dtype;
        let size = info.shape.elem_count() / dtype.block_size() * dtype.type_size();
        let end = (self.gguf.tensor_data_offset as usize)
            .checked_add(info.offset as usize)
            .and_then(|start| start.checked_add(size).map(|end| (start, end)));
        let (start, end) = end.filter(|&(_, end)| end <= self.mmap.len())
            .with_context(|| format!("tensor {name} out of file bounds"))?;
        Ok(Bytes { mmap: self.mmap.clone(), range: start..end })
    }

    fn qtensor(&mut self, name: &str) -> Result<QTensor> {
        Ok(self.gguf.tensor(&mut self.file, name, &Device::Cpu)?)
    }

    fn f32_vec(&mut self, name: &str) -> Result<Vec<f32>> {
        let qt = self.qtensor(name)?;
        Ok(qt.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?)
    }

    fn mat(&mut self, name: &str) -> Result<MatWeight> {
        let (dims, dtype) = self.shape_of(name)?;
        if !self.force_candle && dims.len() == 2 {
            let native = || -> Result<NativeWeight> {
                Ok(NativeWeight { bytes: self.bytes(name)?, rows: dims[0], cols: dims[1] })
            };
            match dtype {
                GgmlDType::Q4K if dims[1] % QK_K == 0 => return Ok(MatWeight::Q4K(native()?)),
                GgmlDType::Q6K if dims[1] % QK_K == 0 => return Ok(MatWeight::Q6K(native()?)),
                GgmlDType::Q4_0 if dims[1] % 32 == 0 => return Ok(MatWeight::Q40(native()?)),
                _ => {}
            }
        }
        Ok(MatWeight::Candle(QMatMul::from_arc(Arc::new(self.qtensor(name)?))?))
    }

    fn experts(&mut self, name: &str) -> Result<ExpertMat> {
        let (dims, dtype) = self.shape_of(name)?;
        ensure!(dims.len() == 3, "expert tensor {name} must be 3D");
        let (n_expert, rows, cols) = (dims[0], dims[1], dims[2]);
        ensure!(cols % 32 == 0, "expert cols {cols} not a multiple of 32");
        let native = ExpertNative { bytes: self.bytes(name)?, n_expert, rows, cols };
        match dtype {
            GgmlDType::Q4K if cols % QK_K == 0 => Ok(ExpertMat::Q4K(native)),
            GgmlDType::Q6K if cols % QK_K == 0 => Ok(ExpertMat::Q6K(native)),
            GgmlDType::Q4_0 => Ok(ExpertMat::Q40(native)),
            GgmlDType::Q8_0 => Ok(ExpertMat::Q80(native)),
            GgmlDType::Q5_0 => Ok(ExpertMat::Q50(native)),
            other => bail!("unsupported expert dtype: {other:?} (Q4_K/Q6_K/Q4_0/Q5_0/Q8_0 only)"),
        }
    }

    fn embedding(&mut self) -> Result<(Embedding, usize)> {
        let (dims, dtype) = self.shape_of("token_embd.weight")?;
        ensure!(dims.len() == 2, "token_embd must be 2D");
        let wtype = match dtype {
            GgmlDType::F16 => EmbdType::F16,
            GgmlDType::F32 => EmbdType::F32,
            GgmlDType::Q4K => EmbdType::Q4K,
            GgmlDType::Q6K => EmbdType::Q6K,
            GgmlDType::Q4_0 => EmbdType::Q40,
            GgmlDType::Q8_0 => EmbdType::Q80,
            other => bail!("unsupported token_embd dtype: {other:?}"),
        };
        Ok((Embedding { bytes: self.bytes("token_embd.weight")?, wtype }, dims[0]))
    }
}

fn load_ffn(loader: &mut Loader, gate: &str, up: &str, down: &str) -> Result<FfnWeights> {
    let (dims, _) = loader.shape_of(gate)?;
    ensure!(dims.len() == 2, "ffn gate must be 2D");
    Ok(FfnWeights {
        gate: loader.mat(gate)?,
        up: loader.mat(up)?,
        down: loader.mat(down)?,
        ff: dims[0],
    })
}

/// Read an f32 vector tensor if present, else None.
fn opt_f32_vec(loader: &mut Loader, name: &str) -> Option<Vec<f32>> {
    loader.has(name).then(|| loader.f32_vec(name).ok()).flatten()
}

/// Coerce a GGUF metadata value to usize across integer widths (arrays may be
/// stored as any width).
fn value_to_usize(v: &gguf_file::Value) -> Option<usize> {
    v.to_u32()
        .ok()
        .map(|n| n as usize)
        .or_else(|| v.to_i32().ok().map(|n| n as usize))
        .or_else(|| v.to_u64().ok().map(|n| n as usize))
        .or_else(|| v.to_i64().ok().map(|n| n as usize))
        .or_else(|| v.to_u16().ok().map(|n| n as usize))
        .or_else(|| v.to_i16().ok().map(|n| n as usize))
        .or_else(|| v.to_u8().ok().map(|n| n as usize))
        .or_else(|| v.to_i8().ok().map(|n| n as usize))
}

/// Arch-aware metadata accessor: looks up `diffuse.{key}` then `{arch}.{key}`.
struct Meta<'a> {
    gguf: &'a gguf_file::Content,
    arch: String,
}

impl<'a> Meta<'a> {
    fn new(gguf: &'a gguf_file::Content) -> Self {
        let arch = gguf
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "diffuse".to_string());
        Self { gguf, arch }
    }
    fn find(&self, key: &str) -> Option<&gguf_file::Value> {
        self.gguf
            .metadata
            .get(&format!("diffuse.{key}"))
            .or_else(|| self.gguf.metadata.get(&format!("{}.{key}", self.arch)))
    }
    fn u32(&self, key: &str) -> Option<u32> {
        self.find(key).and_then(|v| v.to_u32().ok())
    }
    fn req_u32(&self, key: &str) -> Result<u32> {
        self.u32(key).with_context(|| format!("missing: {key}"))
    }
    fn usize(&self, key: &str) -> Option<usize> {
        self.u32(key).map(|v| v as usize)
    }
    fn f32(&self, key: &str, default: f32) -> f32 {
        self.find(key).and_then(|v| v.to_f32().ok()).unwrap_or(default)
    }
    fn bool(&self, key: &str, default: bool) -> bool {
        self.find(key).and_then(|v| v.to_bool().ok()).unwrap_or(default)
    }
    fn raw_u32(&self, key: &str) -> Option<u32> {
        self.gguf.metadata.get(key).and_then(|v| v.to_u32().ok())
    }
    fn raw_bool(&self, key: &str) -> Option<bool> {
        self.gguf.metadata.get(key).and_then(|v| v.to_bool().ok())
    }
    fn raw_string(&self, key: &str) -> Option<String> {
        self.gguf.metadata.get(key).and_then(|v| v.to_string().ok().cloned())
    }
    fn usize_array(&self, key: &str) -> Option<Vec<usize>> {
        self.find(key)
            .and_then(|v| v.to_vec().ok())
            .map(|vs| vs.iter().filter_map(value_to_usize).collect())
    }
    fn bool_array(&self, key: &str) -> Option<Vec<bool>> {
        self.find(key)
            .and_then(|v| v.to_vec().ok())
            .map(|vs| vs.iter().filter_map(|x| x.to_bool().ok()).collect())
    }
}

/// Scalar model configuration parsed from GGUF metadata, before any weights.
struct Config {
    model_type: String,
    logit_shift: bool,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    n_layer: usize,
    n_ff: usize,
    head_dim: usize,
    rope_dim: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    mask_token_id: u32,
    eos_token_id: Option<i32>,
    context_length: Option<usize>,
    canvas_length: Option<usize>,
    n_vocab_meta: Option<usize>,
    logit_softcap: f32,
    is_gemma: bool,
    embed_scale: f32,
    moe_cfg: Option<MoeConfig>,
}

impl Config {
    fn read(meta: &Meta) -> Result<Config> {
        let n_embd = meta.req_u32("embedding_length")? as usize;
        let n_head = meta.req_u32("attention.head_count")? as usize;
        ensure!(n_head > 0, "attention.head_count must be positive");
        // Qwen3-family models (RND1) declare an explicit head width that is
        // not n_embd / n_head; honor it when present.
        let head_dim = meta.usize("attention.key_length").unwrap_or(n_embd / n_head);
        let arch = meta.arch.clone();
        let model_type = meta.raw_string("diffuse.model_type").unwrap_or_else(|| arch.clone());
        // DIFFUSE_SHIFT=1/0 overrides metadata (some conversions mislabel it).
        // Dream and RND1 are AR-initialized (Qwen lineage) and predict token i
        // from row i-1; the shift defaults on for them when metadata is absent.
        let logit_shift = match std::env::var("DIFFUSE_SHIFT").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => meta
                .raw_bool("diffusion.shift_logits")
                .unwrap_or(matches!(model_type.as_str(), "dream" | "rnd1")),
        };
        // MoE defaults match the LLaDA-MoE reference (norm_topk_prob unset).
        // Qwen3-family MoE (RND1) normalizes the top-k weights but its GGUF
        // carries no expert_weights_norm key; llama.cpp hardcodes this per arch.
        let weights_norm_default = model_type == "rnd1";
        let moe_cfg = meta.u32("expert_count").filter(|&c| c > 0).map(|n_expert| {
            let n_expert = n_expert as usize;
            // A group count above the expert count (or zero) would make the
            // per-group size zero and silently route every token to no expert.
            let n_group = (meta.u32("expert_group_count").unwrap_or(1) as usize).clamp(1, n_expert);
            MoeConfig {
                n_expert,
                n_used: meta.u32("expert_used_count").unwrap_or(1) as usize,
                weights_norm: meta.bool("expert_weights_norm", weights_norm_default),
                gating_sigmoid: meta.u32("expert_gating_func").unwrap_or(1) == 2,
                scale: meta.f32("expert_weights_scale", 1.0),
                n_group,
                group_used: (meta.u32("expert_group_used_count").unwrap_or(1) as usize).clamp(1, n_group),
            }
        });
        let is_gemma = arch == "diffusion-gemma";
        Ok(Config {
            model_type,
            logit_shift,
            n_embd,
            n_head,
            n_head_kv: meta.usize("attention.head_count_kv").unwrap_or(n_head),
            n_layer: meta.req_u32("block_count")? as usize,
            n_ff: meta.usize("feed_forward_length").unwrap_or(0),
            head_dim,
            rope_dim: meta.usize("rope.dimension_count").unwrap_or(head_dim),
            rope_theta: meta.f32("rope.freq_base", 500000.0),
            rms_norm_eps: meta.f32("attention.layer_norm_rms_epsilon", 1e-5),
            mask_token_id: meta
                .u32("mask_token_id")
                .or_else(|| meta.raw_u32("tokenizer.ggml.mask_token_id"))
                .context("missing mask_token_id")?,
            eos_token_id: meta
                .u32("eos_token_id")
                .or_else(|| meta.raw_u32("tokenizer.ggml.eos_token_id"))
                .map(|id| id as i32),
            context_length: meta.usize("context_length"),
            canvas_length: meta.raw_u32("diffusion.canvas_length").map(|v| v as usize),
            n_vocab_meta: meta.usize("vocab_size"),
            logit_softcap: if is_gemma { meta.f32("final_logit_softcapping", 0.0) } else { 0.0 },
            is_gemma,
            embed_scale: if is_gemma { (n_embd as f32).sqrt() } else { 1.0 },
            moe_cfg,
        })
    }

    fn log_arch(&self) {
        let desc = match self.model_type.as_str() {
            "dream" => "Dream (Qwen2.5, GQA)",
            "llada-moe" => "LLaDA-MoE",
            "llada2" => "LLaDA2.0 (grouped-expert MoE)",
            "mdlm" => "MDLM (masked discrete)",
            "diffusion-gemma" => "DiffusionGemma (gemma4, dual dense+MoE)",
            "rnd1" => "RND1 (Qwen3 MoE, AR-converted)",
            _ => "LLaDA (Llama)",
        };
        eprintln!(
            "[diffuse-rs] {desc}: embd={} heads={}/{} layers={} moe={:?}",
            self.n_embd, self.n_head, self.n_head_kv, self.n_layer, self.moe_cfg
        );
    }
}

/// rope_freqs.weight as f32, if present (Gemma global-layer freq factors).
fn read_rope_freqs(gguf: &gguf_file::Content, file: &mut std::fs::File) -> Option<Vec<f32>> {
    if !gguf.tensor_infos.contains_key("rope_freqs.weight") {
        return None;
    }
    gguf.tensor(file, "rope_freqs.weight", &Device::Cpu)
        .ok()
        .and_then(|qt| qt.dequantize(&Device::Cpu).ok())
        .and_then(|t| t.flatten_all().ok())
        .and_then(|t| t.to_vec1::<f32>().ok())
}

/// Per-layer attention geometry. Must run before the Loader move: it reads
/// metadata arrays and the rope_freqs tensor through `meta.gguf` + `&mut file`.
/// Gemma alternates sliding (narrow head, windowed, low theta) and global
/// (wide head, full, high theta) layers; others repeat one config + RoPE table.
fn build_layer_attn(meta: &Meta, file: &mut std::fs::File, cfg: &Config) -> Result<Vec<LayerAttn>> {
    let rope_len = cfg.context_length.unwrap_or(4096).clamp(4096, crate::model::MAX_TOTAL_LEN);
    if !cfg.is_gemma {
        let rope = Arc::new(build_rope_cache(rope_len, cfg.rope_dim, cfg.rope_theta));
        return Ok(vec![
            LayerAttn {
                n_head: cfg.n_head,
                n_head_kv: cfg.n_head_kv,
                head_dim: cfg.head_dim,
                rope_dim: cfg.rope_dim,
                sliding_window: None,
                rope,
            };
            cfg.n_layer
        ]);
    }

    let kv = meta.usize_array("attention.head_count_kv").context("gemma: head_count_kv array")?;
    // sliding_window_pattern[i] == true marks a sliding (local) layer.
    let pattern =
        meta.bool_array("attention.sliding_window_pattern").context("gemma: sliding_window_pattern")?;
    let hd_global = meta.usize("attention.key_length").unwrap_or(cfg.head_dim);
    let hd_sliding = meta.usize("attention.key_length_swa").unwrap_or(hd_global);
    let rd_global = meta.usize("rope.dimension_count").unwrap_or(hd_global);
    let rd_sliding = meta.usize("rope.dimension_count_swa").unwrap_or(hd_sliding);
    let theta_sliding = meta.f32("rope.freq_base_swa", 10000.0);
    let window = meta.usize("attention.sliding_window");
    let rope_freqs = read_rope_freqs(meta.gguf, file);
    let rope_global =
        Arc::new(build_rope_cache_ff(rope_len, rd_global, cfg.rope_theta, rope_freqs.as_deref()));
    let rope_sliding = Arc::new(build_rope_cache(rope_len, rd_sliding, theta_sliding));

    Ok((0..cfg.n_layer)
        .map(|i| {
            let sliding = *pattern.get(i).unwrap_or(&true);
            let n_head_kv = *kv.get(i).unwrap_or(kv.first().unwrap_or(&cfg.n_head));
            if sliding {
                LayerAttn {
                    n_head: cfg.n_head,
                    n_head_kv,
                    head_dim: hd_sliding,
                    rope_dim: rd_sliding,
                    sliding_window: window,
                    rope: rope_sliding.clone(),
                }
            } else {
                LayerAttn {
                    n_head: cfg.n_head,
                    n_head_kv,
                    head_dim: hd_global,
                    rope_dim: rd_global,
                    sliding_window: None,
                    rope: rope_global.clone(),
                }
            }
        })
        .collect())
}

/// QKV projections for one block: fused, or split (Gemma global layers omit
/// attn_v and reuse the K projection as V).
fn load_qkv(loader: &mut Loader, p: &str) -> Result<Qkv> {
    if loader.has(&format!("{p}.attn_qkv.weight")) {
        return Ok(Qkv::Fused(loader.mat(&format!("{p}.attn_qkv.weight"))?));
    }
    let v_name = if loader.has(&format!("{p}.attn_v.weight")) {
        format!("{p}.attn_v.weight")
    } else {
        format!("{p}.attn_k.weight")
    };
    Ok(Qkv::Split {
        wq: loader.mat(&format!("{p}.attn_q.weight"))?,
        wk: loader.mat(&format!("{p}.attn_k.weight"))?,
        wv: loader.mat(&v_name)?,
    })
}

/// Load one block's FFN, returning it and the widest intermediate it uses.
fn load_block_ffn(loader: &mut Loader, p: &str, cfg: &Config) -> Result<(Ffn, usize)> {
    if cfg.is_gemma {
        // Dense shared expert + fused-expert MoE in parallel (docs/diffusiongemma.md).
        // Gemma renormalizes the selected experts' softmax weights (norm_w=true).
        let mut moe = cfg.moe_cfg.context("diffusion-gemma needs expert_count")?;
        moe.weights_norm = true;
        let dense = load_ffn(
            loader,
            &format!("{p}.ffn_gate.weight"),
            &format!("{p}.ffn_up.weight"),
            &format!("{p}.ffn_down.weight"),
        )?;
        let gate_up = loader.experts(&format!("{p}.ffn_gate_up_exps.weight"))?;
        let down = loader.experts(&format!("{p}.ffn_down_exps.weight"))?;
        let ff_exp = gate_up.rows() / 2;
        let max_ff = gate_up.rows().max(dense.ff);
        let out_scale = loader
            .f32_vec(&format!("{p}.layer_output_scale.weight"))?
            .first()
            .copied()
            .unwrap_or(1.0);
        let gemma = GemmaFfn {
            dense,
            dense_post_norm: loader.f32_vec(&format!("{p}.post_ffw_norm_1.weight"))?,
            moe_pre_norm: loader.f32_vec(&format!("{p}.pre_ffw_norm_2.weight"))?,
            router: loader.f32_vec(&format!("{p}.ffn_gate_inp.weight"))?,
            router_scale: loader.f32_vec(&format!("{p}.ffn_gate_inp.scale"))?,
            gate_up,
            down,
            down_scale: loader.f32_vec(&format!("{p}.ffn_down_exps.scale"))?,
            moe_post_norm: loader.f32_vec(&format!("{p}.post_ffw_norm_2.weight"))?,
            post_norm: loader.f32_vec(&format!("{p}.post_ffw_norm.weight"))?,
            out_scale,
            cfg: moe,
            ff_exp,
        };
        return Ok((Ffn::Gemma(Box::new(gemma)), max_ff));
    }

    if loader.has(&format!("{p}.ffn_gate_inp.weight")) {
        let moe = cfg.moe_cfg.context("expert tensors present but no expert_count")?;
        let gate = loader.experts(&format!("{p}.ffn_gate_exps.weight"))?;
        let up = loader.experts(&format!("{p}.ffn_up_exps.weight"))?;
        let down = loader.experts(&format!("{p}.ffn_down_exps.weight"))?;
        let shared = if loader.has(&format!("{p}.ffn_gate_shexp.weight")) {
            Some(load_ffn(
                loader,
                &format!("{p}.ffn_gate_shexp.weight"),
                &format!("{p}.ffn_up_shexp.weight"),
                &format!("{p}.ffn_down_shexp.weight"),
            )?)
        } else {
            None
        };
        let max_ff = gate.rows().max(shared.as_ref().map_or(0, |s| s.ff));
        let bias_name = format!("{p}.exp_probs_b.bias");
        let weights = MoeWeights {
            router: loader.f32_vec(&format!("{p}.ffn_gate_inp.weight"))?,
            sel_bias: if loader.has(&bias_name) { Some(loader.f32_vec(&bias_name)?) } else { None },
            gate,
            up,
            down,
            shared,
            cfg: moe,
        };
        return Ok((Ffn::Moe(Box::new(weights)), max_ff));
    }

    let ffn = load_ffn(
        loader,
        &format!("{p}.ffn_gate.weight"),
        &format!("{p}.ffn_up.weight"),
        &format!("{p}.ffn_down.weight"),
    )?;
    let max_ff = ffn.ff;
    Ok((Ffn::Dense(ffn), max_ff))
}

/// Load one transformer block; returns the layer and its widest FFN width.
fn load_block(loader: &mut Loader, idx: usize, cfg: &Config) -> Result<(Layer, usize)> {
    let p = format!("blk.{idx}");
    let (ffn, ffn_max) = load_block_ffn(loader, &p, cfg)?;
    let layer = Layer {
        attn_norm: loader.f32_vec(&format!("{p}.attn_norm.weight"))?,
        ffn_norm: loader.f32_vec(&format!("{p}.ffn_norm.weight"))?,
        bq: opt_f32_vec(loader, &format!("{p}.attn_q.bias")),
        bk: opt_f32_vec(loader, &format!("{p}.attn_k.bias")),
        bv: opt_f32_vec(loader, &format!("{p}.attn_v.bias")),
        q_norm: opt_f32_vec(loader, &format!("{p}.attn_q_norm.weight")),
        k_norm: opt_f32_vec(loader, &format!("{p}.attn_k_norm.weight")),
        post_attn_norm: opt_f32_vec(loader, &format!("{p}.post_attention_norm.weight")),
        qkv: load_qkv(loader, &p)?,
        wo: loader.mat(&format!("{p}.attn_output.weight"))?,
        ffn,
    };
    Ok((layer, ffn_max))
}

/// Self-conditioning weights (Gemma). The pre-norm is a plain RMSNorm here.
fn load_self_cond(loader: &mut Loader) -> Result<Option<SelfCond>> {
    if !loader.has("self_cond_gate.weight") {
        return Ok(None);
    }
    Ok(Some(SelfCond {
        pre_norm: loader.f32_vec("self_cond_pre_norm.weight")?,
        mlp: load_ffn(loader, "self_cond_gate.weight", "self_cond_up.weight", "self_cond_down.weight")?,
    }))
}

impl Model {
    pub fn from_gguf(path: &str) -> Result<Self> {
        eprintln!("[diffuse-rs] loading {path}");
        let mut file =
            std::fs::File::open(path).with_context(|| format!("failed to open {path}"))?;
        let gguf = gguf_file::Content::read(&mut file)?;

        // Metadata keys: our converters write "diffuse.*"; llama.cpp GGUFs
        // (llada, dream, llada-moe) write "{arch}.*" plus tokenizer.ggml.*.
        let meta = Meta::new(&gguf);
        let cfg = Config::read(&meta)?;
        cfg.log_arch();

        // Escape hatch for A/B benchmarking the matmul paths.
        let force_candle = std::env::var("DIFFUSE_MATMUL").as_deref() == Ok("candle");
        if force_candle {
            eprintln!("[diffuse-rs] DIFFUSE_MATMUL=candle: using candle QMatMul for all weights");
        }

        // Attention geometry must be built before the Loader move (it consumes
        // gguf); it reads metadata arrays + rope_freqs through &gguf + &mut file.
        // `meta` is unused past this call, so its borrow of gguf ends here and
        // gguf may move into the Loader below.
        let attn = build_layer_attn(&meta, &mut file, &cfg)?;

        // SAFETY: the mmap is read-only; mutating the GGUF while loaded is
        // unsupported (same contract as llama.cpp).
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file)? });
        let mut loader = Loader { gguf, file, mmap, force_candle };

        let (tok_embd, embd_rows) = loader.embedding()?;
        // Clamp to the actual embedding row count. A GGUF whose metadata
        // vocab_size exceeds the table would otherwise let a token id past the
        // end of the embedding rows through validation and into the unchecked
        // pointer read in the quantized embedding path.
        let n_vocab = cfg.n_vocab_meta.map_or(embd_rows, |v| v.min(embd_rows));
        eprintln!("[diffuse-rs] vocab={n_vocab} token_embd wtype={:?} (mmap)", tok_embd.wtype);

        let output = match loader.mat("output.weight") {
            Ok(w) => w,
            Err(_) => {
                eprintln!("[diffuse-rs] output = token_embd (tied)");
                loader.mat("token_embd.weight")?
            }
        };

        let mut max_ff = cfg.n_ff.max(1);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let (layer, ffn_max) = load_block(&mut loader, i, &cfg)?;
            max_ff = max_ff.max(ffn_max);
            layers.push(layer);
        }
        eprintln!("[diffuse-rs] loaded {} layers", cfg.n_layer);

        let self_cond = load_self_cond(&mut loader)?;
        let output_norm = loader.f32_vec("output_norm.weight")?;

        Ok(Self {
            logit_shift: cfg.logit_shift,
            model_type: cfg.model_type,
            n_vocab,
            n_embd: cfg.n_embd,
            n_head: cfg.n_head,
            n_layer: cfg.n_layer,
            n_ff: cfg.n_ff,
            mask_token_id: cfg.mask_token_id,
            eos_token_id: cfg.eos_token_id,
            rms_norm_eps: cfg.rms_norm_eps,
            max_positions: cfg.context_length.unwrap_or(4096).clamp(4096, crate::model::MAX_TOTAL_LEN),
            context_length: cfg.context_length,
            canvas_length: cfg.canvas_length,
            attn,
            is_gemma: cfg.is_gemma,
            embed_scale: cfg.embed_scale,
            logit_softcap: cfg.logit_softcap,
            canvas_start: 0,
            self_cond,
            sc_signal: None,
            smooth_signal: None,
            max_ff,
            tok_embd,
            output_norm,
            output,
            layers,
            bufs: None,
        })
    }
}

