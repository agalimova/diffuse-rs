use super::*;

/// Identity sequence positions [0, n) for full-forward attention with no cache.
fn idpos(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// Single-sequence row metadata over positions [0, n).
fn rows(pos: &[usize]) -> Rows<'_> {
    Rows { pos, seq: &[], prefix: &[] }
}

#[test]
fn test_schedule_unmasks_everything() {
    for schedule in [Schedule::Cosine, Schedule::Linear] {
        for total_steps in [1, 4, 16] {
            let mut masked = 128usize;
            for step in 0..total_steps {
                masked -= tokens_to_unmask(step, total_steps, masked, schedule);
            }
            assert_eq!(masked, 0, "{schedule:?} with {total_steps} steps");
        }
    }
}

#[test]
fn test_entropy_uniform() {
    let logits = vec![0.0f32; 16];
    assert!((compute_entropy(&logits) - 16f32.ln()).abs() < 1e-5);
}

#[test]
fn test_entropy_peaked() {
    let mut logits = vec![0.0f32; 16];
    logits[3] = 100.0;
    assert!(compute_entropy(&logits) < 1e-3);
}

#[test]
fn test_split_active() {
    let mut cache = StepCache::new(4, vec![8]);
    cache.update_seq(&[10, 20, -1, -1]);
    // Position 1 changed, 2 and 3 still masked, 0 stable.
    let split = cache.split_active(&[10, 21, -1, -1], &[false, false, true, true], 5);
    assert_eq!(split.active, vec![1, 2, 3]);
    assert_eq!(split.cached, vec![0]);
    // EXTRA_ACTIVE_STEPS later, position 1 (unchanged since) goes back to cached.
    cache.update_seq(&[10, 21, -1, -1]);
    let later = 5 + EXTRA_ACTIVE_STEPS + 1;
    let split = cache.split_active(&[10, 21, -1, -1], &[false, false, true, true], later);
    assert_eq!(split.active, vec![2, 3]);
    assert_eq!(split.cached, vec![0, 1]);
}

#[test]
fn test_cache_store_gather_roundtrip() {
    let stride = 4;
    let mut cache = StepCache::new(3, vec![stride]);
    let k: Vec<f32> = (0..8).map(|x| x as f32).collect();
    let v: Vec<f32> = (0..8).map(|x| (x + 100) as f32).collect();
    cache.store(0, &[2, 0], &k, &v);

    let mut out_k = vec![0.0; 8];
    let mut out_v = vec![0.0; 8];
    cache.gather(0, &[0, 2], &mut out_k, &mut out_v);
    assert_eq!(out_k, vec![4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0]);
    assert_eq!(out_v, vec![104.0, 105.0, 106.0, 107.0, 100.0, 101.0, 102.0, 103.0]);
}

#[test]
fn test_attention_block_diagonal_batches() {
    // Two sequences sharing one forward pass must not attend across each
    // other: batched output equals running each sequence on its own.
    let hd = 4;
    let (na, nb) = (2usize, 3usize);
    let shape = |nq, nk| AttnShape {
        nq, nk, n_head: 1, n_head_kv: 1, head_dim: hd,
        sliding_window: None, causal_prefix: None, scale: 1.0,
    };
    let q: Vec<f32> = (0..(na + nb) * hd).map(|x| ((x * 37 % 101) as f32 - 50.0) / 50.0).collect();
    let k = q.clone();
    let v: Vec<f32> = (0..(na + nb) * hd).map(|x| ((x * 71 % 89) as f32 - 44.0) / 44.0).collect();

    // Batched: seq ids [0,0,1,1,1], per-sequence positions [0,1,0,1,2].
    let pos = vec![0, 1, 0, 1, 2];
    let seq = vec![0, 0, 1, 1, 1];
    let br = Rows { pos: &pos, seq: &seq, prefix: &[] };
    let mut batched = vec![0.0; (na + nb) * hd];
    attention(&mut batched, &q, &k, &v, shape(na + nb, na + nb), br, br);

    let mut sep_a = vec![0.0; na * hd];
    attention(&mut sep_a, &q[..na * hd], &k[..na * hd], &v[..na * hd], shape(na, na), rows(&idpos(na)), rows(&idpos(na)));
    let mut sep_b = vec![0.0; nb * hd];
    attention(&mut sep_b, &q[na * hd..], &k[na * hd..], &v[na * hd..], shape(nb, nb), rows(&idpos(nb)), rows(&idpos(nb)));

    for d in 0..na * hd {
        assert!((batched[d] - sep_a[d]).abs() < 1e-5, "seq A row {d}");
    }
    for d in 0..nb * hd {
        assert!((batched[na * hd + d] - sep_b[d]).abs() < 1e-5, "seq B row {d}");
    }
}

#[test]
fn test_attention_uniform_scores() {
    // Identical keys -> uniform attention -> output is mean of values.
    let s = AttnShape { nq: 1, nk: 2, n_head: 1, n_head_kv: 1, head_dim: 2, sliding_window: None, causal_prefix: None, scale: 1.0 };
    let q = vec![1.0, 0.0];
    let k = vec![1.0, 0.0, 1.0, 0.0];
    let v = vec![0.0, 2.0, 4.0, 6.0];
    let mut out = vec![0.0; 2];
    attention(&mut out, &q, &k, &v, s, rows(&idpos(s.nq)), rows(&idpos(s.nk)));
    assert!((out[0] - 2.0).abs() < 1e-6 && (out[1] - 4.0).abs() < 1e-6);
}

#[test]
fn test_attention_gqa_matches_repeated_kv() {
    // 4 query heads sharing 2 kv heads must equal plain MHA with the
    // kv heads explicitly repeated.
    let (nq, nk, hd) = (3, 5, 8);
    let gqa = AttnShape { nq, nk, n_head: 4, n_head_kv: 2, head_dim: hd, sliding_window: None, causal_prefix: None, scale: 1.0 };
    let mha = AttnShape { nq, nk, n_head: 4, n_head_kv: 4, head_dim: hd, sliding_window: None, causal_prefix: None, scale: 1.0 };

    let q: Vec<f32> = (0..nq * 4 * hd).map(|x| ((x * 37 % 101) as f32 - 50.0) / 50.0).collect();
    let kv_small: Vec<f32> =
        (0..nk * 2 * hd).map(|x| ((x * 53 % 97) as f32 - 48.0) / 48.0).collect();
    let v_small: Vec<f32> =
        (0..nk * 2 * hd).map(|x| ((x * 71 % 89) as f32 - 44.0) / 44.0).collect();

    // Repeat each kv head twice: [a, b] -> [a, a, b, b]
    let repeat = |src: &[f32]| -> Vec<f32> {
        let mut dst = vec![0.0; nk * 4 * hd];
        for t in 0..nk {
            for kvh in 0..2 {
                for r in 0..2 {
                    let s_off = t * 2 * hd + kvh * hd;
                    let d_off = t * 4 * hd + (kvh * 2 + r) * hd;
                    dst[d_off..d_off + hd].copy_from_slice(&src[s_off..s_off + hd]);
                }
            }
        }
        dst
    };

    let mut out_a = vec![0.0; nq * 4 * hd];
    let mut out_b = out_a.clone();
    attention(&mut out_a, &q, &kv_small, &v_small, gqa, rows(&idpos(nq)), rows(&idpos(nk)));
    attention(&mut out_b, &q, &repeat(&kv_small), &repeat(&v_small), mha, rows(&idpos(nq)), rows(&idpos(nk)));
    for (a, b) in out_a.iter().zip(&out_b) {
        assert!((a - b).abs() < 1e-5);
    }
}

/// Online softmax must match the naive three-pass reference.
#[test]
fn test_attention_matches_naive_reference() {
    let s = AttnShape { nq: 4, nk: 6, n_head: 2, n_head_kv: 2, head_dim: 8, sliding_window: None, causal_prefix: None, scale: 1.0 };
    let (h, hd) = (s.n_head, s.head_dim);
    let q: Vec<f32> = (0..s.nq * h * hd).map(|x| ((x * 37 % 101) as f32 - 50.0) / 25.0).collect();
    let k: Vec<f32> = (0..s.nk * h * hd).map(|x| ((x * 53 % 97) as f32 - 48.0) / 24.0).collect();
    let v: Vec<f32> = (0..s.nk * h * hd).map(|x| ((x * 71 % 89) as f32 - 44.0) / 44.0).collect();

    let mut out = vec![0.0; s.nq * h * hd];
    attention(&mut out, &q, &k, &v, s, rows(&idpos(s.nq)), rows(&idpos(s.nk)));

    let scale = 1.0f32; // matches AttnShape.scale above
    for head in 0..h {
        for i in 0..s.nq {
            let qo = i * h * hd + head * hd;
            let mut scores: Vec<f32> = (0..s.nk)
                .map(|j| {
                    let ko = j * h * hd + head * hd;
                    (0..hd).map(|d| q[qo + d] * k[ko + d]).sum::<f32>() * scale
                })
                .collect();
            softmax_row_scalar(&mut scores);
            for d in 0..hd {
                let want: f32 = (0..s.nk)
                    .map(|j| scores[j] * v[j * h * hd + head * hd + d])
                    .sum();
                let got = out[qo + d];
                assert!((got - want).abs() < 1e-5, "head {head} q {i} d {d}: {got} vs {want}");
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_attention_avx2_matches_scalar() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let s = AttnShape { nq: 3, nk: 3, n_head: 2, n_head_kv: 2, head_dim: 16, sliding_window: None, causal_prefix: None, scale: 1.0 };
    let len = s.nq * s.n_head * s.head_dim;
    let q: Vec<f32> = (0..len).map(|x| ((x * 37 % 101) as f32 - 50.0) / 50.0).collect();
    let k: Vec<f32> = (0..len).map(|x| ((x * 53 % 97) as f32 - 48.0) / 48.0).collect();
    let v: Vec<f32> = (0..len).map(|x| ((x * 71 % 89) as f32 - 44.0) / 44.0).collect();

    let (qpos, kpos) = (idpos(s.nq), idpos(s.nk));
    let attn = Attn { q: &q, k: &k, v: &v, s, qr: rows(&qpos), kr: rows(&kpos) };
    let mut out_a = vec![0.0f32; len];
    let mut out_b = vec![0.0f32; len];
    for head in 0..s.n_head {
        for i in 0..s.nq {
            attention_row_scalar(SendPtr(out_a.as_mut_ptr()), attn, head, i);
            unsafe { attention_row_avx2(SendPtr(out_b.as_mut_ptr()), attn, head, i) };
        }
    }
    for (a, b) in out_a.iter().zip(&out_b) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_attention_neon_matches_scalar() {
    let s = AttnShape { nq: 3, nk: 3, n_head: 2, n_head_kv: 2, head_dim: 16, sliding_window: None, causal_prefix: None, scale: 1.0 };
    let len = s.nq * s.n_head * s.head_dim;
    let q: Vec<f32> = (0..len).map(|x| ((x * 37 % 101) as f32 - 50.0) / 50.0).collect();
    let k: Vec<f32> = (0..len).map(|x| ((x * 53 % 97) as f32 - 48.0) / 48.0).collect();
    let v: Vec<f32> = (0..len).map(|x| ((x * 71 % 89) as f32 - 44.0) / 44.0).collect();

    let (qpos, kpos) = (idpos(s.nq), idpos(s.nk));
    let attn = Attn { q: &q, k: &k, v: &v, s, qr: rows(&qpos), kr: rows(&kpos) };
    let mut out_a = vec![0.0f32; len];
    let mut out_b = vec![0.0f32; len];
    for head in 0..s.n_head {
        for i in 0..s.nq {
            attention_row_scalar(SendPtr(out_a.as_mut_ptr()), attn, head, i);
            unsafe { attention_row_neon(SendPtr(out_b.as_mut_ptr()), attn, head, i) };
        }
    }
    for (a, b) in out_a.iter().zip(&out_b) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_neon_ops_match_scalar() {
    let x: Vec<f32> = (0..70).map(|i| ((i * 17 % 91) as f32 - 45.0) / 30.0).collect();
    let w: Vec<f32> = (0..70).map(|i| ((i * 13 % 53) as f32 - 26.0) / 26.0).collect();
    let mut a = vec![0.0f32; 70];
    let mut b = vec![0.0f32; 70];
    rms_norm_scalar(&mut a, &x, &w, 1e-6);
    unsafe { rms_norm_neon(&mut b, &x, &w, 1e-6) };
    for (p, q) in a.iter().zip(&b) {
        assert!((p - q).abs() < 1e-5, "rms_norm: {p} vs {q}");
    }

    let mut g2 = x.clone();
    let g1: Vec<f32> = x.iter().zip(&w).map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv).collect();
    unsafe { silu_mul_neon(&mut g2, &w) };
    for (p, q) in g1.iter().zip(&g2) {
        assert!((p - q).abs() < 1e-5, "silu_mul: {p} vs {q}");
    }
}

#[test]
fn test_distribute_steps() {
    assert_eq!(distribute_steps(16, 4), vec![4, 4, 4, 4]);
    assert_eq!(distribute_steps(10, 3), vec![4, 3, 3]);
    // Fewer steps than blocks: every block still gets at least one.
    assert_eq!(distribute_steps(2, 5), vec![1, 1, 1, 1, 1]);
}

#[test]
fn test_truncate_at_eos() {
    let mut tokens = vec![5, 6, 99, 7, 99];
    assert!(truncate_at_eos(&mut tokens, Some(99)));
    assert_eq!(tokens, vec![5, 6]);

    let mut tokens = vec![5, 6, 7];
    assert!(!truncate_at_eos(&mut tokens, Some(99)));
    assert_eq!(tokens, vec![5, 6, 7]);
    assert!(!truncate_at_eos(&mut tokens, None));
}

#[test]
fn test_top2_logits() {
    assert_eq!(top2_logits(&[1.0, 5.0, 3.0, 5.0]), (5.0, 5.0));
    assert_eq!(top2_logits(&[2.0, -1.0]), (2.0, -1.0));
}

#[test]
fn test_log_sum_exp_and_prob() {
    // Uniform pair: lse = ln(2), p(top1) = 0.5
    let lse = log_sum_exp(&[0.0, 0.0]);
    assert!((lse - 2f32.ln()).abs() < 1e-6);
    assert!(((0.0 - lse).exp() - 0.5).abs() < 1e-6);
}

#[test]
fn test_top_k_logits() {
    let top = top_k_logits(&[1.0, 9.0, 3.0, 7.0], 2);
    assert_eq!(top, vec![(1, 9.0), (3, 7.0)]);
    // k larger than the row: everything, sorted
    let all = top_k_logits(&[1.0, 2.0], 10);
    assert_eq!(all, vec![(1, 2.0), (0, 1.0)]);
}

#[test]
fn test_nucleus_prefix() {
    // probs ~ [0.72, 0.27, 0.007, ...]: top_p=0.9 keeps two entries
    let row = [2.0f32, 1.0, -2.0, -10.0];
    let sorted = top_k_logits(&row, 4);
    let kept = nucleus_prefix(&sorted, log_sum_exp(&row), 0.9);
    assert_eq!(kept.len(), 2);
    // Always keeps at least one entry
    let kept = nucleus_prefix(&sorted, log_sum_exp(&row), 0.0);
    assert_eq!(kept.len(), 1);
}

#[test]
fn test_q8k_layout_matches_candle() {
    assert_eq!(
        std::mem::size_of::<BlockQ8K>(),
        std::mem::size_of::<k_quants::BlockQ8K>()
    );
    assert_eq!(
        std::mem::align_of::<BlockQ8K>(),
        std::mem::align_of::<k_quants::BlockQ8K>()
    );
}

#[test]
fn test_dequantize_q4k_matches_candle() {
    // dequantize_q4k_row backs Q4_K token embeddings; pin it against
    // candle's own dequantization of the same quantized bytes.
    let data: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) / 50.0).collect();
    let t = Tensor::from_vec(data, &[QK_K], &Device::Cpu).unwrap();
    let qt = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();
    let reference = qt.dequantize(&Device::Cpu).unwrap().to_vec1::<f32>().unwrap();

    let bytes = qt.data().unwrap();
    let blocks = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const BlockQ4K, 1) };
    let mut mine = vec![0.0f32; QK_K];
    kernels::dequantize_q4k_row(blocks, &mut mine);
    for (a, b) in mine.iter().zip(&reference) {
        assert!((a - b).abs() < 1e-4, "Q4K dequant mismatch: {a} vs {b}");
    }
}

#[test]
fn test_native_matmul_matches_dequantized() {
    // Build one Q4K block row from raw bytes, matmul against a known
    // activation, and compare with the f32 dequantized dot product.
    let mut bytes = vec![0u8; Q4K_BLOCK_SIZE];
    // d = 1.0 (f16 0x3C00), dmin = 0.5 (f16 0x3800)
    bytes[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    bytes[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
    for i in 0..12 {
        bytes[4 + i] = ((i * 5 + 1) & 0x3F) as u8;
    }
    for i in 0..128 {
        bytes[16 + i] = ((i * 7 + 3) & 0xFF) as u8;
    }
    let w = NativeWeight { bytes: Bytes::from_vec(bytes), rows: 1, cols: QK_K };

    let act: Vec<f32> = (0..QK_K).map(|i| (i as f32 - 128.0) / 64.0).collect();
    let mut act_q8k = Vec::new();
    kernels::quantize_rows_q8_k(&act, 1, QK_K, &mut act_q8k);

    let mut out = vec![0.0f32; 1];
    native_matmul::<k_quants::BlockQ4K>(
        w.bytes.as_slice(), 1, QK_K, &mut out, cast_q8k(&act_q8k), 1,
    )
    .unwrap();

    let blocks = unsafe {
        std::slice::from_raw_parts(w.bytes.as_slice().as_ptr() as *const BlockQ4K, 1)
    };
    let mut deq = vec![0.0f32; QK_K];
    kernels::dequantize_q4k_row(blocks, &mut deq);
    let reference: f32 = deq.iter().zip(&act).map(|(w, a)| w * a).sum();

    let tol = reference.abs().max(1.0) * 2e-2; // Q8K-quantized activations
    assert!(
        (out[0] - reference).abs() < tol,
        "native {} vs dequant {}",
        out[0],
        reference
    );
}

const FX: (usize, usize, u32, u32, usize, usize, usize) =
    (512, 256, 4, 2, 64, 512, 2); // vocab, e, h, hkv, hd, ff, layers

fn write_gguf(path: &std::path::Path, metadata: Vec<(&str, candle_core::quantized::gguf_file::Value)>, tensors: &[(String, QTensor)]) {
    use candle_core::quantized::gguf_file;
    let mut file = std::fs::File::create(path).unwrap();
    let meta_refs: Vec<(&str, &gguf_file::Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> =
        tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    gguf_file::write(&mut file, &meta_refs, &tensor_refs).unwrap();
}

/// Write a matched pair of tiny GGUFs built from the same weights:
/// a dense diffuse-format model, and a llama.cpp-style llada-moe model
/// whose 2 experts are both identical copies of the dense FFN. With
/// top-1 routing and weight normalization, the MoE must reproduce the
/// dense model exactly.
fn write_fixture_pair(dense_path: &std::path::Path, moe_path: &std::path::Path) {
    use candle_core::quantized::gguf_file::Value;
    use candle_core::Tensor;

    let (vocab, e, h, hkv, hd, ff, layers) = FX;
    let mut rng = StdRng::seed_from_u64(7);
    let mut rand_data = |n: usize| -> Vec<f32> {
        (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect()
    };

    let q4k = |data: &[f32], shape: &[usize]| {
        let t = Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap();
        QTensor::quantize(&t, GgmlDType::Q4K).unwrap()
    };
    let f32t = |data: &[f32], shape: &[usize]| {
        let t = Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap();
        QTensor::quantize(&t, GgmlDType::F32).unwrap()
    };
    let ones = |n: usize| f32t(&vec![1.0; n], &[n]);

    let embd = rand_data(vocab * e);
    let mut dense: Vec<(String, QTensor)> = vec![
        ("token_embd.weight".into(), f32t(&embd, &[vocab, e])),
        ("output_norm.weight".into(), ones(e)),
    ];
    let mut moe: Vec<(String, QTensor)> = vec![
        ("token_embd.weight".into(), f32t(&embd, &[vocab, e])),
        ("output_norm.weight".into(), ones(e)),
    ];

    for i in 0..layers {
        let p = format!("blk.{i}");
        let (gate, up, down) = (rand_data(ff * e), rand_data(ff * e), rand_data(e * ff));

        for out in [&mut dense, &mut moe] {
            out.push((format!("{p}.attn_norm.weight"), ones(e)));
            out.push((format!("{p}.ffn_norm.weight"), ones(e)));
        }
        for (name, rows) in [("attn_q", h as usize * hd), ("attn_k", hkv as usize * hd), ("attn_v", hkv as usize * hd), ("attn_output", e)] {
            let data = rand_data(rows * e);
            dense.push((format!("{p}.{name}.weight"), q4k(&data, &[rows, e])));
            moe.push((format!("{p}.{name}.weight"), q4k(&data, &[rows, e])));
        }

        dense.push((format!("{p}.ffn_gate.weight"), q4k(&gate, &[ff, e])));
        dense.push((format!("{p}.ffn_up.weight"), q4k(&up, &[ff, e])));
        dense.push((format!("{p}.ffn_down.weight"), q4k(&down, &[e, ff])));

        // MoE: 2 identical experts + a random router.
        let dup = |d: &[f32]| [d, d].concat();
        moe.push((format!("{p}.ffn_gate_inp.weight"), f32t(&rand_data(2 * e), &[2, e])));
        moe.push((format!("{p}.ffn_gate_exps.weight"), q4k(&dup(&gate), &[2, ff, e])));
        moe.push((format!("{p}.ffn_up_exps.weight"), q4k(&dup(&up), &[2, ff, e])));
        moe.push((format!("{p}.ffn_down_exps.weight"), q4k(&dup(&down), &[2, e, ff])));
    }

    write_gguf(
        dense_path,
        vec![
            ("diffuse.vocab_size", Value::U32(vocab as u32)),
            ("diffuse.embedding_length", Value::U32(e as u32)),
            ("diffuse.attention.head_count", Value::U32(h)),
            ("diffuse.attention.head_count_kv", Value::U32(hkv)),
            ("diffuse.block_count", Value::U32(layers as u32)),
            ("diffuse.feed_forward_length", Value::U32(ff as u32)),
            ("diffuse.mask_token_id", Value::U32(500)),
            ("diffuse.model_type", Value::String("llada".into())),
        ],
        &dense,
    );
    // llama.cpp-style metadata: arch-prefixed keys, tokenizer.ggml mask id.
    write_gguf(
        moe_path,
        vec![
            ("general.architecture", Value::String("llada-moe".into())),
            ("llada-moe.embedding_length", Value::U32(e as u32)),
            ("llada-moe.attention.head_count", Value::U32(h)),
            ("llada-moe.attention.head_count_kv", Value::U32(hkv)),
            ("llada-moe.block_count", Value::U32(layers as u32)),
            ("llada-moe.expert_count", Value::U32(2)),
            ("llada-moe.expert_used_count", Value::U32(1)),
            // Explicit: top-1 weight becomes exactly 1.0, making the MoE
            // bit-equivalent to the dense reference.
            ("llada-moe.expert_weights_norm", Value::Bool(true)),
            ("tokenizer.ggml.mask_token_id", Value::U32(500)),
            ("diffusion.shift_logits", Value::Bool(false)),
        ],
        &moe,
    );
}

fn write_tiny_gguf(path: &std::path::Path) {
    let moe_path = path.with_extension("moe.tmp.gguf");
    write_fixture_pair(path, &moe_path);
    let _ = std::fs::remove_file(moe_path);
}

#[test]
fn test_moe_single_expert_matches_dense() {
    let base = std::env::temp_dir();
    let dense_path = base.join(format!("diffuse-rs-dense-{}.gguf", std::process::id()));
    let moe_path = base.join(format!("diffuse-rs-moe-{}.gguf", std::process::id()));
    write_fixture_pair(&dense_path, &moe_path);

    let tokens: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let mut dense = Model::from_gguf(dense_path.to_str().unwrap()).unwrap();
    let mut moe = Model::from_gguf(moe_path.to_str().unwrap()).unwrap();
    assert_eq!(moe.model_type, "llada-moe");
    assert_eq!(moe.mask_token_id, 500);

    let a = dense.forward(&tokens).unwrap();
    let b = moe.forward(&tokens).unwrap();
    let max_abs = a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    for (x, y) in a.iter().zip(&b) {
        assert!(
            (x - y).abs() <= max_abs * 1e-5 + 1e-5,
            "dense {x} vs moe {y} (max_abs {max_abs})"
        );
    }

    let _ = std::fs::remove_file(&dense_path);
    let _ = std::fs::remove_file(&moe_path);
}

#[test]
fn test_tiny_model_native_matches_candle_and_is_deterministic() {
    let path = std::env::temp_dir().join(format!("diffuse-rs-fixture-{}.gguf", std::process::id()));
    write_tiny_gguf(&path);
    let path_str = path.to_str().unwrap();

    let tokens: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6];

    let mut native = Model::from_gguf(path_str).unwrap();
    let logits_a = native.forward(&tokens).unwrap();
    let logits_b = native.forward(&tokens).unwrap();
    assert_eq!(logits_a, logits_b, "forward must be deterministic");

    std::env::set_var("DIFFUSE_MATMUL", "candle");
    let mut candle = Model::from_gguf(path_str).unwrap();
    std::env::remove_var("DIFFUSE_MATMUL");
    let logits_c = candle.forward(&tokens).unwrap();

    let max_abs = logits_a.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    for (a, c) in logits_a.iter().zip(&logits_c) {
        assert!(
            (a - c).abs() <= max_abs * 1e-4 + 1e-4,
            "native {a} vs candle {c} (max_abs {max_abs})"
        );
    }

    // End-to-end generation: deterministic, fills all positions.
    let params = SamplerParams { n_steps: 4, ..Default::default() };
    let out1 = generate(&mut native, &tokens, 8, &params).unwrap();
    let out2 = generate(&mut native, &tokens, 8, &params).unwrap();
    assert_eq!(out1.tokens, out2.tokens);
    assert_eq!(out1.tokens.len(), 8);
    assert_eq!(out1.finish_reason, FinishReason::Stop);
    assert!(out1.tokens.iter().all(|&t| t != 500), "no masks in output");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_forward_batch_matches_separate() {
    // A batched forward over two sequences must reproduce each sequence's
    // logits exactly (block-diagonal attention => independent sequences).
    let path = std::env::temp_dir().join(format!("diffuse-rs-batch-{}.gguf", std::process::id()));
    write_tiny_gguf(&path);
    let mut model = Model::from_gguf(path.to_str().unwrap()).unwrap();
    let nv = model.n_vocab;

    let s0: Vec<i32> = vec![3, 1, 4, 1, 5];
    let s1: Vec<i32> = vec![9, 2, 6];
    let single0 = model.forward(&s0).unwrap();
    let single1 = model.forward(&s1).unwrap();

    let batch = Batch {
        tokens: s0.iter().chain(&s1).copied().collect(),
        pos: (0..s0.len()).chain(0..s1.len()).collect(),
        seq: std::iter::repeat_n(0, s0.len())
            .chain(std::iter::repeat_n(1, s1.len()))
            .collect(),
        canvas_start: vec![0, 0],
        n_canvas: 0,
        sc_signal: None,
    };
    let mut batched = Vec::new();
    model.forward_batch(&batch, &mut batched).unwrap();

    for i in 0..s0.len() * nv {
        assert!((batched[i] - single0[i]).abs() < 1e-4, "seq0 logit {i}");
    }
    for i in 0..s1.len() * nv {
        assert!((batched[s0.len() * nv + i] - single1[i]).abs() < 1e-4, "seq1 logit {i}");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_generate_batch_matches_separate() {
    // Batched generation must reproduce per-sequence generation exactly when
    // both run cacheless full recompute (block-diagonal => independent).
    let path = std::env::temp_dir().join(format!("diffuse-rs-genb-{}.gguf", std::process::id()));
    write_tiny_gguf(&path);
    let mut model = Model::from_gguf(path.to_str().unwrap()).unwrap();

    let p0: Vec<i32> = vec![3, 1, 4, 1, 5];
    let p1: Vec<i32> = vec![9, 2, 6, 8];
    let params = SamplerParams { n_steps: 4, use_cache: false, ..Default::default() };

    let s0 = generate(&mut model, &p0, 6, &params).unwrap();
    let s1 = generate(&mut model, &p1, 6, &params).unwrap();
    let batched = generate_batch(&mut model, &[p0, p1], 6, &params).unwrap();

    assert_eq!(batched.len(), 2);
    assert_eq!(batched[0].tokens, s0.tokens, "seq0 batched != separate");
    assert_eq!(batched[1].tokens, s1.tokens, "seq1 batched != separate");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_rotate_rows_position_zero_is_identity() {
    let rope = build_rope_cache(4, 4, 10000.0);
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    rotate_rows(&mut x, &rope, &[0], 1, 4, 4);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0]);
}
