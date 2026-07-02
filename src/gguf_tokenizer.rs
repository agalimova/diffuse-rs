//! Tokenizer loaded from GGUF metadata (`tokenizer.ggml.*`), following
//! llama.cpp's vocab handling. Two backends are supported:
//! - Byte-level BPE (`gpt2`): LLaDA (Llama-3), Dream (Qwen2), LLaDA-MoE.
//! - SentencePiece unigram (`gemma4` and other scored vocabs): DiffusionGemma.
//!
//! No separate tokenizer.json is needed for llama.cpp-converted GGUFs.

use anyhow::{bail, ensure, Context, Result};
use std::cmp::Reverse;
use std::collections::HashMap;

// llama.cpp token types (llama_token_type)
const TYPE_NORMAL: i64 = 1;
const TYPE_UNKNOWN: i64 = 2;
const TYPE_CONTROL: i64 = 3;
const TYPE_USER_DEFINED: i64 = 4;
const TYPE_BYTE: i64 = 6;

/// SentencePiece word-boundary marker (U+2581, "lower one eighth block").
const SP_SPACE: char = '\u{2581}';

pub enum GgufTokenizer {
    Bpe(BpeTokenizer),
    Unigram(UnigramTokenizer),
}

impl GgufTokenizer {
    /// Load from a GGUF's metadata. Returns None when the file carries no
    /// embedded vocab (e.g. diffuse-cpp conversions).
    pub fn from_gguf(path: &str) -> Result<Option<Self>> {
        use candle_core::quantized::gguf_file;
        let mut file = std::fs::File::open(path)?;
        let gguf = gguf_file::Content::read(&mut file)?;
        let md = &gguf.metadata;

        let Some(tokens_v) = md.get("tokenizer.ggml.tokens") else {
            return Ok(None);
        };
        let model = md
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_default();

        let strings = |v: &gguf_file::Value| -> Result<Vec<String>> {
            Ok(v.to_vec()?
                .iter()
                .map(|s| s.to_string().map(|s| s.to_string()))
                .collect::<std::result::Result<_, _>>()?)
        };
        let as_int = |v: &gguf_file::Value| -> Option<i64> {
            v.to_i64()
                .ok()
                .or_else(|| v.to_i32().ok().map(i64::from))
                .or_else(|| v.to_u32().ok().map(i64::from))
                .or_else(|| v.to_i16().ok().map(i64::from))
                .or_else(|| v.to_u16().ok().map(i64::from))
                .or_else(|| v.to_i8().ok().map(i64::from))
                .or_else(|| v.to_u8().ok().map(i64::from))
        };
        let tokens = strings(tokens_v)?;
        let types: Vec<i64> = md
            .get("tokenizer.ggml.token_type")
            .map(|v| -> Result<Vec<i64>> {
                let vals = v.to_vec()?;
                let ints: Vec<i64> = vals.iter().filter_map(as_int).collect();
                ensure!(ints.len() == vals.len(), "token_type array has non-integer entries");
                Ok(ints)
            })
            .transpose()?
            .unwrap_or_default();
        let pre = md
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "llama-bpe".into());

        if model == "gpt2" {
            let merges = strings(md.get("tokenizer.ggml.merges").context("missing merges")?)?;
            return Ok(Some(Self::Bpe(BpeTokenizer::build(&tokens, &types, merges, &pre))));
        }

        // Scored vocab implies SentencePiece unigram.
        if let Some(scores_v) = md.get("tokenizer.ggml.scores") {
            let scores: Vec<f32> = scores_v
                .to_vec()?
                .iter()
                .map(|v| v.to_f32().or_else(|_| v.to_f64().map(|x| x as f32)))
                .collect::<std::result::Result<_, _>>()?;
            return Ok(Some(Self::Unigram(UnigramTokenizer::build(tokens, &types, scores)?)));
        }

        bail!("unsupported tokenizer.ggml.model {model:?} (no merges or scores)");
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match self {
            Self::Bpe(t) => t.encode(text),
            Self::Unigram(t) => Ok(t.encode(text)),
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        match self {
            Self::Bpe(t) => t.decode(ids, skip_special),
            Self::Unigram(t) => t.decode(ids, skip_special),
        }
    }

    /// Look up a token's id (used by the server's chat-template handling).
    #[allow(dead_code)] // exercised only under the `server` feature
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Bpe(t) => t.vocab.get(token).copied(),
            Self::Unigram(t) => t.str_to_id.get(token).copied(),
        }
    }
}

/// Earliest, then longest, special-token match in `text`. The min_by_key
/// tiebreak picks the longest at a given position regardless of list order.
fn next_special(specials: &[(String, u32)], text: &str) -> Option<(usize, usize, u32)> {
    specials
        .iter()
        .filter_map(|(s, id)| text.find(s.as_str()).map(|at| (at, s.len(), *id)))
        .min_by_key(|&(at, len, _)| (at, Reverse(len)))
}

// =============================================================================
// Byte-level BPE (gpt2)
// =============================================================================

pub struct BpeTokenizer {
    /// Raw bytes each token decodes to.
    bytes_of: Vec<Vec<u8>>,
    /// Control/user-defined tokens, matched whole in input text.
    special_tokens: Vec<(String, u32)>,
    /// Dropped from output when decoding with skip_special.
    skip_on_decode: Vec<bool>,
    /// Token string (byte-level alphabet) -> id.
    vocab: HashMap<String, u32>,
    /// BPE merge "left right" -> rank (lower merges first).
    merge_rank: HashMap<String, usize>,
    /// GPT-2 byte -> printable stand-in, cached so encode never rebuilds it.
    byte_to_char: Vec<char>,
    pre: fancy_regex::Regex,
}

/// GPT-2 byte-level maps: every byte gets a printable unicode stand-in.
fn byte_maps() -> (Vec<char>, HashMap<char, u8>) {
    let mut byte_to_char = vec!['\0'; 256];
    let mut char_to_byte = HashMap::new();
    let printable = (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF);
    let mut taken = [false; 256];
    for b in printable {
        byte_to_char[b as usize] = char::from_u32(b as u32).unwrap();
        taken[b as usize] = true;
    }
    let mut next = 0u32;
    for b in 0..256 {
        if !taken[b] {
            byte_to_char[b] = char::from_u32(256 + next).unwrap();
            next += 1;
        }
    }
    for (b, &c) in byte_to_char.iter().enumerate() {
        char_to_byte.insert(c, b as u8);
    }
    (byte_to_char, char_to_byte)
}

/// Pretokenizer regex by `tokenizer.ggml.pre`, mirroring llama.cpp.
fn pre_regex(pre: &str) -> fancy_regex::Regex {
    let llama3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let qwen2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let bailing = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
    let pattern = match pre {
        "qwen2" | "dream" => qwen2,
        "llama-bpe" | "llama3" | "llada" => llama3,
        // Ling/Bailing family (LLaDA-MoE, LLaDA2); mirrors llama.cpp's
        // LLAMA_VOCAB_PRE_TYPE_BAILINGMOE.
        "bailingmoe" | "bailingmoe2" | "llada-moe" => bailing,
        other => {
            eprintln!("[diffuse-rs] unknown tokenizer.ggml.pre {other:?}, using llama-bpe rules");
            llama3
        }
    };
    fancy_regex::Regex::new(pattern).expect("pretokenizer regex")
}

impl BpeTokenizer {
    fn build(tokens: &[String], types: &[i64], merges: Vec<String>, pre: &str) -> Self {
        let (byte_to_char, char_to_byte) = byte_maps();
        let ttype = |i: usize| types.get(i).copied().unwrap_or(TYPE_NORMAL);

        let mut bytes_of = Vec::with_capacity(tokens.len());
        let mut special_tokens = Vec::new();
        let mut skip_on_decode = vec![false; tokens.len()];
        let mut vocab = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            let t = ttype(i);
            let raw: Vec<u8> = if t == TYPE_CONTROL || t == TYPE_USER_DEFINED {
                special_tokens.push((tok.clone(), i as u32));
                skip_on_decode[i] = t == TYPE_CONTROL;
                tok.as_bytes().to_vec()
            } else if t == TYPE_BYTE {
                let hex = tok.trim_start_matches("<0x").trim_end_matches('>');
                let b = u8::from_str_radix(hex, 16).unwrap_or_else(|_| {
                    eprintln!("[diffuse-rs] malformed byte token {tok:?} (id {i}); decoding as 0x00");
                    0
                });
                vec![b]
            } else {
                tok.chars().map(|c| char_to_byte.get(&c).copied().unwrap_or(b'?')).collect()
            };
            bytes_of.push(raw);
            vocab.insert(tok.clone(), i as u32);
        }
        special_tokens.sort_by_key(|(s, _)| Reverse(s.len()));
        let merge_rank = merges.into_iter().enumerate().map(|(r, m)| (m, r)).collect();

        eprintln!(
            "[diffuse-rs] embedded tokenizer: {} tokens, {} specials, pre={pre} (BPE)",
            tokens.len(),
            special_tokens.len()
        );
        Self {
            bytes_of,
            special_tokens,
            skip_on_decode,
            vocab,
            merge_rank,
            byte_to_char,
            pre: pre_regex(pre),
        }
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            match next_special(&self.special_tokens, rest) {
                Some((at, len, id)) => {
                    self.encode_plain(&rest[..at], &mut out)?;
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out)?;
                    break;
                }
            }
        }
        Ok(out)
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for piece in self.pre.find_iter(text) {
            let piece = piece?;
            let mapped: Vec<String> = piece
                .as_str()
                .bytes()
                .map(|b| self.byte_to_char[b as usize].to_string())
                .collect();
            self.bpe(mapped, out)?;
        }
        Ok(())
    }

    fn bpe(&self, mut parts: Vec<String>, out: &mut Vec<u32>) -> Result<()> {
        // One reused key buffer: the merge loop scans O(len^2) pairs, and a
        // fresh `format!` per pair dominated encode time.
        let mut key = String::new();
        loop {
            let best = (0..parts.len().saturating_sub(1))
                .filter_map(|i| {
                    key.clear();
                    key.push_str(&parts[i]);
                    key.push(' ');
                    key.push_str(&parts[i + 1]);
                    self.merge_rank.get(key.as_str()).map(|&r| (r, i))
                })
                .min();
            match best {
                Some((_, i)) => {
                    let right = parts.remove(i + 1);
                    parts[i].push_str(&right);
                }
                None => break,
            }
        }
        for p in &parts {
            out.push(
                self.vocab
                    .get(p)
                    .copied()
                    .with_context(|| format!("token piece {p:?} not in vocab"))?,
            );
        }
        Ok(())
    }

    fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let id = id as usize;
            if id >= self.bytes_of.len() || (skip_special && self.skip_on_decode[id]) {
                continue;
            }
            bytes.extend_from_slice(&self.bytes_of[id]);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// =============================================================================
// SentencePiece unigram (gemma4)
// =============================================================================

pub struct UnigramTokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    /// Normal-piece bytes -> id, for the Viterbi lattice.
    vocab: HashMap<Vec<u8>, u32>,
    /// Raw byte value -> id of its `<0xNN>` fallback token.
    byte_fallback: Vec<Option<u32>>,
    is_byte: Vec<bool>,
    byte_val: Vec<u8>,
    max_len: usize,
    unk_id: u32,
    special_tokens: Vec<(String, u32)>,
    skip_on_decode: Vec<bool>,
    str_to_id: HashMap<String, u32>,
}

impl UnigramTokenizer {
    fn build(tokens: Vec<String>, types: &[i64], scores: Vec<f32>) -> Result<Self> {
        let n = tokens.len();
        ensure!(scores.len() == n, "scores length {} != tokens {n}", scores.len());
        let ttype = |i: usize| types.get(i).copied().unwrap_or(TYPE_NORMAL);

        let mut vocab = HashMap::new();
        let mut byte_fallback = vec![None; 256];
        let mut is_byte = vec![false; n];
        let mut byte_val = vec![0u8; n];
        let mut special_tokens = Vec::new();
        let mut skip_on_decode = vec![false; n];
        let mut str_to_id = HashMap::with_capacity(n);
        let mut max_len = 1;
        let mut unk_id = 0u32;

        for (i, tok) in tokens.iter().enumerate() {
            str_to_id.insert(tok.clone(), i as u32);
            match ttype(i) {
                TYPE_BYTE => {
                    let hex = tok.trim_start_matches("<0x").trim_end_matches('>');
                    if let Ok(b) = u8::from_str_radix(hex, 16) {
                        is_byte[i] = true;
                        byte_val[i] = b;
                        byte_fallback[b as usize] = Some(i as u32);
                    }
                }
                TYPE_CONTROL | TYPE_USER_DEFINED => {
                    special_tokens.push((tok.clone(), i as u32));
                    skip_on_decode[i] = ttype(i) == TYPE_CONTROL;
                }
                TYPE_UNKNOWN => unk_id = i as u32,
                _ => {
                    let b = tok.as_bytes().to_vec();
                    max_len = max_len.max(b.len());
                    vocab.insert(b, i as u32);
                }
            }
        }
        special_tokens.sort_by_key(|(s, _)| Reverse(s.len()));

        eprintln!(
            "[diffuse-rs] embedded tokenizer: {n} tokens, {} specials (unigram)",
            special_tokens.len()
        );
        Ok(Self {
            pieces: tokens,
            scores,
            vocab,
            byte_fallback,
            is_byte,
            byte_val,
            max_len,
            unk_id,
            special_tokens,
            skip_on_decode,
            str_to_id,
        })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            match next_special(&self.special_tokens, rest) {
                Some((at, len, id)) => {
                    self.encode_plain(&rest[..at], &mut out);
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    /// Apply the gemma normalizer (prepend the word-boundary marker, replace
    /// each space with the marker), then run the SentencePiece Viterbi lattice
    /// with byte fallback.
    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut norm = String::with_capacity(text.len() + 3);
        norm.push(SP_SPACE);
        for ch in text.chars() {
            norm.push(if ch == ' ' { SP_SPACE } else { ch });
        }
        let bytes = norm.as_bytes();
        let n = bytes.len();

        let neg = f32::NEG_INFINITY;
        let mut best = vec![neg; n + 1];
        let mut back: Vec<(usize, u32)> = vec![(0, 0); n + 1];
        best[0] = 0.0;
        for i in 0..n {
            if !best[i].is_finite() {
                continue; // position not reachable yet
            }
            for l in 1..=self.max_len.min(n - i) {
                if let Some(&id) = self.vocab.get(&bytes[i..i + l]) {
                    let s = best[i] + self.scores[id as usize];
                    if s > best[i + l] {
                        best[i + l] = s;
                        back[i + l] = (i, id);
                    }
                }
            }
            // A single-byte fallback keeps the lattice connected when no piece
            // matches. The fallback score is low, so any real piece wins.
            let fid = self.byte_fallback[bytes[i] as usize].unwrap_or(self.unk_id);
            let s = best[i] + self.scores[fid as usize];
            if s > best[i + 1] {
                best[i + 1] = s;
                back[i + 1] = (i, fid);
            }
        }

        let start = out.len();
        let mut pos = n;
        while pos > 0 {
            let (prev, id) = back[pos];
            out.push(id);
            pos = prev;
        }
        out[start..].reverse();
    }

    fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut out = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();
        for &id in ids {
            let id = id as usize;
            if id >= self.pieces.len() || (skip_special && self.skip_on_decode[id]) {
                continue;
            }
            if self.is_byte[id] {
                byte_buf.push(self.byte_val[id]);
                continue;
            }
            if !byte_buf.is_empty() {
                out.push_str(&String::from_utf8_lossy(&byte_buf));
                byte_buf.clear();
            }
            out.push_str(&self.pieces[id].replace(SP_SPACE, " "));
        }
        if !byte_buf.is_empty() {
            out.push_str(&String::from_utf8_lossy(&byte_buf));
        }
        // Strip the single leading space introduced by the normalizer.
        if out.starts_with(' ') {
            out.remove(0);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_bpe() -> BpeTokenizer {
        let toks = ["a", "b", "ab", "\u{0120}", "\u{0120}c", "c", "<|end|>"];
        let mut vocab = HashMap::new();
        let mut bytes_of = Vec::new();
        let (_, c2b) = byte_maps();
        for (i, t) in toks.iter().enumerate() {
            vocab.insert(t.to_string(), i as u32);
            if *t == "<|end|>" {
                bytes_of.push(t.as_bytes().to_vec());
            } else {
                bytes_of.push(t.chars().map(|c| c2b[&c]).collect());
            }
        }
        BpeTokenizer {
            bytes_of,
            special_tokens: vec![("<|end|>".into(), 6)],
            skip_on_decode: vec![false, false, false, false, false, false, true],
            vocab,
            merge_rank: [("a b".to_string(), 0), ("\u{0120} c".to_string(), 1)].into(),
            byte_to_char: byte_maps().0,
            pre: pre_regex("llama-bpe"),
        }
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tok = tiny_bpe();
        let ids = tok.encode("ab c<|end|>").unwrap();
        assert_eq!(ids, vec![2, 4, 6]);
        assert_eq!(tok.decode(&ids, false), "ab c<|end|>");
        assert_eq!(tok.decode(&ids, true), "ab c");
    }

    #[test]
    fn test_pre_regex_bailing() {
        let re = pre_regex("bailingmoe2");
        let pieces: Vec<&str> =
            re.find_iter("Hello's 123 world!").map(|m| m.unwrap().as_str()).collect();
        // Digits split individually; contraction and punctuation as their
        // own pieces, matching llama.cpp's bailingmoe rules.
        assert_eq!(pieces, vec!["Hello", "'s", " ", "1", "2", "3", " world", "!"]);
    }

    #[test]
    fn test_byte_maps_roundtrip() {
        let (b2c, c2b) = byte_maps();
        for b in 0..=255u8 {
            assert_eq!(c2b[&b2c[b as usize]], b);
        }
    }

    /// A tiny unigram: pieces "he", "llo", "h", "e", "l", "o" plus a space
    /// marker and one byte token. Viterbi should prefer the higher-score
    /// multi-char pieces and round-trip through decode.
    fn tiny_unigram() -> UnigramTokenizer {
        let space = SP_SPACE.to_string();
        let toks = vec![
            space.clone(), // 0
            "he".into(),   // 1
            "llo".into(),  // 2
            "h".into(),    // 3
            "e".into(),    // 4
            "l".into(),    // 5
            "o".into(),    // 6
            "<0x21>".into(), // 7  '!'
        ];
        let types = vec![1, 1, 1, 1, 1, 1, 1, TYPE_BYTE];
        // Favor "he" and "llo" over the single chars.
        let scores = vec![-1.0, -1.0, -1.0, -5.0, -5.0, -5.0, -5.0, -10.0];
        UnigramTokenizer::build(toks, &types, scores).unwrap()
    }

    #[test]
    fn test_unigram_encode_decode() {
        let tok = tiny_unigram();
        let ids = tok.encode("hello");
        // "▁" + "he" + "llo"
        assert_eq!(ids, vec![0, 1, 2]);
        assert_eq!(tok.decode(&ids, false), "hello");
    }

    #[test]
    fn test_unigram_byte_fallback() {
        let tok = tiny_unigram();
        // '!' is only reachable through the <0x21> byte token.
        let ids = tok.encode("he!");
        assert_eq!(ids, vec![0, 1, 7]);
        assert_eq!(tok.decode(&ids, false), "he!");
    }
}
