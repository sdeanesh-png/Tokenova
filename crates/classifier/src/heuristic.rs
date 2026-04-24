//! Deterministic zero-dependency embedder — the default backend.
//!
//! Hashes unigrams and bigrams into a fixed-dimensional vector using FNV-1a,
//! then L2-normalizes. Not semantically sophisticated, but:
//!
//! * Sub-millisecond per classification on a modern CPU (no model load, no
//!   tokenizer, no ONNX runtime).
//! * Deterministic across runs and platforms — same prompt → same vector.
//! * Swappable via the [`Embedder`] trait for the ONNX backend (see
//!   [`super::onnx`] behind the `onnx` feature).
//!
//! Good enough for Session 1's 10-way classification against short enterprise
//! prompts. Accuracy on long/ambiguous prompts is bounded by hash collisions
//! and lack of semantic generalization — that's what the ONNX backend fixes.

use super::{l2_normalize_in_place, Embedder, HEURISTIC_DIM};

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "this", "that", "of", "in", "for", "to", "from", "on", "at", "by", "as",
    "it", "its", "and", "or", "but",
];

const UNIGRAM_WEIGHT: f32 = 1.0;
const BIGRAM_WEIGHT: f32 = 1.5;

pub struct HeuristicEmbedder {
    dim: usize,
}

impl Default for HeuristicEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl HeuristicEmbedder {
    pub fn new() -> Self {
        Self { dim: HEURISTIC_DIM }
    }

    pub fn with_dim(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for HeuristicEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        let tokens: Vec<String> = tokenize(text);

        // Bigrams use the raw token sequence (including stopwords) so phrase
        // structure like "how are you" is preserved.
        for pair in tokens.windows(2) {
            let bigram = format!("{}_{}", pair[0], pair[1]);
            let idx = (fnv1a(bigram.as_bytes()) as usize) % self.dim;
            v[idx] += BIGRAM_WEIGHT;
        }

        // Unigrams drop stopwords so topical tokens dominate.
        for tok in tokens.iter().filter(|t| !is_stopword(t)) {
            let idx = (fnv1a(tok.as_bytes()) as usize) % self.dim;
            v[idx] += UNIGRAM_WEIGHT;
        }

        l2_normalize_in_place(&mut v);
        v
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() {
            for lc in c.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn is_stopword(tok: &str) -> bool {
    STOPWORDS.contains(&tok)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_strips_punctuation_and_lowercases() {
        let tokens = tokenize("Hey, How are YOU doing today?");
        assert_eq!(tokens, vec!["hey", "how", "are", "you", "doing", "today"]);
    }

    #[test]
    fn embedding_is_unit_length() {
        let e = HeuristicEmbedder::new();
        let v = e.embed("Write a Python function that parses JSON");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }

    #[test]
    fn empty_text_produces_zero_vector() {
        let e = HeuristicEmbedder::new();
        let v = e.embed("");
        assert!(v.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn embeddings_are_deterministic() {
        let e = HeuristicEmbedder::new();
        let a = e.embed("summarize this document");
        let b = e.embed("summarize this document");
        assert_eq!(a, b);
    }
}
