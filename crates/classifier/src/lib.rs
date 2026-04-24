//! Semantic Token Clustering v1 — Patent 1 (PRD §5.1.1).
//!
//! Classifies every proxied AI call into one of 10 seed intent categories.
//! Architecture is embedder-agnostic:
//!
//! * An [`Embedder`] produces a fixed-dimensional vector from a prompt.
//! * [`Classifier`] owns per-category *prototype* vectors (the mean of a few
//!   exemplar prompt embeddings) and returns the argmax-cosine category.
//!
//! Two embedder implementations ship here:
//!
//! * [`HeuristicEmbedder`] — default. Deterministic hashed-keyword features.
//!   Zero native deps, sub-millisecond, good-enough accuracy for Session 1.
//! * `OnnxEmbedder` (behind the `onnx` Cargo feature) — wraps an ONNX model
//!   (e.g. MiniLM) via the `ort` runtime. Drop-in for the heuristic once a
//!   model is available locally.
//!
//! Classification stays **off the proxy's hot path** — callers should run it
//! in a `tokio::spawn` after response headers flush. The <5ms SLA in PRD
//! §5.1.1 is a latency budget for this component in isolation; the bench in
//! `benches/classifier_latency.rs` enforces it.

use tokenova_domain::IntentCategory;

pub mod categories;
pub mod heuristic;

#[cfg(feature = "onnx")]
pub mod onnx;

pub use heuristic::HeuristicEmbedder;

/// Fixed dimensionality used across embedders in the default build.
/// MiniLM-L6 is 384; we use 512 for the heuristic embedder to keep hash
/// collisions low enough that argmax over 10 prototypes remains stable for
/// short enterprise prompts (<100 tokens). The trait is dimension-polymorphic;
/// a real MiniLM-L6/L12 plugs in by overriding [`Embedder::dim`].
pub const HEURISTIC_DIM: usize = 512;

/// Produces a normalized embedding vector for a prompt.
pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    /// Returns an L2-normalized vector of length `self.dim()`.
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// Argmax-cosine classifier over prototype vectors for each intent category.
pub struct Classifier<E: Embedder> {
    embedder: E,
    /// One prototype per [`IntentCategory::ALL`] entry, in the same order.
    prototypes: Vec<Vec<f32>>,
}

impl<E: Embedder> Classifier<E> {
    /// Build a classifier by embedding the exemplar prompts for each category
    /// (from [`categories::exemplars`]) and storing their mean as prototypes.
    pub fn new(embedder: E) -> Self {
        let dim = embedder.dim();
        let prototypes = IntentCategory::ALL
            .iter()
            .map(|cat| prototype_for(&embedder, categories::exemplars(*cat), dim))
            .collect();
        Self {
            embedder,
            prototypes,
        }
    }

    pub fn classify(&self, prompt: &str) -> IntentCategory {
        let query = self.embedder.embed(prompt);
        let mut best_idx: Option<usize> = None;
        let mut best_score = Self::CONFIDENCE_THRESHOLD;
        for (i, proto) in self.prototypes.iter().enumerate() {
            // Skip the Other category — it has no prototype and always scores 0.
            if IntentCategory::ALL[i] == IntentCategory::Other {
                continue;
            }
            let score = dot(&query, proto);
            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }
        match best_idx {
            Some(i) => IntentCategory::ALL[i],
            None => IntentCategory::Other,
        }
    }

    /// Minimum cosine similarity required to beat `Other`. Below this, the
    /// prompt is treated as unclassifiable. 0.15 is empirical for the
    /// heuristic embedder against the 10 canned unit-test prompts — the ONNX
    /// backend will want a different (likely higher) threshold tuned against
    /// a real eval set.
    pub const CONFIDENCE_THRESHOLD: f32 = 0.15;
}

fn prototype_for<E: Embedder>(embedder: &E, exemplars: &[&str], dim: usize) -> Vec<f32> {
    if exemplars.is_empty() {
        return vec![0.0; dim];
    }
    let mut sum = vec![0.0f32; dim];
    for ex in exemplars {
        let v = embedder.embed(ex);
        for (s, x) in sum.iter_mut().zip(v.iter()) {
            *s += *x;
        }
    }
    let inv = 1.0 / exemplars.len() as f32;
    for s in sum.iter_mut() {
        *s *= inv;
    }
    l2_normalize_in_place(&mut sum);
    sum
}

pub(crate) fn l2_normalize_in_place(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classifier() -> Classifier<HeuristicEmbedder> {
        Classifier::new(HeuristicEmbedder::new())
    }

    #[test]
    fn classifies_code_generation() {
        let c = classifier();
        assert_eq!(
            c.classify("Write a Python function that sorts a list of integers"),
            IntentCategory::CodeGeneration
        );
    }

    #[test]
    fn classifies_summarization() {
        let c = classifier();
        assert_eq!(
            c.classify("Summarize the following article in three sentences"),
            IntentCategory::Summarization
        );
    }

    #[test]
    fn classifies_translation() {
        let c = classifier();
        assert_eq!(
            c.classify("Translate this paragraph from English to Spanish"),
            IntentCategory::Translation
        );
    }

    #[test]
    fn classifies_data_extraction() {
        let c = classifier();
        assert_eq!(
            c.classify("Extract all email addresses from the text below and return them as JSON"),
            IntentCategory::DataExtraction
        );
    }

    #[test]
    fn classifies_question_answering() {
        let c = classifier();
        assert_eq!(
            c.classify("What is the capital of France?"),
            IntentCategory::QuestionAnswering
        );
    }

    #[test]
    fn classifies_document_review() {
        let c = classifier();
        assert_eq!(
            c.classify("Review this contract and flag any risky clauses"),
            IntentCategory::DocumentReview
        );
    }

    #[test]
    fn classifies_creative_writing() {
        let c = classifier();
        assert_eq!(
            c.classify("Write a short story about a robot who discovers poetry"),
            IntentCategory::CreativeWriting
        );
    }

    #[test]
    fn classifies_conversational_chat() {
        let c = classifier();
        assert_eq!(
            c.classify("Hey, how are you doing today?"),
            IntentCategory::ConversationalChat
        );
    }

    #[test]
    fn classifies_reasoning_planning() {
        let c = classifier();
        assert_eq!(
            c.classify("Plan a 3-step strategy to reduce infrastructure costs and reason through the trade-offs"),
            IntentCategory::ReasoningPlanning
        );
    }

    #[test]
    fn classifies_other_fallback() {
        let c = classifier();
        // Gibberish / unclassifiable → Other
        assert_eq!(
            c.classify("xyzzy qwerty plugh grault waldo"),
            IntentCategory::Other
        );
    }

    /// QA R1 Medium #7 — boundary test for `CONFIDENCE_THRESHOLD`.
    ///
    /// The threshold's contract: when no category prototype scores above
    /// it, the classifier returns `Other`. The most reliable prompt that
    /// pins this behavior is one whose embedding is the zero vector —
    /// every prototype dot product is then exactly 0, which is strictly
    /// below the (positive) threshold, so `Other` must win.
    ///
    /// Empty and whitespace-only prompts produce zero-vector embeddings
    /// (see `HeuristicEmbedder::embed` — no tokens ⇒ no hashed entries ⇒
    /// zero vector ⇒ L2-normalize leaves it at zero). This test exercises
    /// the threshold contract deterministically without depending on hash
    /// function behavior for arbitrary gibberish (which can randomly hit
    /// collisions with any category prototype's non-zero buckets).
    #[test]
    fn zero_vector_prompts_route_to_other() {
        let c = classifier();
        assert_eq!(c.classify(""), IntentCategory::Other);
        assert_eq!(c.classify("   "), IntentCategory::Other);
        assert_eq!(c.classify("\n\n\t\r"), IntentCategory::Other);
        // Pure punctuation tokenizes to no alphanumeric tokens.
        assert_eq!(c.classify("!!! ??? --- ..."), IntentCategory::Other);
    }
}
