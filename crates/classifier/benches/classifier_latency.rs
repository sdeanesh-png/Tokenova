//! Criterion bench enforcing the <5ms classification SLA from PRD §5.1.2.
//!
//! Measures end-to-end `Classifier::classify` wall-time for representative
//! enterprise prompts. The heuristic embedder should land well under 1ms on
//! any modern CPU; the ONNX variant (when enabled) is budgeted to 5ms p99.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tokenova_classifier::{Classifier, HeuristicEmbedder};

const PROMPTS: &[&str] = &[
    "Write a Python function that sorts a list of integers",
    "Summarize the following article in three sentences",
    "Extract all email addresses from the text below and return them as JSON",
    "What is the capital of France?",
    "Review this contract and flag any risky clauses",
    "Translate this paragraph from English to Spanish",
    "Write a short story about a robot who discovers poetry",
    "Hey, how are you doing today?",
    "Plan a 3-step strategy to reduce infrastructure costs and reason through the trade-offs",
    "xyzzy qwerty plugh grault waldo",
];

fn bench_heuristic(c: &mut Criterion) {
    let clf = Classifier::new(HeuristicEmbedder::new());
    c.bench_function("heuristic_classify_10_prompts", |b| {
        b.iter(|| {
            for p in PROMPTS {
                let _ = black_box(clf.classify(black_box(p)));
            }
        })
    });

    c.bench_function("heuristic_classify_single", |b| {
        b.iter(|| {
            black_box(clf.classify(black_box(
                "Summarize the following article in three sentences",
            )))
        })
    });
}

criterion_group!(benches, bench_heuristic);
criterion_main!(benches);
