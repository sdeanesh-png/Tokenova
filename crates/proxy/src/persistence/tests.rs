//! Unit tests for the persistence dispatch trait. These don't touch a
//! real database — they verify that the trait surface and the
//! `dispatch_for_test` plumbing behave correctly.
//!
//! Real-database integration tests live in `crates/proxy/tests/persistence_postgres.rs`
//! and are gated on `TOKENOVA_TEST_DATABASE_URL` being set, so CI doesn't
//! need a Postgres container.

use std::sync::{Arc, Mutex};

use tokenova_domain::{AttributionTags, IntentCategory, LogRecord, Provider, TokenUsage};
use uuid::Uuid;

use super::{dispatch_for_test, LogRecordSink};

/// A vector-backed sink for tests.
struct VecSink {
    records: Mutex<Vec<LogRecord>>,
}

impl VecSink {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            records: Mutex::new(Vec::new()),
        })
    }

    fn snapshot(&self) -> Vec<LogRecord> {
        self.records.lock().unwrap().clone()
    }
}

impl LogRecordSink for VecSink {
    fn enqueue(&self, record: LogRecord) {
        self.records.lock().unwrap().push(record);
    }
}

fn sample_record() -> LogRecord {
    LogRecord {
        request_id: Uuid::new_v4(),
        received_at: time::OffsetDateTime::now_utc(),
        provider: Provider::OpenAi,
        model: "gpt-4o".into(),
        attribution: AttributionTags {
            user: Some("steve".into()),
            ..Default::default()
        },
        usage: TokenUsage::new(10, 5),
        cost_usd: 0.01,
        intent: IntentCategory::QuestionAnswering,
        latency_added_ms: 1.0,
        upstream_status: 200,
        streamed: false,
        stream_truncated: false,
        stream_error: false,
        stream_duration_ms: None,
    }
}

#[test]
fn vec_sink_records_dispatched_records() {
    let sink = VecSink::new();
    let r1 = sample_record();
    let r2 = sample_record();

    dispatch_for_test(&r1, &*sink);
    dispatch_for_test(&r2, &*sink);

    let snap = sink.snapshot();
    assert_eq!(snap.len(), 2);
    assert_eq!(snap[0].request_id, r1.request_id);
    assert_eq!(snap[1].request_id, r2.request_id);
}

#[test]
fn sink_is_object_safe() {
    // Compile-time test: trait must be object-safe so we can store
    // `Arc<dyn LogRecordSink>` in a `OnceLock` (see `super::SINK`).
    let _: Arc<dyn LogRecordSink> = VecSink::new();
}
