//! Durable persistence for emitted [`LogRecord`]s.
//!
//! This module decouples the proxy's request path from the database. The
//! global [`dispatch`] entry point — called once per request from
//! [`crate::observability::emit_log_record`] — is sync, fire-and-forget,
//! and a no-op when no sink is installed. Production deployments install
//! a [`PostgresSink`] that owns an unbounded mpsc channel; a background
//! writer task drains the channel, batches records, and writes to
//! Postgres / TimescaleDB.
//!
//! Failure modes:
//!
//! * **No DB configured** — `dispatch` does nothing. `LogRecord`s still
//!   reach stdout via the existing `tracing::info!` emission. This is
//!   the default for tests, local dev, and any deployment where
//!   `TOKENOVA_DATABASE_URL` isn't set.
//! * **DB unreachable mid-flight** — the writer task logs at `warn` and
//!   drops the batch. Channel growth is unbounded, so a sustained DB
//!   outage will accumulate memory. (A bounded channel + drop-on-full
//!   policy is a Slice-N follow-up; flagged for the next QA round.)
//! * **DB schema drift** — sqlx insert errors are logged at `error` and
//!   the batch is dropped. Migrations run at startup, so live drift is
//!   only possible if someone hot-altered the table.

use std::sync::{Arc, OnceLock};

use tokenova_domain::LogRecord;

mod postgres;

#[cfg(test)]
mod tests;

pub use postgres::{PostgresSink, PostgresSinkConfig};

/// Object-safe trait for any backend that can persist `LogRecord`s.
///
/// Implementations must be cheap to clone (`Arc` internally) and never
/// block the calling thread — `enqueue` is called from the request hot
/// path. Real implementations push onto a channel; tests use a vector
/// behind a `Mutex`.
pub trait LogRecordSink: Send + Sync + 'static {
    fn enqueue(&self, record: LogRecord);
}

static SINK: OnceLock<Arc<dyn LogRecordSink>> = OnceLock::new();

/// Install the process-wide sink. Idempotent — only the first install
/// wins. Returns `true` if this call installed the sink.
pub fn install_sink(sink: Arc<dyn LogRecordSink>) -> bool {
    SINK.set(sink).is_ok()
}

/// Forward a record to the installed sink. No-op if none is installed.
/// Called from [`crate::observability::emit_log_record`].
pub(crate) fn dispatch(record: &LogRecord) {
    if let Some(sink) = SINK.get() {
        sink.enqueue(record.clone());
    }
}

/// Test-only: clear the installed sink so successive tests can install
/// their own. `OnceLock` doesn't expose `take`, so this uses `take()` on
/// a wrapper... actually it can't — instead we use a separate atomic
/// indirection in tests. See `tests::install_test_sink` which deliberately
/// uses a different mechanism. This function intentionally does NOT
/// exist outside `cfg(test)` — global state is set-once in production.
#[cfg(test)]
pub(crate) fn dispatch_for_test(record: &LogRecord, sink: &dyn LogRecordSink) {
    sink.enqueue(record.clone());
}
