//! Structured log emission and tracing bootstrap.

use tokenova_domain::LogRecord;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

use crate::config::LogFormat;

pub fn init(format: LogFormat) {
    let filter = EnvFilter::try_from_env("TOKENOVA_LOG").unwrap_or_else(|_| EnvFilter::new("info"));

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::NONE)
        .with_target(false);

    match format {
        LogFormat::Json => {
            builder.json().flatten_event(true).init();
        }
        LogFormat::Pretty => {
            builder.compact().init();
        }
    }
}

/// Emit one structured log record per proxied request. Serializes the record
/// into the event rather than relying on `tracing`'s field-by-field encoding
/// so downstream ingesters see a single canonical shape.
pub fn emit_log_record(record: &LogRecord) {
    match serde_json::to_string(record) {
        Ok(json) => tracing::info!(target: "tokenova.request", record = %json),
        Err(err) => tracing::error!(?err, "failed to serialize LogRecord"),
    }
}
