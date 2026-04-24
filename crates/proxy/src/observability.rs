//! Structured log emission, tracing bootstrap, and Sentry wiring.

use sentry::ClientInitGuard;
use tokenova_domain::LogRecord;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::config::{Config, LogFormat};

/// Initialize the Sentry client if `TOKENOVA_SENTRY_DSN` is set. When the
/// DSN is unset (tests, local dev) this is a no-op and Sentry is never
/// activated — no outbound connections, no panic hook, no overhead.
///
/// The returned guard **must be held for the process lifetime** (typically
/// `let _guard = observability::init_sentry(&config);` at the top of
/// `main`). Dropping the guard flushes any queued events and disables the
/// client, so dropping early silently disables error reporting.
#[must_use = "Sentry guard must live for the lifetime of the process"]
pub fn init_sentry(config: &Config) -> Option<ClientInitGuard> {
    let dsn = config.sentry_dsn.as_ref()?;

    let guard = sentry::init((
        dsn.as_str(),
        sentry::ClientOptions {
            environment: Some(config.sentry_environment.clone().into()),
            release: Some(config.sentry_release.clone().into()),
            traces_sample_rate: config.sentry_traces_sample_rate,
            attach_stacktrace: true,
            // Send events synchronously at shutdown so we don't lose the
            // last error when a container is killed.
            shutdown_timeout: std::time::Duration::from_secs(3),
            ..Default::default()
        },
    ));
    Some(guard)
}

/// Initialize the `tracing` subscriber with Sentry breadcrumbs + event
/// capture layered in. When Sentry isn't initialized, the
/// `sentry_tracing::layer()` layer is still installed but is inert —
/// breadcrumbs go nowhere and `tracing::error!` calls don't produce
/// Sentry events.
pub fn init_tracing(format: LogFormat) {
    let filter = EnvFilter::try_from_env("TOKENOVA_LOG").unwrap_or_else(|_| EnvFilter::new("info"));

    // Sentry layer: all `tracing::error!` calls become Sentry events,
    // `tracing::warn!` and `info!` are captured as breadcrumbs.
    let sentry_layer = sentry_tracing::layer();

    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(sentry_layer);

    match format {
        LogFormat::Json => {
            registry
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_span_events(FmtSpan::NONE)
                        .with_target(false)
                        .json()
                        .flatten_event(true),
                )
                .init();
        }
        LogFormat::Pretty => {
            registry
                .with(
                    tracing_subscriber::fmt::layer()
                        .with_span_events(FmtSpan::NONE)
                        .with_target(false)
                        .compact(),
                )
                .init();
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

    #[cfg(any(test, feature = "test-utils"))]
    test_sink::dispatch(record);
}

/// Test-log-capture mechanism: lets tests register a sink that receives
/// every `LogRecord` emitted by `emit_log_record`.
///
/// Compiled only under `cfg(test)` (unit tests) or the `test-utils` Cargo
/// feature (integration tests, which can't see `cfg(test)` items in the
/// library target). Production builds omit this module entirely — no
/// runtime overhead and no API surface for a third-party consumer to
/// register a sink that could exfiltrate `LogRecord`s in a deployed proxy
/// (QA R2 Low #2).
#[cfg(any(test, feature = "test-utils"))]
pub mod test_sink {
    use std::sync::{Arc, OnceLock, RwLock};

    use tokenova_domain::LogRecord;

    type Sink = Arc<dyn Fn(&LogRecord) + Send + Sync>;

    fn cell() -> &'static RwLock<Option<Sink>> {
        static CELL: OnceLock<RwLock<Option<Sink>>> = OnceLock::new();
        CELL.get_or_init(|| RwLock::new(None))
    }

    /// Register a sink that will be invoked for every emitted `LogRecord`.
    /// Overwrites any previously registered sink.
    pub fn set_test_sink<F>(sink: F)
    where
        F: Fn(&LogRecord) + Send + Sync + 'static,
    {
        let mut guard = cell().write().expect("test sink lock poisoned");
        *guard = Some(Arc::new(sink));
    }

    /// Remove any previously registered sink.
    pub fn clear_test_sink() {
        let mut guard = cell().write().expect("test sink lock poisoned");
        *guard = None;
    }

    pub(super) fn dispatch(record: &LogRecord) {
        // Clone the Arc out of the lock so the sink itself runs without
        // holding the RwLock — sinks might do arbitrary work and we don't
        // want to deadlock with a concurrent set/clear.
        let sink = {
            let guard = cell().read().expect("test sink lock poisoned");
            guard.clone()
        };
        if let Some(sink) = sink {
            sink(record);
        }
    }
}
