//! Process-level configuration loaded from environment variables.
//!
//! 12-factor by design. No config file. Defaults favor the "drop-in for
//! OpenAI/Anthropic" story — run the binary with no env vars and it proxies
//! to the real providers on port 8080.

use std::env;
use std::net::SocketAddr;

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: SocketAddr,
    pub openai_upstream: String,
    pub anthropic_upstream: String,
    /// Azure OpenAI resource base URL, e.g.
    /// `https://my-resource.openai.azure.com`. When unset, Azure routes
    /// return 503 Service Unavailable. This is the default safe shape —
    /// a proxy with no Azure configuration shouldn't expose routes that
    /// would forward to an unintended upstream.
    pub azure_upstream: Option<String>,
    /// Postgres / TimescaleDB connection string. When unset, durable
    /// persistence is disabled — `LogRecord`s still go to stdout via
    /// `tracing` but nothing writes to a database.
    pub database_url: Option<String>,
    /// How many records to batch per INSERT. Higher = better throughput,
    /// worse latency-to-visible. 100 is a reasonable default.
    pub persistence_batch_size: usize,
    /// Maximum delay between INSERTs even if the batch isn't full.
    pub persistence_flush_ms: u64,
    pub log_format: LogFormat,
    /// Sentry DSN. When unset, Sentry is not initialized and the proxy
    /// runs with zero Sentry overhead — this is the default for tests
    /// and local dev. See `observability::init_sentry`.
    pub sentry_dsn: Option<String>,
    /// Environment label reported to Sentry (e.g. `production`, `staging`).
    /// Defaults to `development` when unset.
    pub sentry_environment: String,
    /// Release tag reported to Sentry (for deploy correlation). Defaults
    /// to the crate version.
    pub sentry_release: String,
    /// Fraction of transactions sampled for Sentry performance monitoring,
    /// 0.0–1.0. Defaults to 0.0 (disabled) — perf monitoring is opt-in.
    pub sentry_traces_sample_rate: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    Json,
    Pretty,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let port: u16 = env::var("TOKENOVA_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8080);
        let listen_addr = SocketAddr::from(([0, 0, 0, 0], port));

        let openai_upstream = env::var("TOKENOVA_OPENAI_UPSTREAM")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        let anthropic_upstream = env::var("TOKENOVA_ANTHROPIC_UPSTREAM")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
        let azure_upstream = env::var("TOKENOVA_AZURE_UPSTREAM")
            .ok()
            .filter(|s| !s.trim().is_empty());

        let database_url = env::var("TOKENOVA_DATABASE_URL")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let persistence_batch_size = env::var("TOKENOVA_PERSISTENCE_BATCH_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(100);
        let persistence_flush_ms = env::var("TOKENOVA_PERSISTENCE_FLUSH_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(500);

        let log_format = match env::var("TOKENOVA_LOG_FORMAT").unwrap_or_default().as_str() {
            "pretty" => LogFormat::Pretty,
            _ => LogFormat::Json,
        };

        let sentry_dsn = env::var("TOKENOVA_SENTRY_DSN")
            .ok()
            .filter(|s| !s.trim().is_empty());
        let sentry_environment =
            env::var("TOKENOVA_SENTRY_ENVIRONMENT").unwrap_or_else(|_| "development".into());
        let sentry_release = env::var("TOKENOVA_SENTRY_RELEASE")
            .unwrap_or_else(|_| format!("tokenova-proxy@{}", env!("CARGO_PKG_VERSION")));
        let sentry_traces_sample_rate = env::var("TOKENOVA_SENTRY_TRACES_SAMPLE_RATE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);

        Ok(Self {
            listen_addr,
            openai_upstream,
            anthropic_upstream,
            azure_upstream,
            database_url,
            persistence_batch_size,
            persistence_flush_ms,
            log_format,
            sentry_dsn,
            sentry_environment,
            sentry_release,
            sentry_traces_sample_rate,
        })
    }
}
