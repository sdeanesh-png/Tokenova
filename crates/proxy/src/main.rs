//! Tokenova proxy — entry point.

use anyhow::Context;
use tokenova_proxy::{build_router, AppState};

fn main() -> anyhow::Result<()> {
    let config = tokenova_proxy::config::Config::from_env()?;

    // Sentry MUST be initialized before the tokio runtime starts so its
    // panic hook is registered on the main thread. The guard must live
    // for the process lifetime — dropping it flushes + disables Sentry.
    let _sentry = tokenova_proxy::observability::init_sentry(&config);
    tokenova_proxy::observability::init_tracing(config.log_format);

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async { run(config).await })
}

async fn run(config: tokenova_proxy::config::Config) -> anyhow::Result<()> {
    tracing::info!(
        addr = %config.listen_addr,
        openai = %config.openai_upstream,
        anthropic = %config.anthropic_upstream,
        azure = ?config.azure_upstream,
        sentry = config.sentry_dsn.is_some(),
        persistence = config.database_url.is_some(),
        "tokenova-proxy starting",
    );

    // Optional durable persistence. When TOKENOVA_DATABASE_URL is unset
    // this is skipped entirely — `LogRecord`s only reach stdout.
    if let Some(database_url) = config.database_url.clone() {
        let cfg = tokenova_proxy::persistence::PostgresSinkConfig {
            database_url,
            max_connections: 8,
            batch_size: config.persistence_batch_size,
            flush_interval: std::time::Duration::from_millis(config.persistence_flush_ms),
        };
        let sink = tokenova_proxy::persistence::PostgresSink::start(cfg)
            .await
            .context("starting Postgres persistence sink")?;
        tokenova_proxy::persistence::install_sink(sink);
        tracing::info!("durable persistence enabled");
    }

    let state = AppState::new(
        config.openai_upstream.clone(),
        config.anthropic_upstream.clone(),
        config.azure_upstream.clone(),
    )?;
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(config.listen_addr)
        .await
        .with_context(|| format!("binding {}", config.listen_addr))?;

    axum::serve(listener, app).await?;
    Ok(())
}
