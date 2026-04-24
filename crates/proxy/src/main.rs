//! Tokenova proxy — entry point.

use anyhow::Context;
use tokenova_proxy::{build_router, AppState};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = tokenova_proxy::config::Config::from_env()?;
    tokenova_proxy::observability::init(config.log_format);

    tracing::info!(
        addr = %config.listen_addr,
        openai = %config.openai_upstream,
        anthropic = %config.anthropic_upstream,
        "tokenova-proxy starting",
    );

    let state = AppState::new(
        config.openai_upstream.clone(),
        config.anthropic_upstream.clone(),
    )?;
    let app = build_router(state);

    let listener = tokio::net::TcpListener::bind(config.listen_addr)
        .await
        .with_context(|| format!("binding {}", config.listen_addr))?;

    axum::serve(listener, app).await?;
    Ok(())
}
