//! Axum router wiring.

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use tower_http::trace::TraceLayer;

use crate::handlers;
use crate::state::AppState;

/// Maximum request body size accepted by the proxy, in bytes.
///
/// Tokenova is a prompt proxy — a 256 KB cap is generous for enterprise
/// chat/completion prompts while preventing the axum-default 2 MB unbuffered
/// body DoS surface (QA Blocker #1).
pub const MAX_REQUEST_BODY_BYTES: usize = 256 * 1024;

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route(
            "/v1/chat/completions",
            post(handlers::openai::chat_completions),
        )
        .route("/v1/embeddings", post(handlers::openai::embeddings))
        .route("/v1/messages", post(handlers::anthropic::messages))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}
