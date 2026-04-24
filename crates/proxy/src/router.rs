//! Axum router wiring.

use axum::routing::{get, post};
use axum::Router;
use tower_http::trace::TraceLayer;

use crate::handlers;
use crate::state::AppState;

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(health))
        .route(
            "/v1/chat/completions",
            post(handlers::openai::chat_completions),
        )
        .route("/v1/embeddings", post(handlers::openai::embeddings))
        .route("/v1/messages", post(handlers::anthropic::messages))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}
