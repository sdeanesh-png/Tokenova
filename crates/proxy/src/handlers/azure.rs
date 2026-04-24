//! Azure OpenAI route handlers.
//!
//! Azure OpenAI is wire-compatible with OpenAI — same request/response
//! body shape, same SSE streaming format — but:
//!
//! * URLs are deployment-based:
//!   `POST /openai/deployments/{deployment}/chat/completions?api-version=...`
//!   The "model" identifier is the deployment name in the URL path, not
//!   a `"model"` field in the body.
//! * Auth via the `api-key` header (not `Authorization: Bearer`). Both
//!   styles pass through our header forwarder unchanged.
//!
//! Because the body shape is identical, this handler reuses OpenAI's
//! prompt extractor, usage parser, streaming parser, and
//! `stream_options.include_usage` injection. Only the URL construction
//! and model-from-path logic is Azure-specific.

use axum::extract::{OriginalUri, Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use tokenova_domain::{Provider, TokenUsage};

use crate::handlers::openai;
use crate::handlers::shared::{proxy_request, BodyMutation, ProviderContract};
use crate::state::AppState;
use crate::streaming::{OpenAiStreamParser, StreamingUsageParser};
use crate::usage;

/// `POST /openai/deployments/{deployment}/chat/completions?api-version=...`
pub async fn chat_completions(
    State(state): State<AppState>,
    Path(deployment): Path<String>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    dispatch(state, deployment, uri, headers, body).await
}

/// `POST /openai/deployments/{deployment}/completions?api-version=...`
pub async fn completions(
    State(state): State<AppState>,
    Path(deployment): Path<String>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    dispatch(state, deployment, uri, headers, body).await
}

/// `POST /openai/deployments/{deployment}/embeddings?api-version=...`
pub async fn embeddings(
    State(state): State<AppState>,
    Path(deployment): Path<String>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    dispatch(state, deployment, uri, headers, body).await
}

async fn dispatch(
    state: AppState,
    deployment: String,
    uri: axum::http::Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    let Some(azure_base) = state.azure_upstream.clone() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Azure OpenAI not configured; set TOKENOVA_AZURE_UPSTREAM".into(),
        ));
    };

    // Preserve the path + query verbatim so `api-version` and any other
    // query params reach Azure unchanged. The Azure-style URL the client
    // used IS the URL we forward.
    let path_and_query = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let url = format!("{}{}", azure_base.trim_end_matches('/'), path_and_query);

    proxy_request::<AzureContract>(state, headers, body, url, deployment).await
}

pub(crate) struct AzureContract;

impl ProviderContract for AzureContract {
    const PROVIDER: Provider = Provider::AzureOpenAi;

    fn extract_prompt(body: &[u8]) -> String {
        // Same request shape as OpenAI.
        openai::extract_openai_prompt(body)
    }

    fn parse_buffered_usage(body: &[u8]) -> TokenUsage {
        // Same response shape as OpenAI.
        usage::parse_openai(body)
    }

    fn streaming_parser() -> Box<dyn StreamingUsageParser> {
        // Same SSE wire format as OpenAI.
        Box::new(OpenAiStreamParser::new())
    }

    fn prepare_streaming_body(body: Bytes) -> (Bytes, BodyMutation) {
        // Same `stream_options.include_usage` semantics as OpenAI.
        // Requires Azure API version 2024-08-06-preview or later for the
        // usage frame; older versions will still proxy cleanly but will
        // report zero streaming usage.
        openai::maybe_inject_stream_options(body)
    }
}
