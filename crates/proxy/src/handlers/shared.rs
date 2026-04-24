//! Provider-agnostic proxy request pipeline.
//!
//! Both the OpenAI and Anthropic handlers run the exact same sequence:
//! extract attribution, detect streaming, extract the prompt for
//! classification, optionally rewrite the request body, forward upstream,
//! classify, then emit exactly one `LogRecord` (directly for non-streaming,
//! via the [`crate::streaming::StreamingAccumulator`] for streaming). The
//! only per-provider variation is in four hooks exposed through the
//! [`ProviderContract`] trait.
//!
//! Before this refactor the two handler files were ~90% identical — any
//! bug fix risked being applied to only one of them (QA R1 Low #8).

use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::Body;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use time::OffsetDateTime;
use tokenova_domain::{AttributionTags, IntentCategory, LogRecord, Provider, TokenUsage};
use uuid::Uuid;

use crate::attribution;
use crate::observability::emit_log_record;
use crate::state::AppState;
use crate::streaming::{wrap_response_stream, StreamingAccumulator, StreamingUsageParser};
use crate::upstream;

/// Per-provider behavior injected into the shared pipeline.
pub(crate) trait ProviderContract: 'static {
    const PROVIDER: Provider;
    /// Extract the user-visible prompt text for intent classification.
    fn extract_prompt(body: &[u8]) -> String;
    /// Parse `TokenUsage` from a non-streaming provider response body.
    fn parse_buffered_usage(body: &[u8]) -> TokenUsage;
    /// Build a fresh streaming-usage parser for a new SSE stream.
    fn streaming_parser() -> Box<dyn StreamingUsageParser>;
    /// Optionally rewrite the outgoing body on the streaming path.
    /// Default: identity (no mutation). OpenAI overrides to inject
    /// `stream_options.include_usage = true`.
    fn prepare_streaming_body(body: Bytes) -> (Bytes, BodyMutation) {
        (body, BodyMutation::None)
    }
}

/// Outcome of an optional body mutation on the streaming path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BodyMutation {
    /// Body forwarded unchanged.
    None,
    /// We injected `stream_options.include_usage = true`.
    InjectedIncludeUsage,
    /// Client had `stream_options.include_usage = true` already.
    AlreadyHadIncludeUsage,
    /// Client explicitly set `stream_options.include_usage = false`.
    /// Respect the choice; usage will be zero.
    ClientOptedOut,
}

/// The shared pipeline. Every proxy request flows through here.
pub(crate) async fn proxy_request<P: ProviderContract>(
    state: AppState,
    headers: HeaderMap,
    body: Bytes,
    url: String,
) -> Result<Response, (StatusCode, String)> {
    let request_id = Uuid::new_v4();
    let received_at = OffsetDateTime::now_utc();
    let started = Instant::now();

    let attribution = attribution::extract(&headers);

    let streamed = is_streaming_request(&body);
    let prompt_text = P::extract_prompt(&body);
    let request_model = extract_model(&body);

    let (upstream_body, mutation) = if streamed {
        P::prepare_streaming_body(body)
    } else {
        (body, BodyMutation::None)
    };

    if mutation == BodyMutation::ClientOptedOut {
        tracing::warn!(
            request_id = %request_id,
            provider = P::PROVIDER.as_str(),
            "client set stream_options.include_usage=false; streaming usage will be zero"
        );
    }

    let upstream_resp = upstream::forward(&state.http, url, &headers, upstream_body)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")))?;

    let status = upstream_resp.status();
    let resp_headers = upstream_resp.headers().clone();

    // Inline classification. The heuristic embedder benches at ~2.6 µs per
    // call, well under the per-request budget; spawn_blocking's thread-pool
    // hop was pure overhead (QA R1 Low #9). Restore it when the ONNX
    // backend (~1–10 ms) lands — feature-gate the wrapper at that point.
    let intent = state.classifier.classify(&prompt_text);

    if streamed {
        build_streaming_response::<P>(
            state,
            resp_headers,
            status,
            upstream_resp,
            request_id,
            received_at,
            started,
            attribution,
            request_model,
            intent,
        )
    } else {
        build_buffered_response::<P>(
            state,
            resp_headers,
            status,
            upstream_resp,
            request_id,
            received_at,
            started,
            attribution,
            request_model,
            intent,
        )
        .await
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_buffered_response<P: ProviderContract>(
    state: AppState,
    resp_headers: HeaderMap,
    status: StatusCode,
    upstream_resp: reqwest::Response,
    request_id: Uuid,
    received_at: OffsetDateTime,
    started: Instant,
    attribution: AttributionTags,
    request_model: String,
    intent: IntentCategory,
) -> Result<Response, (StatusCode, String)> {
    let resp_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("reading upstream: {e}")))?;

    let latency_added_ms = started.elapsed().as_secs_f64() * 1000.0;
    let token_usage = P::parse_buffered_usage(&resp_bytes);
    let response_model = extract_response_model(&resp_bytes).unwrap_or(request_model);
    let cost_usd = state
        .pricing
        .rate(P::PROVIDER, &response_model)
        .cost_usd(&token_usage);

    let record = LogRecord {
        request_id,
        received_at,
        provider: P::PROVIDER,
        model: response_model,
        attribution,
        usage: token_usage,
        cost_usd,
        intent,
        latency_added_ms,
        upstream_status: status.as_u16(),
        streamed: false,
        stream_truncated: false,
        stream_error: false,
        stream_duration_ms: None,
    };
    emit_log_record(&record);

    build_response(status, &resp_headers, Body::from(resp_bytes))
}

#[allow(clippy::too_many_arguments)]
fn build_streaming_response<P: ProviderContract>(
    state: AppState,
    resp_headers: HeaderMap,
    status: StatusCode,
    upstream_resp: reqwest::Response,
    request_id: Uuid,
    received_at: OffsetDateTime,
    started: Instant,
    attribution: AttributionTags,
    request_model: String,
    intent: IntentCategory,
) -> Result<Response, (StatusCode, String)> {
    // Latency-added = proxy overhead before first byte out. Total wall time
    // from handler entry to stream close lands in `stream_duration_ms`.
    let latency_added_ms = started.elapsed().as_secs_f64() * 1000.0;

    let accumulator = Arc::new(Mutex::new(StreamingAccumulator::new(
        request_id,
        received_at,
        started,
        P::PROVIDER,
        request_model,
        attribution,
        intent,
        state.pricing.clone(),
        status.as_u16(),
        latency_added_ms,
        P::streaming_parser(),
    )));

    let upstream_stream = upstream_resp.bytes_stream();
    let observed = wrap_response_stream(upstream_stream, accumulator);

    build_response(status, &resp_headers, Body::from_stream(observed))
}

/// Build the downstream response with upstream headers forwarded, minus
/// hop-by-hop entries that would confuse the client's HTTP stack.
fn build_response(
    status: StatusCode,
    upstream_headers: &HeaderMap,
    body: Body,
) -> Result<Response, (StatusCode, String)> {
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream_headers.iter() {
        let lname = name.as_str().to_ascii_lowercase();
        if matches!(
            lname.as_str(),
            "transfer-encoding" | "connection" | "keep-alive" | "content-length"
        ) {
            continue;
        }
        builder = builder.header(name.clone(), value.clone());
    }
    builder.body(body).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("building response: {e}"),
        )
    })
}

/// Top-level `"stream": true` detector — same shape for OpenAI and
/// Anthropic request bodies.
pub(crate) fn is_streaming_request(body: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false)
}

/// Extract the `"model"` field from a request body. Returns empty string
/// when absent or unparseable. Both providers use the same key.
pub(crate) fn extract_model(body: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from))
        .unwrap_or_default()
}

/// Extract the `"model"` field from a response body (buffered path only).
pub(crate) fn extract_response_model(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from))
}

/// Hard cap on classifier input size. Prevents unbounded concatenation
/// during prompt extraction on pathological requests (QA R1 Medium #5).
pub(crate) const PROMPT_CAP_BYTES: usize = 2048;

/// Append `s` to `out`, truncated at `PROMPT_CAP_BYTES`. Returns `true` if
/// the cap was reached and the caller should break out of its outer loop.
pub(crate) fn append_capped(out: &mut String, s: &str) -> bool {
    if out.len() >= PROMPT_CAP_BYTES {
        return true;
    }
    let remaining = PROMPT_CAP_BYTES - out.len();
    if s.len() <= remaining {
        out.push_str(s);
    } else {
        // Safe UTF-8 boundary at or below `remaining`.
        let mut cut = remaining;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        out.push_str(&s[..cut]);
    }
    out.len() >= PROMPT_CAP_BYTES
}
