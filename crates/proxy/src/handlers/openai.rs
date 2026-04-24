use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use time::OffsetDateTime;
use tokenova_domain::{LogRecord, Provider};
use uuid::Uuid;

use crate::attribution;
use crate::observability::emit_log_record;
use crate::state::AppState;
use crate::streaming::{wrap_response_stream, OpenAiStreamParser, StreamingAccumulator};
use crate::upstream;
use crate::usage;

/// `POST /v1/chat/completions` — streaming-aware proxy to OpenAI's chat endpoint.
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    proxy(state, headers, body, "/v1/chat/completions").await
}

/// `POST /v1/embeddings` — buffered proxy to OpenAI's embeddings endpoint.
///
/// Embeddings responses aren't streamed by OpenAI (the endpoint has no SSE
/// mode), so this reuses the non-streaming branch only.
pub async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    proxy(state, headers, body, "/v1/embeddings").await
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamInjection {
    /// Request is not a streaming request.
    NotStreaming,
    /// Request already had `stream_options.include_usage = true`.
    AlreadyHadIncludeUsage,
    /// We injected `stream_options.include_usage = true`.
    Injected,
    /// Client explicitly set `stream_options.include_usage = false` — respect it.
    ClientOptedOut,
}

/// If the request is a streaming request, ensure `stream_options.include_usage`
/// is `true` so OpenAI emits the terminal usage frame we need for billing.
/// Never flips an explicit `false` — the client's opt-out is preserved.
fn maybe_inject_stream_options(body: Bytes) -> (Bytes, StreamInjection) {
    let Ok(mut v) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return (body, StreamInjection::NotStreaming);
    };
    let streamed = v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    if !streamed {
        return (body, StreamInjection::NotStreaming);
    }

    // Inspect existing stream_options.include_usage.
    let existing = v
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(|i| i.as_bool());

    match existing {
        Some(true) => (body, StreamInjection::AlreadyHadIncludeUsage),
        Some(false) => (body, StreamInjection::ClientOptedOut),
        None => {
            // Inject include_usage = true into (possibly-missing) stream_options.
            let obj = v.as_object_mut().expect("top-level must be object");
            let so = obj
                .entry("stream_options".to_string())
                .or_insert_with(|| serde_json::json!({}));
            if let Some(so_obj) = so.as_object_mut() {
                so_obj.insert("include_usage".into(), serde_json::Value::Bool(true));
            }
            let new_bytes =
                serde_json::to_vec(&v).expect("re-serializing a parsed Value must succeed");
            (Bytes::from(new_bytes), StreamInjection::Injected)
        }
    }
}

async fn proxy(
    state: AppState,
    headers: HeaderMap,
    body: Bytes,
    path: &str,
) -> Result<Response, (StatusCode, String)> {
    let request_id = Uuid::new_v4();
    let received_at = OffsetDateTime::now_utc();
    let started = Instant::now();

    let attribution = attribution::extract(&headers);

    let streamed = is_streaming_request(&body);
    let prompt_text = extract_openai_prompt(&body);
    let request_model = extract_openai_model(&body);

    // Conditionally rewrite the body to inject include_usage. Non-streaming
    // passes through unchanged.
    let (upstream_body, injection) = if streamed {
        maybe_inject_stream_options(body)
    } else {
        (body, StreamInjection::NotStreaming)
    };

    if injection == StreamInjection::ClientOptedOut {
        tracing::warn!(
            request_id = %request_id,
            "client set stream_options.include_usage=false; streaming usage will be zero"
        );
    }

    let url = format!("{}{}", state.openai_upstream.trim_end_matches('/'), path);

    let upstream_resp = upstream::forward(&state.http, url, &headers, upstream_body)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")))?;

    let status = upstream_resp.status();
    let resp_headers = upstream_resp.headers().clone();

    // Classify off the hot path. We do this for both streaming and
    // non-streaming so the streaming accumulator has `intent` ready without
    // an async call from `Drop`.
    let classifier = state.classifier.clone();
    let prompt_owned = prompt_text;
    let intent_handle = tokio::task::spawn_blocking(move || classifier.classify(&prompt_owned));
    let intent = intent_handle
        .await
        .unwrap_or(tokenova_domain::IntentCategory::Other);

    if streamed {
        build_streaming_response(
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
        build_buffered_response(
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
async fn build_buffered_response(
    state: AppState,
    resp_headers: axum::http::HeaderMap,
    status: axum::http::StatusCode,
    upstream_resp: reqwest::Response,
    request_id: Uuid,
    received_at: OffsetDateTime,
    started: Instant,
    attribution: tokenova_domain::AttributionTags,
    request_model: String,
    intent: tokenova_domain::IntentCategory,
) -> Result<Response, (StatusCode, String)> {
    let resp_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("reading upstream: {e}")))?;

    let latency_added_ms = started.elapsed().as_secs_f64() * 1000.0;

    let token_usage = usage::parse_openai(&resp_bytes);
    let response_model = extract_openai_response_model(&resp_bytes).unwrap_or(request_model);
    let cost_usd = state
        .pricing
        .rate(Provider::OpenAi, &response_model)
        .cost_usd(&token_usage);

    let record = LogRecord {
        request_id,
        received_at,
        provider: Provider::OpenAi,
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

    let mut builder = Response::builder().status(status);
    for (name, value) in resp_headers.iter() {
        let lname = name.as_str().to_ascii_lowercase();
        if matches!(
            lname.as_str(),
            "transfer-encoding" | "connection" | "keep-alive" | "content-length"
        ) {
            continue;
        }
        builder = builder.header(name.clone(), value.clone());
    }
    builder.body(Body::from(resp_bytes)).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("building response: {e}"),
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn build_streaming_response(
    state: AppState,
    resp_headers: axum::http::HeaderMap,
    status: axum::http::StatusCode,
    upstream_resp: reqwest::Response,
    request_id: Uuid,
    received_at: OffsetDateTime,
    started: Instant,
    attribution: tokenova_domain::AttributionTags,
    request_model: String,
    intent: tokenova_domain::IntentCategory,
) -> Result<Response, (StatusCode, String)> {
    // Latency-added = proxy overhead before first byte out, i.e. the elapsed
    // time from handler entry to this point. `stream_duration_ms` captures
    // end-to-end when the stream closes.
    let latency_added_ms = started.elapsed().as_secs_f64() * 1000.0;

    let accumulator = Arc::new(Mutex::new(StreamingAccumulator::new(
        request_id,
        received_at,
        started,
        Provider::OpenAi,
        request_model,
        attribution,
        intent,
        state.pricing.clone(),
        status.as_u16(),
        latency_added_ms,
        Box::new(OpenAiStreamParser::new()),
    )));

    let upstream_stream = upstream_resp.bytes_stream();
    let observed = wrap_response_stream(upstream_stream, accumulator);

    let mut builder = Response::builder().status(status);
    for (name, value) in resp_headers.iter() {
        let lname = name.as_str().to_ascii_lowercase();
        if matches!(
            lname.as_str(),
            "transfer-encoding" | "connection" | "keep-alive" | "content-length"
        ) {
            continue;
        }
        builder = builder.header(name.clone(), value.clone());
    }
    builder.body(Body::from_stream(observed)).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("building response: {e}"),
        )
    })
}

fn is_streaming_request(body: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false)
}

fn extract_openai_model(body: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from))
        .unwrap_or_default()
}

fn extract_openai_response_model(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from))
}

/// Maximum bytes of prompt text fed to the classifier. A hard cap keeps
/// classification latency bounded on pathological inputs and prevents
/// unbounded string growth during extraction (QA Medium #5).
const PROMPT_CAP_BYTES: usize = 2048;

/// Append at most `remaining = PROMPT_CAP_BYTES - out.len()` bytes of `s` to
/// `out`. Returns true if the cap has been reached (caller should break).
fn append_capped(out: &mut String, s: &str) -> bool {
    if out.len() >= PROMPT_CAP_BYTES {
        return true;
    }
    let remaining = PROMPT_CAP_BYTES - out.len();
    if s.len() <= remaining {
        out.push_str(s);
    } else {
        // Find a safe UTF-8 boundary at or below `remaining`.
        let mut cut = remaining;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        out.push_str(&s[..cut]);
    }
    out.len() >= PROMPT_CAP_BYTES
}

/// Pull the user-visible prompt text out of an OpenAI chat-completions
/// request for classification. Concatenates the `content` field of every
/// `user` and `system` message, capped incrementally at `PROMPT_CAP_BYTES`
/// to bound memory on pathological inputs (QA Medium #5).
fn extract_openai_prompt(body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return String::new();
    };
    let mut out = String::new();
    if let Some(messages) = v.get("messages").and_then(|m| m.as_array()) {
        'outer: for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if !matches!(role, "user" | "system") {
                continue;
            }
            if let Some(content) = msg.get("content") {
                if let Some(s) = content.as_str() {
                    if append_capped(&mut out, s) {
                        break 'outer;
                    }
                    if append_capped(&mut out, " ") {
                        break 'outer;
                    }
                } else if let Some(parts) = content.as_array() {
                    for part in parts {
                        if let Some(s) = part.get("text").and_then(|t| t.as_str()) {
                            if append_capped(&mut out, s) {
                                break 'outer;
                            }
                            if append_capped(&mut out, " ") {
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    } else if let Some(prompt) = v.get("prompt").and_then(|p| p.as_str()) {
        append_capped(&mut out, prompt);
    } else if let Some(input) = v.get("input") {
        // Embeddings endpoint.
        if let Some(s) = input.as_str() {
            append_capped(&mut out, s);
        } else if let Some(arr) = input.as_array() {
            for item in arr {
                if let Some(s) = item.as_str() {
                    if append_capped(&mut out, s) {
                        break;
                    }
                    if append_capped(&mut out, " ") {
                        break;
                    }
                }
            }
        }
    }
    // Belt-and-suspenders.
    out.truncate(PROMPT_CAP_BYTES);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_openai_prompt_bounded_under_many_messages() {
        // 1000 messages, each with ~1 KB of content. Output must be
        // <= PROMPT_CAP_BYTES, and the function must return quickly.
        let big_chunk = "a".repeat(1024);
        let mut messages = Vec::with_capacity(1000);
        for _ in 0..1000 {
            messages.push(serde_json::json!({
                "role": "user",
                "content": big_chunk,
            }));
        }
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "gpt-4o",
            "messages": messages,
        }))
        .unwrap();

        let start = std::time::Instant::now();
        let out = extract_openai_prompt(&body);
        let elapsed = start.elapsed();
        assert!(out.len() <= PROMPT_CAP_BYTES, "len={}", out.len());
        // Sanity: the function should finish in well under a second even in
        // debug builds.
        assert!(elapsed.as_secs() < 2, "took too long: {elapsed:?}");
    }

    #[test]
    fn inject_stream_options_on_streaming_request_without_stream_options() {
        let body = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "gpt-4o-mini",
                "stream": true,
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        );
        let (new_body, inj) = maybe_inject_stream_options(body);
        assert_eq!(inj, StreamInjection::Injected);
        let v: serde_json::Value = serde_json::from_slice(&new_body).unwrap();
        assert_eq!(
            v.get("stream_options").and_then(|o| o.get("include_usage")),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[test]
    fn respect_client_opt_out_false() {
        let body = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "gpt-4o-mini",
                "stream": true,
                "stream_options": {"include_usage": false},
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        );
        let (new_body, inj) = maybe_inject_stream_options(body.clone());
        assert_eq!(inj, StreamInjection::ClientOptedOut);
        assert_eq!(new_body, body);
    }

    #[test]
    fn passthrough_when_already_true() {
        let body = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "gpt-4o-mini",
                "stream": true,
                "stream_options": {"include_usage": true},
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        );
        let (new_body, inj) = maybe_inject_stream_options(body.clone());
        assert_eq!(inj, StreamInjection::AlreadyHadIncludeUsage);
        assert_eq!(new_body, body);
    }

    #[test]
    fn noop_when_not_streaming() {
        let body = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        );
        let (new_body, inj) = maybe_inject_stream_options(body.clone());
        assert_eq!(inj, StreamInjection::NotStreaming);
        assert_eq!(new_body, body);
    }
}
