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
use crate::streaming::{wrap_response_stream, AnthropicStreamParser, StreamingAccumulator};
use crate::upstream;
use crate::usage;

/// `POST /v1/messages` — streaming-aware proxy to Anthropic's messages endpoint.
pub async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    let request_id = Uuid::new_v4();
    let received_at = OffsetDateTime::now_utc();
    let started = Instant::now();

    let attribution = attribution::extract(&headers);

    let streamed = is_streaming_request(&body);
    let prompt_text = extract_anthropic_prompt(&body);
    let request_model = extract_anthropic_model(&body);

    let url = format!(
        "{}/v1/messages",
        state.anthropic_upstream.trim_end_matches('/')
    );

    let upstream_resp = upstream::forward(&state.http, url, &headers, body)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")))?;

    let status = upstream_resp.status();
    let resp_headers = upstream_resp.headers().clone();

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

    let token_usage = usage::parse_anthropic(&resp_bytes);
    let response_model = extract_anthropic_response_model(&resp_bytes).unwrap_or(request_model);
    let cost_usd = state
        .pricing
        .rate(Provider::Anthropic, &response_model)
        .cost_usd(&token_usage);

    let record = LogRecord {
        request_id,
        received_at,
        provider: Provider::Anthropic,
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
    let latency_added_ms = started.elapsed().as_secs_f64() * 1000.0;

    let accumulator = Arc::new(Mutex::new(StreamingAccumulator::new(
        request_id,
        received_at,
        started,
        Provider::Anthropic,
        request_model,
        attribution,
        intent,
        state.pricing.clone(),
        status.as_u16(),
        latency_added_ms,
        Box::new(AnthropicStreamParser::new()),
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

fn extract_anthropic_model(body: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from))
        .unwrap_or_default()
}

fn extract_anthropic_response_model(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from))
}

/// Maximum bytes of prompt text fed to the classifier. See QA Medium #5.
const PROMPT_CAP_BYTES: usize = 2048;

fn append_capped(out: &mut String, s: &str) -> bool {
    if out.len() >= PROMPT_CAP_BYTES {
        return true;
    }
    let remaining = PROMPT_CAP_BYTES - out.len();
    if s.len() <= remaining {
        out.push_str(s);
    } else {
        let mut cut = remaining;
        while cut > 0 && !s.is_char_boundary(cut) {
            cut -= 1;
        }
        out.push_str(&s[..cut]);
    }
    out.len() >= PROMPT_CAP_BYTES
}

/// Pull the user prompt out of an Anthropic messages request. Anthropic's
/// shape nests `content` as either a string or an array of content blocks
/// (`{type: "text", text: "..."}`). Capped incrementally at
/// `PROMPT_CAP_BYTES` to bound memory on pathological inputs (QA Medium #5).
fn extract_anthropic_prompt(body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return String::new();
    };
    let mut out = String::new();

    // Anthropic places system text in a top-level `system` field.
    if let Some(s) = v.get("system").and_then(|s| s.as_str()) {
        if append_capped(&mut out, s) {
            out.truncate(PROMPT_CAP_BYTES);
            return out;
        }
        if append_capped(&mut out, " ") {
            out.truncate(PROMPT_CAP_BYTES);
            return out;
        }
    }

    if let Some(messages) = v.get("messages").and_then(|m| m.as_array()) {
        'outer: for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if role != "user" {
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
                        let ptype = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if ptype == "text" {
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
        }
    }
    out.truncate(PROMPT_CAP_BYTES);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_anthropic_prompt_bounded_under_many_messages() {
        let big_chunk = "b".repeat(1024);
        let mut messages = Vec::with_capacity(1000);
        for _ in 0..1000 {
            messages.push(serde_json::json!({
                "role": "user",
                "content": big_chunk,
            }));
        }
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 64,
            "messages": messages,
        }))
        .unwrap();

        let start = std::time::Instant::now();
        let out = extract_anthropic_prompt(&body);
        let elapsed = start.elapsed();
        assert!(out.len() <= PROMPT_CAP_BYTES, "len={}", out.len());
        assert!(elapsed.as_secs() < 2, "took too long: {elapsed:?}");
    }
}
