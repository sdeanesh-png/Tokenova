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
use crate::upstream;
use crate::usage;

/// `POST /v1/chat/completions` — buffered proxy to OpenAI's chat endpoint.
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    proxy(state, headers, body, "/v1/chat/completions").await
}

/// `POST /v1/embeddings` — buffered proxy to OpenAI's embeddings endpoint.
pub async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    proxy(state, headers, body, "/v1/embeddings").await
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

    let url = format!("{}{}", state.openai_upstream.trim_end_matches('/'), path);

    let upstream_resp = upstream::forward(&state.http, url, &headers, body)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("upstream error: {e}")))?;

    let status = upstream_resp.status();
    let resp_headers = upstream_resp.headers().clone();
    let resp_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("reading upstream: {e}")))?;

    let latency_added_ms = started.elapsed().as_secs_f64() * 1000.0;

    let token_usage = if streamed {
        Default::default()
    } else {
        usage::parse_openai(&resp_bytes)
    };
    let response_model = extract_openai_response_model(&resp_bytes).unwrap_or(request_model);
    let cost_usd = state
        .pricing
        .rate(Provider::OpenAi, &response_model)
        .cost_usd(&token_usage);

    // Off-the-hot-path classification: spawn and await before emitting the
    // log record, but after the response body has been collected. The client
    // latency measurement above is taken before classification runs.
    let classifier = state.classifier.clone();
    let prompt_owned = prompt_text;
    let intent_handle = tokio::task::spawn_blocking(move || classifier.classify(&prompt_owned));
    let intent = intent_handle
        .await
        .unwrap_or(tokenova_domain::IntentCategory::Other);

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
        streamed,
    };
    emit_log_record(&record);

    let mut builder = Response::builder().status(status);
    for (name, value) in resp_headers.iter() {
        // Skip hop-by-hop headers so downstream sees a clean response.
        let lname = name.as_str().to_ascii_lowercase();
        if matches!(
            lname.as_str(),
            "transfer-encoding" | "connection" | "keep-alive" | "content-length"
        ) {
            continue;
        }
        builder = builder.header(name.clone(), value.clone());
    }
    Ok(builder
        .body(Body::from(resp_bytes))
        .expect("response builder"))
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

/// Pull the user-visible prompt text out of an OpenAI chat-completions
/// request for classification. Concatenates the `content` field of every
/// `user` and `system` message. Keeps only the first 2KB to bound
/// classifier work for unusually large prompts.
fn extract_openai_prompt(body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return String::new();
    };
    let mut out = String::new();
    if let Some(messages) = v.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            if !matches!(role, "user" | "system") {
                continue;
            }
            if let Some(content) = msg.get("content") {
                if let Some(s) = content.as_str() {
                    out.push_str(s);
                    out.push(' ');
                } else if let Some(parts) = content.as_array() {
                    for part in parts {
                        if let Some(s) = part.get("text").and_then(|t| t.as_str()) {
                            out.push_str(s);
                            out.push(' ');
                        }
                    }
                }
            }
        }
    } else if let Some(prompt) = v.get("prompt").and_then(|p| p.as_str()) {
        out.push_str(prompt);
    } else if let Some(input) = v.get("input") {
        // Embeddings endpoint
        if let Some(s) = input.as_str() {
            out.push_str(s);
        } else if let Some(arr) = input.as_array() {
            for item in arr {
                if let Some(s) = item.as_str() {
                    out.push_str(s);
                    out.push(' ');
                }
            }
        }
    }
    out.truncate(2048);
    out
}
