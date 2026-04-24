//! Anthropic route handler.
//!
//! Thin adapter over [`crate::handlers::shared::proxy_request`]. All
//! per-provider behavior lives in [`AnthropicContract`].

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use tokenova_domain::{Provider, TokenUsage};

use crate::handlers::shared::{
    append_capped, extract_model, proxy_request, ProviderContract, PROMPT_CAP_BYTES,
};
use crate::state::AppState;
use crate::streaming::{AnthropicStreamParser, StreamingUsageParser};
use crate::usage;

/// `POST /v1/messages` — streaming-aware proxy.
pub async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    let url = format!(
        "{}/v1/messages",
        state.anthropic_upstream.trim_end_matches('/')
    );
    let request_model = extract_model(&body);
    proxy_request::<AnthropicContract>(state, headers, body, url, request_model).await
}

pub(crate) struct AnthropicContract;

impl ProviderContract for AnthropicContract {
    const PROVIDER: Provider = Provider::Anthropic;

    fn extract_prompt(body: &[u8]) -> String {
        extract_anthropic_prompt(body)
    }

    fn parse_buffered_usage(body: &[u8]) -> TokenUsage {
        usage::parse_anthropic(body)
    }

    fn streaming_parser() -> Box<dyn StreamingUsageParser> {
        Box::new(AnthropicStreamParser::new())
    }
    // `prepare_streaming_body` uses the trait default — Anthropic emits
    // usage unconditionally on streaming responses, no injection needed.
}

/// Pull the user prompt out of an Anthropic messages request. Includes the
/// top-level `system` field and every `user` message's `content`.
fn extract_anthropic_prompt(body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return String::new();
    };
    let mut out = String::new();

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
