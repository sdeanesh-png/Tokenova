//! OpenAI route handlers.
//!
//! Thin adapters over [`crate::handlers::shared::proxy_request`]. All
//! per-provider behavior lives in [`OpenAiContract`].

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use tokenova_domain::{Provider, TokenUsage};

use crate::handlers::shared::{
    append_capped, proxy_request, BodyMutation, ProviderContract, PROMPT_CAP_BYTES,
};
use crate::state::AppState;
use crate::streaming::{OpenAiStreamParser, StreamingUsageParser};
use crate::usage;

/// `POST /v1/chat/completions` — streaming-aware proxy.
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    let url = format!(
        "{}/v1/chat/completions",
        state.openai_upstream.trim_end_matches('/')
    );
    proxy_request::<OpenAiContract>(state, headers, body, url).await
}

/// `POST /v1/embeddings` — buffered proxy. Embeddings has no SSE mode on
/// OpenAI, so the streaming path never fires.
pub async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, (StatusCode, String)> {
    let url = format!(
        "{}/v1/embeddings",
        state.openai_upstream.trim_end_matches('/')
    );
    proxy_request::<OpenAiContract>(state, headers, body, url).await
}

pub(crate) struct OpenAiContract;

impl ProviderContract for OpenAiContract {
    const PROVIDER: Provider = Provider::OpenAi;

    fn extract_prompt(body: &[u8]) -> String {
        extract_openai_prompt(body)
    }

    fn parse_buffered_usage(body: &[u8]) -> TokenUsage {
        usage::parse_openai(body)
    }

    fn streaming_parser() -> Box<dyn StreamingUsageParser> {
        Box::new(OpenAiStreamParser::new())
    }

    fn prepare_streaming_body(body: Bytes) -> (Bytes, BodyMutation) {
        maybe_inject_stream_options(body)
    }
}

/// If the request is a streaming request, ensure `stream_options.include_usage`
/// is `true` so OpenAI emits the terminal usage frame we need for billing.
/// Never flips an explicit `false` — the client's opt-out is preserved.
fn maybe_inject_stream_options(body: Bytes) -> (Bytes, BodyMutation) {
    let Ok(mut v) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return (body, BodyMutation::None);
    };
    let streamed = v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    if !streamed {
        return (body, BodyMutation::None);
    }

    let existing = v
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(|i| i.as_bool());

    match existing {
        Some(true) => (body, BodyMutation::AlreadyHadIncludeUsage),
        Some(false) => (body, BodyMutation::ClientOptedOut),
        None => {
            let obj = v.as_object_mut().expect("top-level must be object");
            let so = obj
                .entry("stream_options".to_string())
                .or_insert_with(|| serde_json::json!({}));
            if let Some(so_obj) = so.as_object_mut() {
                so_obj.insert("include_usage".into(), serde_json::Value::Bool(true));
            }
            let new_bytes =
                serde_json::to_vec(&v).expect("re-serializing a parsed Value must succeed");
            (Bytes::from(new_bytes), BodyMutation::InjectedIncludeUsage)
        }
    }
}

/// Pull the user-visible prompt text out of an OpenAI chat-completions
/// request for classification. Concatenates `content` of every `user` and
/// `system` message, capped incrementally at `PROMPT_CAP_BYTES`.
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
    out.truncate(PROMPT_CAP_BYTES);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_openai_prompt_bounded_under_many_messages() {
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
        assert_eq!(inj, BodyMutation::InjectedIncludeUsage);
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
        assert_eq!(inj, BodyMutation::ClientOptedOut);
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
        assert_eq!(inj, BodyMutation::AlreadyHadIncludeUsage);
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
        assert_eq!(inj, BodyMutation::None);
        assert_eq!(new_body, body);
    }
}
