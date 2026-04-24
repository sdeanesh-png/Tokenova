//! Integration test: wiremock stands in for OpenAI. We bind the proxy to
//! an ephemeral port, send a chat-completions request through it, and
//! assert the upstream receives the identical body + auth headers while
//! the client receives the identical response body.

use std::net::SocketAddr;

use axum::serve;
use reqwest::Client;
use serde_json::json;
use tokenova_proxy::{build_router, AppState};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn spawn_proxy(openai_upstream: String) -> SocketAddr {
    let state = AppState::for_tests(openai_upstream, "http://unused".into(), None).unwrap();
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        serve(listener, app).await.unwrap();
    });
    addr
}

#[tokio::test]
async fn openai_chat_completions_passthrough() {
    let mock = MockServer::start().await;
    let canned = json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Paris"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    });

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&canned))
        .mount(&mock)
        .await;

    let addr = spawn_proxy(mock.uri()).await;

    let client = Client::new();
    let req_body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is the capital of France?"}]
    });
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .bearer_auth("sk-test")
        .header("x-tokenova-user", "steve")
        .header("x-tokenova-team", "founders")
        .header("x-tokenova-project", "session-1")
        .json(&req_body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let got: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(got, canned);

    // Upstream saw the forwarded request.
    let received = mock.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let forwarded_body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(forwarded_body, req_body);
    // Attribution headers must NOT be forwarded upstream.
    assert!(received[0].headers.get("x-tokenova-user").is_none());
    assert!(received[0].headers.get("x-tokenova-team").is_none());
    assert_eq!(
        received[0]
            .headers
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap(),
        "Bearer sk-test"
    );
}

#[tokio::test]
async fn openai_error_propagates_status() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": {"type": "rate_limit_exceeded", "message": "Too many"}
        })))
        .mount(&mock)
        .await;

    let addr = spawn_proxy(mock.uri()).await;
    let client = Client::new();
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({"model": "gpt-4o", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 429);
}
