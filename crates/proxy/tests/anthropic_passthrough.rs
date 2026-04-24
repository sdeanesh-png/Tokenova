//! Integration test: wiremock stands in for Anthropic's `/v1/messages`.

use std::net::SocketAddr;

use axum::serve;
use reqwest::Client;
use serde_json::json;
use tokenova_proxy::{build_router, AppState};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn spawn_proxy(anthropic_upstream: String) -> SocketAddr {
    let state = AppState::for_tests("http://unused".into(), anthropic_upstream, None).unwrap();
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
async fn anthropic_messages_passthrough() {
    let mock = MockServer::start().await;
    let canned = json!({
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-latest",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 9, "output_tokens": 2}
    });

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-ant-test"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&canned))
        .mount(&mock)
        .await;

    let addr = spawn_proxy(mock.uri()).await;
    let client = Client::new();
    let req_body = json!({
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hey, how are you doing today?"}]
    });
    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .header("x-api-key", "sk-ant-test")
        .header("anthropic-version", "2023-06-01")
        .header("x-tokenova-user", "steve")
        .json(&req_body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let got: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(got, canned);

    let received = mock.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let forwarded_body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(forwarded_body, req_body);
    assert!(received[0].headers.get("x-tokenova-user").is_none());
    assert_eq!(
        received[0]
            .headers
            .get("x-api-key")
            .unwrap()
            .to_str()
            .unwrap(),
        "sk-ant-test"
    );
}
