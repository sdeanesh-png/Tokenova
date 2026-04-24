//! Integration tests for the Azure OpenAI handlers.
//!
//! Runs on the default build — no `test-utils` feature needed.

use std::net::SocketAddr;

use axum::serve;
use reqwest::Client;
use serde_json::json;
use tokenova_proxy::{build_router, AppState};
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn spawn_proxy(azure_upstream: Option<String>) -> SocketAddr {
    let state = AppState::for_tests(
        "http://unused".into(),
        "http://unused".into(),
        azure_upstream,
    )
    .unwrap();
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
async fn azure_chat_completions_forwards_path_query_and_body() {
    let mock = MockServer::start().await;
    let canned = json!({
        "id": "chatcmpl-azure-1",
        "object": "chat.completion",
        "model": "gpt-4o-2024-08-06",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Paris"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
    });

    Mock::given(method("POST"))
        .and(path("/openai/deployments/prod-chat/chat/completions"))
        .and(query_param("api-version", "2024-08-06-preview"))
        .and(header("api-key", "azure-key-123"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&canned))
        .mount(&mock)
        .await;

    let addr = spawn_proxy(Some(mock.uri())).await;

    let client = Client::new();
    let req_body = json!({
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "max_tokens": 16
    });
    let resp = client
        .post(format!(
            "http://{addr}/openai/deployments/prod-chat/chat/completions?api-version=2024-08-06-preview"
        ))
        .header("api-key", "azure-key-123")
        .header("x-tokenova-user", "steve")
        .header("x-tokenova-team", "enterprise")
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
    // api-key auth survives the forward.
    assert_eq!(
        received[0]
            .headers
            .get("api-key")
            .unwrap()
            .to_str()
            .unwrap(),
        "azure-key-123"
    );
    // Attribution headers are NOT forwarded to the upstream.
    assert!(received[0].headers.get("x-tokenova-user").is_none());
}

#[tokio::test]
async fn azure_routes_return_503_when_upstream_not_configured() {
    let addr = spawn_proxy(None).await;
    let client = Client::new();

    let resp = client
        .post(format!(
            "http://{addr}/openai/deployments/any/chat/completions?api-version=2024-08-06-preview"
        ))
        .json(&json!({"messages": []}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn azure_embeddings_passthrough() {
    let mock = MockServer::start().await;
    let canned = json!({
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3]
        }],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 3, "total_tokens": 3}
    });

    Mock::given(method("POST"))
        .and(path("/openai/deployments/embed-prod/embeddings"))
        .and(query_param("api-version", "2024-08-06-preview"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&canned))
        .mount(&mock)
        .await;

    let addr = spawn_proxy(Some(mock.uri())).await;
    let client = Client::new();
    let req_body = json!({"input": "Hello, Tokenova"});
    let resp = client
        .post(format!(
            "http://{addr}/openai/deployments/embed-prod/embeddings?api-version=2024-08-06-preview"
        ))
        .header("api-key", "ak")
        .json(&req_body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let got: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(got, canned);
}
