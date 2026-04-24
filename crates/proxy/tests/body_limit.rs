//! Integration test for QA Blocker #1: enforce a 256 KB request body limit.
//!
//! The proxy is designed for small prompts; an unbuffered 2 MB body (the
//! axum default) is a DoS surface. A 512 KB POST must be rejected with
//! 413 Payload Too Large before any upstream work is attempted.

use std::net::SocketAddr;

use axum::serve;
use reqwest::Client;
use tokenova_proxy::{build_router, AppState};

async fn spawn_proxy() -> SocketAddr {
    let state = AppState::for_tests("http://unused".into(), "http://unused".into()).unwrap();
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
async fn oversized_openai_body_is_rejected_413() {
    let addr = spawn_proxy().await;
    let client = Client::new();

    // 512 KB of 'a' wrapped as a JSON object so the handler reads it as a
    // Bytes extractor. We expect the DefaultBodyLimit layer on the router
    // to short-circuit before the handler runs.
    let big: String = "a".repeat(512 * 1024);
    let req_body = serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": big}]
    });
    let serialized = serde_json::to_vec(&req_body).unwrap();
    assert!(serialized.len() > 256 * 1024);

    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body(serialized)
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status().as_u16(),
        413,
        "expected 413 Payload Too Large"
    );
}

#[tokio::test]
async fn oversized_anthropic_body_is_rejected_413() {
    let addr = spawn_proxy().await;
    let client = Client::new();

    let big: String = "b".repeat(512 * 1024);
    let req_body = serde_json::json!({
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": big}]
    });
    let serialized = serde_json::to_vec(&req_body).unwrap();

    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .header("content-type", "application/json")
        .body(serialized)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 413);
}
