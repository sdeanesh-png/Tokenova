//! Integration tests for Anthropic streaming token accounting (Session 2 Slice 1).

use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use axum::serve;
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;
use tokenova_domain::LogRecord;
use tokenova_proxy::observability::test_sink;
use tokenova_proxy::{build_router, AppState};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Serialize tests in this binary — the log sink is process-global. We
/// use a `tokio::sync::Mutex` so the guard can be held across `.await`.
async fn serial_guard() -> tokio::sync::OwnedMutexGuard<()> {
    static LOCK: OnceLock<Arc<tokio::sync::Mutex<()>>> = OnceLock::new();
    let lock = LOCK
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone();
    lock.lock_owned().await
}

async fn spawn_proxy(anthropic_upstream: String) -> SocketAddr {
    let state = AppState::for_tests("http://unused".into(), anthropic_upstream).unwrap();
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

#[derive(Clone, Default)]
struct Sink {
    records: Arc<Mutex<Vec<LogRecord>>>,
}

impl Sink {
    fn install(&self) {
        let records = self.records.clone();
        test_sink::set_test_sink(move |rec| {
            records.lock().unwrap().push(rec.clone());
        });
    }

    fn take(&self) -> Vec<LogRecord> {
        let mut guard = self.records.lock().unwrap();
        std::mem::take(&mut *guard)
    }
}

fn anthropic_sse_full() -> String {
    concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude-3-5-sonnet-latest\",\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":12}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    )
    .to_string()
}

#[tokio::test]
async fn anthropic_messages_streaming_accounts_tokens() {
    let _guard = serial_guard().await;
    test_sink::clear_test_sink();
    let sink = Sink::default();
    sink.install();

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(anthropic_sse_full()),
        )
        .mount(&mock)
        .await;

    let addr = spawn_proxy(mock.uri()).await;
    let client = Client::new();
    let req_body = json!({
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 64,
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .header("x-api-key", "sk-ant-test")
        .header("anthropic-version", "2023-06-01")
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;
    let records = sink.take();
    test_sink::clear_test_sink();
    assert_eq!(records.len(), 1);
    let rec = &records[0];
    assert!(rec.streamed);
    assert_eq!(rec.usage.prompt_tokens, 25);
    assert_eq!(rec.usage.completion_tokens, 12);
    assert!(!rec.stream_truncated);
    assert!(!rec.stream_error);
    assert_eq!(rec.model, "claude-3-5-sonnet-latest");
}

#[tokio::test]
async fn anthropic_messages_streaming_partial_frame_boundary() {
    let _guard = serial_guard().await;
    test_sink::clear_test_sink();
    let sink = Sink::default();
    sink.install();

    // Pad the first text delta with filler so the final message_delta frame
    // crosses a TCP chunk boundary under normal network conditions.
    let filler: String = "q".repeat(8000);
    let body = format!(
        concat!(
            "event: message_start\n",
            "data: {{\"type\":\"message_start\",\"message\":{{\"id\":\"m1\",\"model\":\"claude-3-5-sonnet-latest\",\"usage\":{{\"input_tokens\":25,\"output_tokens\":1}}}}}}\n\n",
            "event: content_block_delta\n",
            "data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{}\"}}}}\n\n",
            "event: message_delta\n",
            "data: {{\"type\":\"message_delta\",\"delta\":{{\"stop_reason\":\"end_turn\"}},\"usage\":{{\"output_tokens\":12}}}}\n\n",
            "event: message_stop\n",
            "data: {{\"type\":\"message_stop\"}}\n\n",
        ),
        filler,
    );

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&mock)
        .await;

    let addr = spawn_proxy(mock.uri()).await;
    let client = Client::new();
    let req_body = json!({
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 64,
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    let mut stream = resp.bytes_stream();
    let mut total = 0usize;
    while let Some(item) = stream.next().await {
        total += item.unwrap().len();
    }
    assert!(total > 8000);

    tokio::time::sleep(Duration::from_millis(50)).await;
    let records = sink.take();
    test_sink::clear_test_sink();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].usage.prompt_tokens, 25);
    assert_eq!(records[0].usage.completion_tokens, 12);
}

#[tokio::test]
async fn anthropic_messages_streaming_no_message_stop() {
    let _guard = serial_guard().await;
    test_sink::clear_test_sink();
    let sink = Sink::default();
    sink.install();

    // Stream ends after message_delta, no message_stop.
    let body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"claude-3-5-sonnet-latest\",\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":12}}\n\n",
    );

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .mount(&mock)
        .await;

    let addr = spawn_proxy(mock.uri()).await;
    let client = Client::new();
    let req_body = json!({
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 64,
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = client
        .post(format!("http://{addr}/v1/messages"))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    let _ = resp.bytes().await.unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;
    let records = sink.take();
    test_sink::clear_test_sink();
    assert_eq!(records.len(), 1);
    let rec = &records[0];
    assert!(rec.streamed);
    assert_eq!(rec.usage.prompt_tokens, 25);
    assert_eq!(rec.usage.completion_tokens, 12);
    // No message_stop seen → truncated.
    assert!(rec.stream_truncated);
}
