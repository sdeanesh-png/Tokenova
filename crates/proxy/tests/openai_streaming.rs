//! Integration tests for OpenAI streaming token accounting (Session 2 Slice 1).
//!
//! wiremock stands in for the upstream. We bind the proxy to an ephemeral port,
//! send `stream: true` requests, and verify:
//!   * wiremock receives an `include_usage=true` body (unless the client opted out)
//!   * the client receives the SSE bytes
//!   * a `LogRecord` is emitted exactly once with the reconstructed usage.

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

/// Serialize tests in this binary — the log sink is process-global, so
/// concurrent tests would cross-contaminate each other's assertions. We use
/// a `tokio::sync::Mutex` because the guard must be held across `.await`.
async fn serial_guard() -> tokio::sync::OwnedMutexGuard<()> {
    static LOCK: OnceLock<Arc<tokio::sync::Mutex<()>>> = OnceLock::new();
    let lock = LOCK
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone();
    lock.lock_owned().await
}

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

/// Shared sink collector. Each test installs this and later reads the captured
/// records. Clearing is the test's responsibility.
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

/// Canonical OpenAI SSE body for a happy-path streaming completion that
/// includes a usage frame (because we injected `include_usage=true`).
fn openai_sse_with_usage() -> String {
    concat!(
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}],\"usage\":null}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}],\"usage\":null}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2,\"total_tokens\":13}}\n\n",
        "data: [DONE]\n\n",
    )
    .to_string()
}

#[tokio::test]
async fn openai_chat_streaming_injects_include_usage_and_accounts_tokens() {
    let _guard = serial_guard().await;
    test_sink::clear_test_sink();
    let sink = Sink::default();
    sink.install();

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(openai_sse_with_usage()),
        )
        .mount(&mock)
        .await;

    let addr = spawn_proxy(mock.uri()).await;
    let client = Client::new();
    let req_body = json!({
        "model": "gpt-4o-mini",
        "stream": true,
        "messages": [{"role": "user", "content": "Say hi."}]
    });
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .bearer_auth("sk-test")
        .header("x-tokenova-user", "steve")
        .json(&req_body)
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    // Drain the body so the stream adapter sees clean EOF before we inspect sink.
    let body_bytes = resp.bytes().await.unwrap();
    assert!(body_bytes.windows(6).any(|w| w == b"[DONE]"));

    // Wiremock should have received the injected body.
    let received = mock.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let forwarded: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(
        forwarded
            .get("stream_options")
            .and_then(|o| o.get("include_usage")),
        Some(&serde_json::Value::Bool(true)),
        "expected include_usage=true to be injected"
    );

    // Give the accumulator a beat to emit — EOF → emit(Clean) is synchronous
    // in the stream adapter but happens on the axum task.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let records = sink.take();
    test_sink::clear_test_sink();
    assert_eq!(records.len(), 1, "exactly one LogRecord per request");
    let rec = &records[0];
    assert!(rec.streamed);
    assert_eq!(rec.usage.prompt_tokens, 11);
    assert_eq!(rec.usage.completion_tokens, 2);
    assert_eq!(rec.usage.total_tokens, 13);
    assert!(rec.cost_usd > 0.0);
    assert!(!rec.stream_truncated);
    assert!(!rec.stream_error);
    assert!(rec.stream_duration_ms.is_some());
    assert_eq!(rec.model, "gpt-4o-mini");
}

#[tokio::test]
async fn openai_chat_streaming_respects_explicit_include_usage_false() {
    let _guard = serial_guard().await;
    test_sink::clear_test_sink();
    let sink = Sink::default();
    sink.install();

    // Fixture has NO usage frame because client opted out.
    let body = concat!(
        "data: {\"id\":\"x\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
        "data: [DONE]\n\n",
    );

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
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
        "model": "gpt-4o-mini",
        "stream": true,
        "stream_options": {"include_usage": false},
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .bearer_auth("sk-test")
        .json(&req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let _ = resp.bytes().await.unwrap();

    // The body we sent upstream should still have include_usage=false.
    let received = mock.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    let forwarded: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(
        forwarded
            .get("stream_options")
            .and_then(|o| o.get("include_usage")),
        Some(&serde_json::Value::Bool(false)),
        "we must not flip an explicit false"
    );

    tokio::time::sleep(Duration::from_millis(50)).await;
    let records = sink.take();
    test_sink::clear_test_sink();
    assert_eq!(records.len(), 1);
    let rec = &records[0];
    assert!(rec.streamed);
    assert_eq!(rec.usage.prompt_tokens, 0);
    assert_eq!(rec.usage.completion_tokens, 0);
}

/// Chunk the SSE body mid-frame by configuring wiremock with a small body but
/// sending the bytes in pieces using the raw socket. wiremock-rs delivers the
/// body as one hyper write, so the test exercises the framer at the boundary
/// via our own parser. As a lower-impact check, we verify the framer works
/// for *any* single-byte split of the stream — this is the unit test in
/// `streaming::tests::sse_framer_handles_concat_split_anywhere`.
///
/// At the integration level we chunk by feeding wiremock a long body and
/// relying on network MTU-induced chunking; we assert the parsed usage
/// matches regardless of how many chunks reqwest emits.
#[tokio::test]
async fn openai_chat_streaming_partial_frame_boundary() {
    let _guard = serial_guard().await;
    test_sink::clear_test_sink();
    let sink = Sink::default();
    sink.install();

    // Pad the pre-usage deltas with a large content block to force the
    // terminal usage frame across a TCP segment boundary.
    let filler: String = "x".repeat(8000);
    let body = format!(
        concat!(
            "data: {{\"id\":\"x\",\"model\":\"gpt-4o-mini\",\"choices\":[{{\"delta\":{{\"content\":\"{}\"}}}}],\"usage\":null}}\n\n",
            "data: {{\"id\":\"x\",\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{{\"prompt_tokens\":11,\"completion_tokens\":2,\"total_tokens\":13}}}}\n\n",
            "data: [DONE]\n\n",
        ),
        filler,
    );

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
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
        "model": "gpt-4o-mini",
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&req_body)
        .send()
        .await
        .unwrap();

    // Drain body chunk by chunk to ensure the proxy is actually streaming.
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
    assert_eq!(records[0].usage.prompt_tokens, 11);
    assert_eq!(records[0].usage.completion_tokens, 2);
}

#[tokio::test]
async fn openai_chat_streaming_upstream_mid_stream_error() {
    let _guard = serial_guard().await;
    test_sink::clear_test_sink();
    let sink = Sink::default();
    sink.install();

    // Two frames, no [DONE], no usage — simulate truncation.
    let body = concat!(
        "data: {\"id\":\"x\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}],\"usage\":null}\n\n",
        "data: {\"id\":\"x\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"!\"}}],\"usage\":null}\n\n",
    );

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
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
        "model": "gpt-4o-mini",
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    let _ = resp.bytes().await.unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;
    let records = sink.take();
    test_sink::clear_test_sink();
    assert_eq!(records.len(), 1, "exactly one LogRecord");
    let rec = &records[0];
    assert!(rec.streamed);
    // No [DONE] + no usage frame → stream_truncated true, usage zero.
    assert!(rec.stream_truncated);
    assert_eq!(rec.usage.prompt_tokens, 0);
    assert_eq!(rec.usage.completion_tokens, 0);
}

#[tokio::test]
async fn openai_chat_streaming_client_disconnect() {
    let _guard = serial_guard().await;
    test_sink::clear_test_sink();
    let sink = Sink::default();
    sink.install();

    // Big body so the client can disconnect mid-stream. Use a slow-delivered
    // response so the client has a chance to drop before the stream ends.
    let body = openai_sse_with_usage().repeat(50);

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body)
                .set_delay(Duration::from_millis(100)),
        )
        .mount(&mock)
        .await;

    let addr = spawn_proxy(mock.uri()).await;
    let client = Client::new();
    let req_body = json!({
        "model": "gpt-4o-mini",
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&req_body)
        .send()
        .await
        .unwrap();
    // Start draining the stream, then drop the response after the first
    // chunk to simulate client disconnect.
    let mut stream = resp.bytes_stream();
    let _ = stream.next().await;
    drop(stream);

    // Give the accumulator time to be dropped and emit via Drop.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let records = sink.take();
    test_sink::clear_test_sink();
    // Exactly one record — Drop is idempotent via AtomicBool.
    assert_eq!(records.len(), 1, "exactly one LogRecord on disconnect");
    let rec = &records[0];
    assert!(rec.streamed);
}
