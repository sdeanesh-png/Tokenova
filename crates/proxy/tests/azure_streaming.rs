//! Azure OpenAI streaming integration test.
//!
//! Requires the `test-utils` Cargo feature so the `test_sink` module is
//! compiled. Verifies that Azure's streaming path reuses OpenAI's SSE
//! parser correctly and the emitted `LogRecord` has `provider =
//! azure_openai`, `streamed = true`, and non-zero usage.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::serve;
use reqwest::Client;
use tokenova_domain::{LogRecord, Provider};
use tokenova_proxy::observability::test_sink;
use tokenova_proxy::{build_router, AppState};
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn spawn_proxy(azure_upstream: String) -> SocketAddr {
    let state = AppState::for_tests(
        "http://unused".into(),
        "http://unused".into(),
        Some(azure_upstream),
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

// Shared mutex so tests don't race on the global sink.
fn test_mutex() -> &'static tokio::sync::Mutex<()> {
    static M: std::sync::OnceLock<tokio::sync::Mutex<()>> = std::sync::OnceLock::new();
    M.get_or_init(|| tokio::sync::Mutex::new(()))
}

#[tokio::test]
async fn azure_chat_streaming_accounts_tokens() {
    let _guard = test_mutex().lock().await;

    let captured: Arc<Mutex<Vec<LogRecord>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_cl = captured.clone();
    test_sink::set_test_sink(move |r| captured_cl.lock().unwrap().push(r.clone()));

    let mock = MockServer::start().await;
    let sse = concat!(
        r#"data: {"choices":[{"delta":{"content":"Par"}}],"model":"gpt-4o"}"#,
        "\n\n",
        r#"data: {"choices":[{"delta":{"content":"is"}}]}"#,
        "\n\n",
        r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        "\n\n",
        r#"data: {"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/openai/deployments/prod-chat/chat/completions"))
        .and(query_param("api-version", "2024-08-06-preview"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&mock)
        .await;

    let addr = spawn_proxy(mock.uri()).await;
    let client = Client::new();

    let resp = client
        .post(format!(
            "http://{addr}/openai/deployments/prod-chat/chat/completions?api-version=2024-08-06-preview"
        ))
        .header("api-key", "azure-key")
        .header("x-tokenova-user", "steve")
        .json(&serde_json::json!({
            "stream": true,
            "messages": [{"role": "user", "content": "Count."}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body = resp.text().await.unwrap();
    assert_eq!(body, sse, "client received SSE bytes unchanged");

    // Give the stream adapter a beat to emit the LogRecord.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let records = captured.lock().unwrap().clone();
    assert_eq!(records.len(), 1, "exactly one LogRecord emitted");
    let rec = &records[0];
    assert_eq!(rec.provider, Provider::AzureOpenAi);
    assert!(rec.streamed);
    assert_eq!(rec.usage.prompt_tokens, 8);
    assert_eq!(rec.usage.completion_tokens, 2);
    assert!(rec.cost_usd > 0.0, "cost must be non-zero");
    // Model from response body wins over the deployment name.
    assert_eq!(rec.model, "gpt-4o");
    assert_eq!(rec.attribution.user.as_deref(), Some("steve"));

    test_sink::clear_test_sink();
}
