//! Shared domain types for the Tokenova platform.
//!
//! These types cross crate boundaries: the proxy produces them, the classifier
//! consumes them (for the `IntentCategory`), and downstream Session 2 work
//! (TimescaleDB persistence, dashboard API) will consume the `LogRecord`.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

/// Attribution tags extracted from `x-tokenova-*` request headers.
///
/// Required by PRD §5.1.1 ("Attribution tagging: user, team, department,
/// project, application, cost center, environment"). Missing headers
/// serialize as `null`; headers present with an empty value serialize as
/// `""`. This distinction matters for billing analytics — the drop-in
/// SDK promise means we cannot reject requests that lack tags, but we
/// must not lose the signal that a caller didn't send one either
/// (QA R1 Medium #6).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AttributionTags {
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub team: Option<String>,
    #[serde(default)]
    pub department: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub application: Option<String>,
    #[serde(default)]
    pub cost_center: Option<String>,
    #[serde(default)]
    pub environment: Option<String>,
}

/// Which upstream provider handled the call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    OpenAi,
    Anthropic,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::OpenAi => "openai",
            Provider::Anthropic => "anthropic",
        }
    }
}

/// Token counts from the provider response.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn new(prompt: u32, completion: u32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            // Saturating sum so a clamp-to-u32::MAX on one side (see
            // `crate::usage::clamp_u64_to_u32`) doesn't panic on overflow
            // when computing the total.
            total_tokens: prompt.saturating_add(completion),
        }
    }
}

/// The 10 seed intent categories for Semantic Token Clustering v1 (PRD §5.1.1).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum IntentCategory {
    CodeGeneration,
    Summarization,
    DataExtraction,
    QuestionAnswering,
    DocumentReview,
    Translation,
    CreativeWriting,
    ConversationalChat,
    ReasoningPlanning,
    Other,
}

impl IntentCategory {
    pub const ALL: [IntentCategory; 10] = [
        IntentCategory::CodeGeneration,
        IntentCategory::Summarization,
        IntentCategory::DataExtraction,
        IntentCategory::QuestionAnswering,
        IntentCategory::DocumentReview,
        IntentCategory::Translation,
        IntentCategory::CreativeWriting,
        IntentCategory::ConversationalChat,
        IntentCategory::ReasoningPlanning,
        IntentCategory::Other,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            IntentCategory::CodeGeneration => "code_generation",
            IntentCategory::Summarization => "summarization",
            IntentCategory::DataExtraction => "data_extraction",
            IntentCategory::QuestionAnswering => "question_answering",
            IntentCategory::DocumentReview => "document_review",
            IntentCategory::Translation => "translation",
            IntentCategory::CreativeWriting => "creative_writing",
            IntentCategory::ConversationalChat => "conversational_chat",
            IntentCategory::ReasoningPlanning => "reasoning_planning",
            IntentCategory::Other => "other",
        }
    }
}

/// One structured log record per proxied request.
///
/// This is what persistence in Session 2 will ingest. Emitted as JSON to
/// stdout via `tracing_subscriber::fmt::json()` for tonight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRecord {
    pub request_id: Uuid,
    #[serde(with = "time::serde::rfc3339")]
    pub received_at: OffsetDateTime,
    pub provider: Provider,
    pub model: String,
    pub attribution: AttributionTags,
    pub usage: TokenUsage,
    pub cost_usd: f64,
    pub intent: IntentCategory,
    pub latency_added_ms: f64,
    pub upstream_status: u16,
    pub streamed: bool,
    /// True when a streaming response ended without the provider's canonical
    /// completion marker (OpenAI: no `[DONE]`; Anthropic: no `message_stop`).
    /// Always false for non-streaming requests.
    #[serde(default)]
    pub stream_truncated: bool,
    /// True when the upstream byte stream yielded an error mid-flight.
    /// Always false for non-streaming requests.
    #[serde(default)]
    pub stream_error: bool,
    /// Total wall time from proxy receiving the request to the upstream
    /// stream closing. `None` for non-streaming requests.
    #[serde(default)]
    pub stream_duration_ms: Option<f64>,
}
