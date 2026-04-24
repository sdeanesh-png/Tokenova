//! Streaming token accounting.
//!
//! Wraps a `reqwest::Response::bytes_stream()` with a per-provider SSE parser
//! that reconstructs the final `usage` block without buffering the stream.
//! Response bytes flow to the client unchanged; the parser observes each
//! chunk in the same task that forwards it (no mpsc tee, no cloning of the
//! underlying byte buffers beyond refcounted `Bytes` handles).
//!
//! Architect's plan §3.3 / §4.1 — inspect-in-place over `bytes_stream()`.

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use bytes::{Bytes, BytesMut};
use futures::Stream;
use serde_json::Value;
use time::OffsetDateTime;
use tokenova_domain::{AttributionTags, IntentCategory, LogRecord, Provider, TokenUsage};
use uuid::Uuid;

use crate::observability::emit_log_record;
use crate::pricing::PricingTable;
use crate::usage::clamp_u64_to_u32;

/// One complete SSE frame, post-split. `event` may be absent if the frame
/// only carried `data:` lines (OpenAI streams that way).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseFrame {
    pub event: Option<String>,
    pub data: String,
}

/// Accumulates incoming byte chunks and emits complete SSE frames.
///
/// Frames are delimited by `\n\n` per the SSE spec. Within a frame, each
/// line is either `event: <name>`, `data: <payload>`, a comment starting
/// with `:`, or ignored. Multi-line `data:` values are joined with `\n`.
pub struct SseFramer {
    buf: BytesMut,
}

impl Default for SseFramer {
    fn default() -> Self {
        Self::new()
    }
}

impl SseFramer {
    pub fn new() -> Self {
        Self {
            buf: BytesMut::with_capacity(4096),
        }
    }

    /// Append the chunk to the internal buffer and return every complete
    /// frame found. Unterminated trailing bytes stay buffered for the next
    /// call.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseFrame> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();

        loop {
            // Frames are `\n\n` separated. Accept `\r\n\r\n` too for robustness.
            let end_pos = find_frame_end(&self.buf);
            let Some((pos, delim_len)) = end_pos else {
                break;
            };
            let frame_bytes = self.buf.split_to(pos).freeze();
            // Discard the delimiter.
            let _ = self.buf.split_to(delim_len);

            if let Some(frame) = parse_frame(&frame_bytes) {
                out.push(frame);
            }
        }

        out
    }
}

/// Search for the first `\n\n` or `\r\n\r\n` delimiter in the buffer.
/// Returns `(position_of_first_newline_before_delim, delim_length)`.
fn find_frame_end(buf: &[u8]) -> Option<(usize, usize)> {
    // Scan for `\n\n` (len 2) or `\r\n\r\n` (len 4).
    let mut i = 0;
    while i + 1 < buf.len() {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some((i, 2));
        }
        if i + 3 < buf.len()
            && buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some((i, 4));
        }
        i += 1;
    }
    None
}

fn parse_frame(bytes: &[u8]) -> Option<SseFrame> {
    // Split on `\n` (accepting optional `\r`) and pull out `event:` + `data:` lines.
    let text = std::str::from_utf8(bytes).ok()?;
    let mut event: Option<String> = None;
    let mut data_parts: Vec<&str> = Vec::new();
    let mut any_data = false;

    for raw_line in text.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() {
            continue;
        }
        if line.starts_with(':') {
            // Comment line.
            continue;
        }
        if let Some(rest) = line.strip_prefix("event:") {
            event = Some(rest.trim_start().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_parts.push(rest.strip_prefix(' ').unwrap_or(rest));
            any_data = true;
        }
        // Other field names ("id:", "retry:") are ignored — we don't need them.
    }

    if !any_data && event.is_none() {
        return None;
    }
    let data = data_parts.join("\n");
    Some(SseFrame { event, data })
}

/// Per-provider parser that extracts `TokenUsage` (and optionally the response
/// model) from the observed SSE frames.
pub trait StreamingUsageParser: Send + 'static {
    fn on_frame(&mut self, frame: &SseFrame);
    fn finalize(&self) -> TokenUsage;
    fn response_model(&self) -> Option<String>;
    /// True if the stream saw its provider-specific terminal marker
    /// (OpenAI `data: [DONE]`, Anthropic `event: message_stop`).
    fn saw_terminal_marker(&self) -> bool;
}

// --- OpenAI ------------------------------------------------------------------

/// OpenAI chat-completions streaming parser.
///
/// OpenAI streams `data: {delta JSON}` frames and a terminal `data: [DONE]`
/// sentinel. When `stream_options.include_usage=true` is set (we do this in
/// `handlers::openai`), the penultimate frame carries a top-level `usage`
/// object; earlier frames have `usage: null`. We extract the last non-null
/// usage we see.
pub struct OpenAiStreamParser {
    prompt_tokens: u64,
    completion_tokens: u64,
    model: Option<String>,
    saw_done: bool,
    saw_usage: bool,
}

impl Default for OpenAiStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiStreamParser {
    pub fn new() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            model: None,
            saw_done: false,
            saw_usage: false,
        }
    }
}

impl StreamingUsageParser for OpenAiStreamParser {
    fn on_frame(&mut self, frame: &SseFrame) {
        // Skip frames that aren't data-bearing.
        if frame.data.is_empty() {
            return;
        }
        let trimmed = frame.data.trim();
        if trimmed == "[DONE]" {
            self.saw_done = true;
            return;
        }
        let Ok(v) = serde_json::from_str::<Value>(trimmed) else {
            return;
        };

        if self.model.is_none() {
            if let Some(m) = v.get("model").and_then(|m| m.as_str()) {
                self.model = Some(m.to_string());
            }
        }

        if let Some(usage) = v.get("usage").and_then(|u| u.as_object()) {
            if let Some(p) = usage.get("prompt_tokens").and_then(|x| x.as_u64()) {
                self.prompt_tokens = p;
                self.saw_usage = true;
            }
            if let Some(c) = usage.get("completion_tokens").and_then(|x| x.as_u64()) {
                self.completion_tokens = c;
                self.saw_usage = true;
            }
        }
    }

    fn finalize(&self) -> TokenUsage {
        if !self.saw_usage {
            return TokenUsage::default();
        }
        let p = clamp_u64_to_u32(self.prompt_tokens, "prompt_tokens");
        let c = clamp_u64_to_u32(self.completion_tokens, "completion_tokens");
        TokenUsage::new(p, c)
    }

    fn response_model(&self) -> Option<String> {
        self.model.clone()
    }

    fn saw_terminal_marker(&self) -> bool {
        self.saw_done
    }
}

// --- Anthropic ---------------------------------------------------------------

/// Anthropic messages streaming parser.
///
/// State machine driven by `event: message_start` (carries `input_tokens`,
/// prefill `output_tokens`, and the model) and `event: message_delta`
/// (carries the **cumulative** `output_tokens` total, not a delta — we
/// overwrite on each delta event). `event: message_stop` is the canonical
/// completion marker.
pub struct AnthropicStreamParser {
    input_tokens: u64,
    output_tokens: u64,
    model: Option<String>,
    saw_message_stop: bool,
    saw_message_start: bool,
}

impl Default for AnthropicStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicStreamParser {
    pub fn new() -> Self {
        Self {
            input_tokens: 0,
            output_tokens: 0,
            model: None,
            saw_message_stop: false,
            saw_message_start: false,
        }
    }
}

impl StreamingUsageParser for AnthropicStreamParser {
    fn on_frame(&mut self, frame: &SseFrame) {
        let Some(event) = frame.event.as_deref() else {
            return;
        };
        match event {
            "message_start" => {
                if let Ok(v) = serde_json::from_str::<Value>(&frame.data) {
                    if let Some(msg) = v.get("message") {
                        if self.model.is_none() {
                            if let Some(m) = msg.get("model").and_then(|m| m.as_str()) {
                                self.model = Some(m.to_string());
                            }
                        }
                        if let Some(usage) = msg.get("usage").and_then(|u| u.as_object()) {
                            if let Some(i) = usage.get("input_tokens").and_then(|x| x.as_u64()) {
                                self.input_tokens = i;
                            }
                            // Fold cache_creation_input_tokens and
                            // cache_read_input_tokens into prompt_tokens if
                            // present (plan §2: "folded into prompt_tokens
                            // if present but no new fields").
                            if let Some(cc) = usage
                                .get("cache_creation_input_tokens")
                                .and_then(|x| x.as_u64())
                            {
                                self.input_tokens = self.input_tokens.saturating_add(cc);
                            }
                            if let Some(cr) = usage
                                .get("cache_read_input_tokens")
                                .and_then(|x| x.as_u64())
                            {
                                self.input_tokens = self.input_tokens.saturating_add(cr);
                            }
                            if let Some(o) = usage.get("output_tokens").and_then(|x| x.as_u64()) {
                                self.output_tokens = o;
                            }
                        }
                    }
                }
                self.saw_message_start = true;
            }
            "message_delta" => {
                if let Ok(v) = serde_json::from_str::<Value>(&frame.data) {
                    if let Some(usage) = v.get("usage").and_then(|u| u.as_object()) {
                        if let Some(o) = usage.get("output_tokens").and_then(|x| x.as_u64()) {
                            // Cumulative total: replace, do not add.
                            self.output_tokens = o;
                        }
                    }
                }
            }
            "message_stop" => {
                self.saw_message_stop = true;
            }
            _ => {
                // ping, content_block_start, content_block_delta,
                // content_block_stop — not relevant for usage.
            }
        }
    }

    fn finalize(&self) -> TokenUsage {
        if !self.saw_message_start && self.input_tokens == 0 && self.output_tokens == 0 {
            return TokenUsage::default();
        }
        let p = clamp_u64_to_u32(self.input_tokens, "input_tokens");
        let c = clamp_u64_to_u32(self.output_tokens, "output_tokens");
        TokenUsage::new(p, c)
    }

    fn response_model(&self) -> Option<String> {
        self.model.clone()
    }

    fn saw_terminal_marker(&self) -> bool {
        self.saw_message_stop
    }
}

// --- Accumulator + stream adapter -------------------------------------------

/// Why the accumulator emitted its `LogRecord`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitTrigger {
    /// Upstream stream returned `None` (clean EOF).
    Clean,
    /// Upstream stream yielded an error mid-flight.
    StreamError,
    /// Accumulator was dropped before the stream reached EOF (client
    /// disconnect or task cancellation).
    Drop,
}

/// Per-request streaming accumulator. Holds everything needed to build a
/// `LogRecord` at stream termination.
///
/// ## Exactly-one emission invariant
///
/// Every proxied streaming request produces exactly one `LogRecord`,
/// regardless of whether it completes cleanly, hits an upstream error,
/// or gets torn down by a client disconnect. Three code paths can trigger
/// emission:
///
/// 1. **Clean EOF** — [`ObservedStream::poll_next`] sees the upstream
///    stream return `Poll::Ready(None)` and calls `emit(EmitTrigger::Clean)`.
/// 2. **Stream error** — the upstream stream yields `Poll::Ready(Some(Err))`;
///    the adapter records the error and calls `emit(EmitTrigger::StreamError)`.
/// 3. **Drop** — the accumulator is dropped without having reached EOF
///    (typically because axum dropped the response body after a client
///    disconnect). The [`Drop`] impl calls `emit(EmitTrigger::Drop)`.
///
/// Idempotency is enforced by an `AtomicBool::compare_exchange` on
/// [`Self::emitted`]: whichever path wins the CAS emits; the losers are
/// no-ops. This is load-bearing for billing correctness — **never remove
/// the CAS** without replacing it with an equivalent single-winner guard.
/// Both the EOF path and the `Drop` path are legitimate "last" emitters;
/// neither can be assumed to run first, and under cancellation the `Drop`
/// can race a still-running `poll_next` completion on some runtimes.
pub struct StreamingAccumulator {
    request_id: Uuid,
    received_at: OffsetDateTime,
    started: Instant,
    provider: Provider,
    request_model: String,
    attribution: AttributionTags,
    intent: IntentCategory,
    pricing: Arc<PricingTable>,
    upstream_status: u16,
    latency_added_ms: f64,
    parser: Box<dyn StreamingUsageParser>,
    framer: SseFramer,
    emitted: AtomicBool,
    stream_error: AtomicBool,
}

impl StreamingAccumulator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: Uuid,
        received_at: OffsetDateTime,
        started: Instant,
        provider: Provider,
        request_model: String,
        attribution: AttributionTags,
        intent: IntentCategory,
        pricing: Arc<PricingTable>,
        upstream_status: u16,
        latency_added_ms: f64,
        parser: Box<dyn StreamingUsageParser>,
    ) -> Self {
        Self {
            request_id,
            received_at,
            started,
            provider,
            request_model,
            attribution,
            intent,
            pricing,
            upstream_status,
            latency_added_ms,
            parser,
            framer: SseFramer::new(),
            emitted: AtomicBool::new(false),
            stream_error: AtomicBool::new(false),
        }
    }

    fn observe_chunk(&mut self, chunk: &Bytes) {
        let frames = self.framer.push(chunk);
        for frame in frames.iter() {
            self.parser.on_frame(frame);
        }
    }

    fn mark_stream_error(&self) {
        self.stream_error.store(true, Ordering::SeqCst);
    }

    fn emit(&self, trigger: EmitTrigger) {
        if self
            .emitted
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }

        let usage = self.parser.finalize();
        let response_model = self
            .parser
            .response_model()
            .unwrap_or_else(|| self.request_model.clone());
        let cost_usd = self
            .pricing
            .rate(self.provider, &response_model)
            .cost_usd(&usage);
        let stream_duration_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        let stream_error =
            self.stream_error.load(Ordering::SeqCst) || trigger == EmitTrigger::StreamError;
        let stream_truncated = !self.parser.saw_terminal_marker() || trigger == EmitTrigger::Drop;

        let record = LogRecord {
            request_id: self.request_id,
            received_at: self.received_at,
            provider: self.provider,
            model: response_model,
            attribution: self.attribution.clone(),
            usage,
            cost_usd,
            intent: self.intent,
            latency_added_ms: self.latency_added_ms,
            upstream_status: self.upstream_status,
            streamed: true,
            stream_truncated,
            stream_error,
            stream_duration_ms: Some(stream_duration_ms),
        };

        emit_log_record(&record);
    }
}

impl Drop for StreamingAccumulator {
    fn drop(&mut self) {
        // If we haven't emitted yet (client disconnect before the stream
        // adapter could reach EOF), emit now with Drop trigger so
        // stream_truncated is set correctly.
        if !self.emitted.load(Ordering::SeqCst) {
            self.emit(EmitTrigger::Drop);
        }
    }
}

/// Box-typed error alias used on the streaming body; axum accepts any
/// `Into<BoxError>`.
pub type StreamError = Box<dyn std::error::Error + Send + Sync>;

/// A stream that forwards upstream byte chunks unchanged while feeding each
/// chunk through a shared [`StreamingAccumulator`] for usage reconstruction.
///
/// On clean EOF it calls `emit(Clean)`; on error it records the error and
/// calls `emit(StreamError)`. `Drop` on the accumulator covers client
/// disconnects.
pub struct ObservedStream<S> {
    inner: S,
    accumulator: Arc<Mutex<StreamingAccumulator>>,
    finished: bool,
}

impl<S> ObservedStream<S> {
    pub fn new(inner: S, accumulator: Arc<Mutex<StreamingAccumulator>>) -> Self {
        Self {
            inner,
            accumulator,
            finished: false,
        }
    }
}

impl<S, E> Stream for ObservedStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    type Item = Result<Bytes, StreamError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(None);
        }
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                this.finished = true;
                if let Ok(acc) = this.accumulator.lock() {
                    acc.emit(EmitTrigger::Clean);
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(chunk))) => {
                if let Ok(mut acc) = this.accumulator.lock() {
                    acc.observe_chunk(&chunk);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(err))) => {
                this.finished = true;
                if let Ok(acc) = this.accumulator.lock() {
                    acc.mark_stream_error();
                    acc.emit(EmitTrigger::StreamError);
                }
                Poll::Ready(Some(Err(Box::new(err))))
            }
        }
    }
}

/// Convenience constructor: wrap a `reqwest` byte stream (or any compatible
/// byte stream) with the accumulator. Returns a stream ready to feed into
/// `axum::body::Body::from_stream`.
pub fn wrap_response_stream<S, E>(
    inner: S,
    accumulator: Arc<Mutex<StreamingAccumulator>>,
) -> ObservedStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    ObservedStream::new(inner, accumulator)
}

// --- Unit tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn openai_sample() -> Vec<u8> {
        // Representative OpenAI SSE stream: a couple of content deltas, a
        // penultimate frame carrying usage (because include_usage=true), and
        // the terminal [DONE].
        let s = concat!(
            "data: {\"id\":\"x\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}],\"usage\":null}\n\n",
            "data: {\"id\":\"x\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}],\"usage\":null}\n\n",
            "data: {\"id\":\"x\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"!\"}}],\"usage\":null}\n\n",
            "data: {\"id\":\"x\",\"model\":\"gpt-4o-mini\",\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2,\"total_tokens\":13}}\n\n",
            "data: [DONE]\n\n",
        );
        s.as_bytes().to_vec()
    }

    fn anthropic_sample() -> Vec<u8> {
        let s = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"claude-3-5-sonnet-latest\",\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: ping\n",
            "data: {\"type\":\"ping\"}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":12}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        s.as_bytes().to_vec()
    }

    #[test]
    fn sse_framer_handles_concat_split_anywhere() {
        let bytes = openai_sample();
        // For every possible single split point, feed the bytes in two
        // chunks and verify we recover the same frame sequence.
        let mut reference = SseFramer::new();
        let ref_frames = reference.push(&bytes);
        assert!(!ref_frames.is_empty());

        for split in 1..bytes.len() {
            let mut framer = SseFramer::new();
            let mut collected = framer.push(&bytes[..split]);
            collected.extend(framer.push(&bytes[split..]));
            assert_eq!(
                collected, ref_frames,
                "split at {split} produced a different frame sequence"
            );
        }
    }

    #[test]
    fn openai_parser_extracts_usage_from_terminal_frame() {
        let bytes = openai_sample();
        let mut framer = SseFramer::new();
        let frames = framer.push(&bytes);
        let mut parser = OpenAiStreamParser::new();
        for frame in &frames {
            parser.on_frame(frame);
        }
        let usage = parser.finalize();
        assert_eq!(usage.prompt_tokens, 11);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, 13);
        assert_eq!(parser.response_model().as_deref(), Some("gpt-4o-mini"));
        assert!(parser.saw_terminal_marker());
    }

    #[test]
    fn openai_parser_handles_done_sentinel() {
        // `data: [DONE]` must not break JSON parsing or clobber state.
        let mut parser = OpenAiStreamParser::new();
        parser.on_frame(&SseFrame {
            event: None,
            data: "[DONE]".to_string(),
        });
        assert!(parser.saw_terminal_marker());
        assert_eq!(parser.finalize(), TokenUsage::default());
    }

    #[test]
    fn anthropic_parser_replaces_output_tokens_on_delta() {
        // message_start sets output_tokens=1, message_delta says 12 → final=12.
        let bytes = anthropic_sample();
        let mut framer = SseFramer::new();
        let frames = framer.push(&bytes);
        let mut parser = AnthropicStreamParser::new();
        for frame in &frames {
            parser.on_frame(frame);
        }
        let usage = parser.finalize();
        assert_eq!(usage.prompt_tokens, 25);
        assert_eq!(usage.completion_tokens, 12);
        assert!(parser.saw_terminal_marker());
    }

    #[test]
    fn anthropic_parser_ignores_unknown_events() {
        // Inject a random unknown event — parser should ignore it, state
        // from the prior real events stands.
        let mut parser = AnthropicStreamParser::new();
        parser.on_frame(&SseFrame {
            event: Some("message_start".into()),
            data:
                r#"{"message":{"model":"claude-3","usage":{"input_tokens":5,"output_tokens":1}}}"#
                    .into(),
        });
        parser.on_frame(&SseFrame {
            event: Some("totally_unknown_event".into()),
            data: r#"{"garbage":true}"#.into(),
        });
        parser.on_frame(&SseFrame {
            event: Some("message_delta".into()),
            data: r#"{"usage":{"output_tokens":7}}"#.into(),
        });
        let usage = parser.finalize();
        assert_eq!(usage.prompt_tokens, 5);
        assert_eq!(usage.completion_tokens, 7);
    }
}
