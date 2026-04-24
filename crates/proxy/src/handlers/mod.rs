//! Per-provider HTTP handlers.
//!
//! Each handler:
//!
//! 1. Extracts attribution tags from `x-tokenova-*` headers.
//! 2. Buffers the request body (small, by design — prompts).
//! 3. Forwards the request to the upstream provider.
//! 4. Buffers the response body (the common case; streamed responses land
//!    in Session 2).
//! 5. Parses provider-specific token usage + model out of the response.
//! 6. Runs the classifier **off the hot path** in `tokio::spawn` using the
//!    prompt text extracted from the request body — the classification
//!    future is joined before the [`LogRecord`] is emitted, not before the
//!    client sees the response.
//! 7. Emits one structured `LogRecord` to stdout.
//!
//! The <5ms p99 latency-overhead SLA (PRD §5.1.2) is protected by (a) the
//! off-hot-path classification and (b) a persistent HTTP client with HTTP/2
//! multiplexing to the upstream.

pub mod anthropic;
pub mod openai;
