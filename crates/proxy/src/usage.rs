//! Parse token-usage totals from upstream provider responses.
//!
//! Today we parse **non-streamed** JSON bodies only. Streaming responses
//! still pass through transparently but their token counts are reported as
//! zero until the streaming accumulator lands in Session 2. The `streamed`
//! bool on [`tokenova_domain::LogRecord`] makes the distinction explicit so
//! downstream analytics can filter.

use tokenova_domain::TokenUsage;

/// OpenAI response: `{ "usage": { "prompt_tokens": N, "completion_tokens": N, "total_tokens": N } }`.
pub fn parse_openai(body: &[u8]) -> TokenUsage {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return TokenUsage::default();
    };
    let usage = v.get("usage").and_then(|u| u.as_object());
    let Some(usage) = usage else {
        return TokenUsage::default();
    };
    let p = usage
        .get("prompt_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as u32;
    let c = usage
        .get("completion_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as u32;
    TokenUsage::new(p, c)
}

/// Anthropic response: `{ "usage": { "input_tokens": N, "output_tokens": N } }`.
pub fn parse_anthropic(body: &[u8]) -> TokenUsage {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return TokenUsage::default();
    };
    let usage = v.get("usage").and_then(|u| u.as_object());
    let Some(usage) = usage else {
        return TokenUsage::default();
    };
    let p = usage
        .get("input_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as u32;
    let c = usage
        .get("output_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as u32;
    TokenUsage::new(p, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_happy_path() {
        let body = br#"{"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}"#;
        let u = parse_openai(body);
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 20);
        assert_eq!(u.total_tokens, 30);
    }

    #[test]
    fn anthropic_happy_path() {
        let body = br#"{"usage":{"input_tokens":5,"output_tokens":7}}"#;
        let u = parse_anthropic(body);
        assert_eq!(u.prompt_tokens, 5);
        assert_eq!(u.completion_tokens, 7);
        assert_eq!(u.total_tokens, 12);
    }

    #[test]
    fn missing_usage_is_zero() {
        assert_eq!(parse_openai(b"{}"), TokenUsage::default());
        assert_eq!(parse_anthropic(b"{}"), TokenUsage::default());
    }

    #[test]
    fn malformed_body_is_zero() {
        assert_eq!(parse_openai(b"not json"), TokenUsage::default());
        assert_eq!(parse_anthropic(b""), TokenUsage::default());
    }
}
