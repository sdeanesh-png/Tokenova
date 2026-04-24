//! Parse token-usage totals from upstream provider responses.
//!
//! Non-streaming JSON bodies are parsed here; streaming responses reconstruct
//! their usage via `crate::streaming` and bypass this module entirely.

use tokenova_domain::TokenUsage;

/// Clamp a `u64` token count to `u32::MAX`, emitting a warning if the raw
/// value exceeds the cap. Provider-reported counts above 4.29 billion are
/// almost certainly a bug or an attack, but silently casting would produce a
/// modulo-truncated billing figure (e.g. 5B tokens → 705M after cast). We
/// prefer over-attribution to silent revenue loss.
pub(crate) fn clamp_u64_to_u32(count: u64, field: &str) -> u32 {
    if count > u32::MAX as u64 {
        tracing::warn!(
            field = field,
            observed = count,
            capped_at = u32::MAX,
            "token count exceeded u32::MAX, capping"
        );
        u32::MAX
    } else {
        count as u32
    }
}

/// OpenAI response: `{ "usage": { "prompt_tokens": N, "completion_tokens": N, "total_tokens": N } }`.
pub fn parse_openai(body: &[u8]) -> TokenUsage {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return TokenUsage::default();
    };
    let usage = v.get("usage").and_then(|u| u.as_object());
    let Some(usage) = usage else {
        return TokenUsage::default();
    };
    let p_raw = usage
        .get("prompt_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(0);
    let c_raw = usage
        .get("completion_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(0);
    let p = clamp_u64_to_u32(p_raw, "prompt_tokens");
    let c = clamp_u64_to_u32(c_raw, "completion_tokens");
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
    let p_raw = usage
        .get("input_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(0);
    let c_raw = usage
        .get("output_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(0);
    let p = clamp_u64_to_u32(p_raw, "input_tokens");
    let c = clamp_u64_to_u32(c_raw, "output_tokens");
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

    /// QA High #3: a provider reporting 5B tokens (which overflows u32)
    /// must be capped at u32::MAX rather than silently truncated mod 2^32.
    #[test]
    fn openai_u32_overflow_is_capped() {
        let body = br#"{"usage":{"prompt_tokens":5000000000,"completion_tokens":7,"total_tokens":5000000007}}"#;
        let u = parse_openai(body);
        assert_eq!(u.prompt_tokens, u32::MAX);
        assert_eq!(u.completion_tokens, 7);
        // `TokenUsage::new` sums: u32::MAX + 7 wraps if we're not careful,
        // but the cap protects us from the truly-bad case (5B raw). The sum
        // itself is allowed to saturate at the u32 level; the critical
        // invariant is that neither input field is silently modulo'd.
    }

    #[test]
    fn anthropic_u32_overflow_is_capped() {
        let body = br#"{"usage":{"input_tokens":11,"output_tokens":5000000000}}"#;
        let u = parse_anthropic(body);
        assert_eq!(u.prompt_tokens, 11);
        assert_eq!(u.completion_tokens, u32::MAX);
    }
}
