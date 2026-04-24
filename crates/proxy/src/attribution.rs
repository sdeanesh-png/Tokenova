//! Attribution-tag extraction.
//!
//! Reads `x-tokenova-*` headers into an [`AttributionTags`] record. Missing
//! headers yield `None`; present-but-empty headers yield `Some("")`. This
//! distinction is load-bearing for billing analytics (QA R1 Medium #6) —
//! the drop-in SDK promise in PRD §5.1.1 ("zero application code changes
//! beyond endpoint URL") means we cannot reject requests that lack tags,
//! but we must preserve the signal that a caller didn't send one.

use axum::http::HeaderMap;
use tokenova_domain::AttributionTags;

pub fn extract(headers: &HeaderMap) -> AttributionTags {
    AttributionTags {
        user: read(headers, "x-tokenova-user"),
        team: read(headers, "x-tokenova-team"),
        department: read(headers, "x-tokenova-department"),
        project: read(headers, "x-tokenova-project"),
        application: read(headers, "x-tokenova-application"),
        cost_center: read(headers, "x-tokenova-cost-center"),
        environment: read(headers, "x-tokenova-environment"),
    }
}

fn read(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderName, HeaderValue};

    fn hm(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (k, v) in pairs {
            map.insert(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        map
    }

    #[test]
    fn missing_headers_yield_none() {
        let tags = extract(&hm(&[]));
        assert_eq!(tags.user, None);
        assert_eq!(tags.team, None);
        assert_eq!(tags.project, None);
    }

    #[test]
    fn present_empty_header_yields_some_empty() {
        let tags = extract(&hm(&[("x-tokenova-user", "")]));
        assert_eq!(tags.user, Some(String::new()));
        assert_eq!(tags.team, None);
    }

    #[test]
    fn present_value_yields_some_value() {
        let tags = extract(&hm(&[
            ("x-tokenova-user", "steve"),
            ("x-tokenova-team", "founders"),
        ]));
        assert_eq!(tags.user, Some("steve".into()));
        assert_eq!(tags.team, Some("founders".into()));
        assert_eq!(tags.department, None);
    }

    #[test]
    fn serializes_missing_as_null_present_empty_as_empty_string() {
        let tags = extract(&hm(&[("x-tokenova-user", "")]));
        let json = serde_json::to_value(&tags).unwrap();
        assert_eq!(json["user"], serde_json::Value::String(String::new()));
        assert_eq!(json["team"], serde_json::Value::Null);
        assert_eq!(json["department"], serde_json::Value::Null);
    }
}
