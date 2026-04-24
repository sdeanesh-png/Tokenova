//! Attribution-tag extraction.
//!
//! Reads `x-tokenova-*` headers into an [`AttributionTags`] record. Missing
//! headers are treated as empty strings — the drop-in SDK promise in PRD
//! §5.1.1 ("zero application code changes beyond endpoint URL") means we
//! cannot reject a request just because tags are absent.

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

fn read(headers: &HeaderMap, name: &str) -> String {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string()
}
