//! Upstream request construction shared across provider handlers.
//!
//! Rebuilds a `reqwest::Request` from an incoming `axum` request, preserving
//! headers except for hop-by-hop entries that would poison the upstream
//! connection (`Host`, `Connection`, `Keep-Alive`, etc.). The body is passed
//! verbatim — the drop-in promise depends on byte-for-byte passthrough.

use anyhow::Context;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use bytes::Bytes;
use reqwest::{Client, Method, Response};

const HOP_BY_HOP: &[&str] = &[
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
];

// Headers we attach ourselves as part of the proxy contract.
const TOKENOVA_HEADERS_PREFIX: &str = "x-tokenova-";

pub async fn forward(
    http: &Client,
    url: String,
    headers: &HeaderMap,
    body: Bytes,
) -> anyhow::Result<Response> {
    let mut builder = http.request(Method::POST, &url);

    for (name, value) in headers.iter() {
        let lname = name.as_str().to_ascii_lowercase();
        if HOP_BY_HOP.contains(&lname.as_str()) {
            continue;
        }
        if lname.starts_with(TOKENOVA_HEADERS_PREFIX) {
            // Attribution tags are consumed, not forwarded. Upstream doesn't
            // need (or want) them and some providers reject unknown headers.
            continue;
        }
        if let (Ok(hn), Ok(hv)) = (
            HeaderName::from_bytes(name.as_ref()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(hn, hv);
        }
    }

    let response = builder
        .body(body)
        .send()
        .await
        .with_context(|| format!("forwarding to {url}"))?;

    Ok(response)
}
