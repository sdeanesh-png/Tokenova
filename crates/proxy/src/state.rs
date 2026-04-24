//! Shared application state passed to every handler.
//!
//! Built once at startup and cloned cheaply into each request — `Arc`
//! internally for the classifier and pricing table, and `reqwest::Client`
//! is already internally `Arc`ed.

use std::sync::Arc;

use reqwest::Client;
use tokenova_classifier::{Classifier, HeuristicEmbedder};

use crate::pricing::PricingTable;

#[derive(Clone)]
pub struct AppState {
    pub http: Client,
    pub openai_upstream: String,
    pub anthropic_upstream: String,
    /// Azure OpenAI resource base URL. `None` means Azure routes respond
    /// with 503 Service Unavailable — safe default for deployments that
    /// don't proxy Azure.
    pub azure_upstream: Option<String>,
    pub classifier: Arc<Classifier<HeuristicEmbedder>>,
    pub pricing: Arc<PricingTable>,
}

impl AppState {
    pub fn new(
        openai_upstream: String,
        anthropic_upstream: String,
        azure_upstream: Option<String>,
    ) -> anyhow::Result<Self> {
        let http = Client::builder()
            .http2_prior_knowledge()
            // HTTP/2 requires TLS for public endpoints; fall back to
            // protocol negotiation when talking to plain-HTTP test servers.
            .http2_adaptive_window(true)
            .pool_max_idle_per_host(64)
            .build()?;

        Ok(Self {
            http,
            openai_upstream,
            anthropic_upstream,
            azure_upstream,
            classifier: Arc::new(Classifier::new(HeuristicEmbedder::new())),
            pricing: Arc::new(PricingTable::default()),
        })
    }

    /// Test-only constructor that avoids HTTP/2 prior knowledge so it works
    /// against wiremock's plain HTTP/1 server.
    pub fn for_tests(
        openai_upstream: String,
        anthropic_upstream: String,
        azure_upstream: Option<String>,
    ) -> anyhow::Result<Self> {
        let http = Client::builder().pool_max_idle_per_host(4).build()?;
        Ok(Self {
            http,
            openai_upstream,
            anthropic_upstream,
            azure_upstream,
            classifier: Arc::new(Classifier::new(HeuristicEmbedder::new())),
            pricing: Arc::new(PricingTable::default()),
        })
    }
}
