//! Static USD pricing table per (provider, model). Prices are USD per 1M
//! tokens, split into input/output rates.
//!
//! Hand-maintained for Session 1; Phase 2 replaces this with a fetch from a
//! pricing service (and ingests provider SKU updates automatically).
//!
//! Source-of-truth prices as of April 2026 per the respective provider
//! pricing pages. When a model isn't in the table we fall back to the
//! provider's flagship rate — better to over-attribute than to under-bill.

use std::collections::HashMap;

use tokenova_domain::{Provider, TokenUsage};

#[derive(Debug, Clone, Copy)]
pub struct ModelRate {
    pub input_per_m: f64,
    pub output_per_m: f64,
}

impl ModelRate {
    pub fn cost_usd(&self, usage: &TokenUsage) -> f64 {
        let input = (usage.prompt_tokens as f64 / 1_000_000.0) * self.input_per_m;
        let output = (usage.completion_tokens as f64 / 1_000_000.0) * self.output_per_m;
        input + output
    }
}

#[derive(Debug, Clone)]
pub struct PricingTable {
    // Key: (provider, lowercased_model_id)
    rates: HashMap<(Provider, String), ModelRate>,
    fallbacks: HashMap<Provider, ModelRate>,
}

impl Default for PricingTable {
    fn default() -> Self {
        let mut rates = HashMap::new();

        // OpenAI
        insert(
            &mut rates,
            Provider::OpenAi,
            "gpt-4o",
            ModelRate {
                input_per_m: 2.50,
                output_per_m: 10.00,
            },
        );
        insert(
            &mut rates,
            Provider::OpenAi,
            "gpt-4o-mini",
            ModelRate {
                input_per_m: 0.15,
                output_per_m: 0.60,
            },
        );
        insert(
            &mut rates,
            Provider::OpenAi,
            "gpt-4-turbo",
            ModelRate {
                input_per_m: 10.00,
                output_per_m: 30.00,
            },
        );
        insert(
            &mut rates,
            Provider::OpenAi,
            "gpt-3.5-turbo",
            ModelRate {
                input_per_m: 0.50,
                output_per_m: 1.50,
            },
        );

        // Anthropic
        insert(
            &mut rates,
            Provider::Anthropic,
            "claude-3-5-sonnet-latest",
            ModelRate {
                input_per_m: 3.00,
                output_per_m: 15.00,
            },
        );
        insert(
            &mut rates,
            Provider::Anthropic,
            "claude-3-opus-latest",
            ModelRate {
                input_per_m: 15.00,
                output_per_m: 75.00,
            },
        );
        insert(
            &mut rates,
            Provider::Anthropic,
            "claude-3-haiku-20240307",
            ModelRate {
                input_per_m: 0.25,
                output_per_m: 1.25,
            },
        );

        let mut fallbacks = HashMap::new();
        fallbacks.insert(
            Provider::OpenAi,
            ModelRate {
                input_per_m: 2.50,
                output_per_m: 10.00,
            },
        );
        fallbacks.insert(
            Provider::Anthropic,
            ModelRate {
                input_per_m: 3.00,
                output_per_m: 15.00,
            },
        );

        Self { rates, fallbacks }
    }
}

fn insert(
    map: &mut HashMap<(Provider, String), ModelRate>,
    provider: Provider,
    model: &str,
    rate: ModelRate,
) {
    map.insert((provider, model.to_lowercase()), rate);
}

impl PricingTable {
    pub fn rate(&self, provider: Provider, model: &str) -> ModelRate {
        self.rates
            .get(&(provider, model.to_lowercase()))
            .copied()
            .unwrap_or_else(|| {
                *self
                    .fallbacks
                    .get(&provider)
                    .expect("fallback rate must exist for every provider")
            })
    }
}
