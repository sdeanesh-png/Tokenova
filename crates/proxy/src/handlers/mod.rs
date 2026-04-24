//! Per-provider HTTP handlers.
//!
//! Each route handler is a thin adapter that builds the upstream URL and
//! dispatches to [`shared::proxy_request`]. All pipeline logic —
//! attribution, streaming detection, forwarding, classification, log
//! emission — lives in the shared module; per-provider variation is
//! injected through the [`shared::ProviderContract`] trait.

pub mod anthropic;
pub mod azure;
pub mod openai;
pub(crate) mod shared;
