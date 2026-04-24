//! Tokenova universal AI proxy — library crate.
//!
//! Binary crate is `main.rs`; the library exposes the same types to the
//! integration tests in `tests/`.

pub mod attribution;
pub mod config;
pub mod handlers;
pub mod observability;
pub mod pricing;
pub mod router;
pub mod state;
pub mod upstream;
pub mod usage;

pub use router::build_router;
pub use state::AppState;
