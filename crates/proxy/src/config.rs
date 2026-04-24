//! Process-level configuration loaded from environment variables.
//!
//! 12-factor by design. No config file. Defaults favor the "drop-in for
//! OpenAI/Anthropic" story — run the binary with no env vars and it proxies
//! to the real providers on port 8080.

use std::env;
use std::net::SocketAddr;

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: SocketAddr,
    pub openai_upstream: String,
    pub anthropic_upstream: String,
    pub log_format: LogFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    Json,
    Pretty,
}

impl Config {
    pub fn from_env() -> anyhow::Result<Self> {
        let port: u16 = env::var("TOKENOVA_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8080);
        let listen_addr = SocketAddr::from(([0, 0, 0, 0], port));

        let openai_upstream = env::var("TOKENOVA_OPENAI_UPSTREAM")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        let anthropic_upstream = env::var("TOKENOVA_ANTHROPIC_UPSTREAM")
            .unwrap_or_else(|_| "https://api.anthropic.com".to_string());

        let log_format = match env::var("TOKENOVA_LOG_FORMAT").unwrap_or_default().as_str() {
            "pretty" => LogFormat::Pretty,
            _ => LogFormat::Json,
        };

        Ok(Self {
            listen_addr,
            openai_upstream,
            anthropic_upstream,
            log_format,
        })
    }
}
