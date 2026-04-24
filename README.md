# Tokenova

**Enterprise AI Token Visibility, Forecasting & Optimization Platform.**

Tokenova is a transparent, high-performance API proxy that sits in front of
every enterprise AI provider (OpenAI, Anthropic, Azure OpenAI, Bedrock,
Gemini, …) and turns unmanaged token spend into a governed, measurable
asset. It intercepts calls, classifies their semantic intent in real time,
attributes token cost to users/teams/projects, and — in later phases —
enforces budget policy, optimizes prompts, and forecasts consumption.

Patents 1–5 cover the novel parts of this stack (see PRD §7):
semantic token clustering, context-window forecasting, intent-aware policy
routing, dynamic context distillation, and semantic cache deduplication.

This repository currently contains the **Session 1** build (PRD §10.5):
the universal proxy plus v1 of the Semantic Token Clustering classifier
(Patent 1).

---

## What's in here

| Crate | Purpose |
| --- | --- |
| [`crates/domain`](crates/domain) | Shared types: `AttributionTags`, `TokenUsage`, `IntentCategory`, `LogRecord`. |
| [`crates/classifier`](crates/classifier) | Semantic Token Clustering v1. Default = zero-dep hashed embedder; ONNX backend behind the `onnx` feature. |
| [`crates/proxy`](crates/proxy) | Axum/Tokio HTTP proxy for OpenAI `/v1/chat/completions`, `/v1/embeddings`, and Anthropic `/v1/messages`. |

### Supported providers (Session 1)

- OpenAI (`/v1/chat/completions`, `/v1/embeddings`)
- Anthropic (`/v1/messages`)

More providers (Azure OpenAI, Bedrock, Vertex, Cohere, Mistral) land in
Phase 2 per the roadmap.

### Intent categories (10)

`code_generation`, `summarization`, `data_extraction`, `question_answering`,
`document_review`, `translation`, `creative_writing`, `conversational_chat`,
`reasoning_planning`, `other`.

---

## Quickstart

### Requirements

- Rust (stable, pinned via `rust-toolchain.toml`). Install with
  [`rustup`](https://rustup.rs/).
- `cargo`, `rustfmt`, `clippy` (installed by default).

### Build and test

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release -p tokenova-proxy
```

### Run the proxy

```bash
TOKENOVA_PORT=8080 \
TOKENOVA_OPENAI_UPSTREAM=https://api.openai.com \
TOKENOVA_ANTHROPIC_UPSTREAM=https://api.anthropic.com \
  cargo run --release -p tokenova-proxy
```

Then point any OpenAI or Anthropic SDK at `http://localhost:8080` instead
of the upstream URL:

```bash
curl -sS http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "x-tokenova-user: steve" \
  -H "x-tokenova-team: founders" \
  -H "x-tokenova-project: tokenova-session-1" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

Every call produces one JSON `LogRecord` line on stdout:

```json
{
  "request_id": "…",
  "received_at": "2026-04-23T23:44:00Z",
  "provider": "openai",
  "model": "gpt-4o",
  "attribution": { "user": "steve", "team": "founders": "…"},
  "usage": { "prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15 },
  "cost_usd": 0.00006,
  "intent": "question_answering",
  "latency_added_ms": 1.24,
  "upstream_status": 200,
  "streamed": false
}
```

### Attribution headers

Pass these on every proxied request to get per-team/per-project cost
attribution:

| Header | Purpose |
| --- | --- |
| `x-tokenova-user` | End-user identifier |
| `x-tokenova-team` | Team / squad |
| `x-tokenova-department` | Business unit |
| `x-tokenova-project` | Application or project |
| `x-tokenova-application` | Logical app name |
| `x-tokenova-cost-center` | Finance cost code |
| `x-tokenova-environment` | `dev` / `staging` / `prod` |

`x-tokenova-*` headers are **never** forwarded upstream — they are
consumed by the proxy and redacted from the outgoing request.

### Config (env vars)

| Var | Default | Purpose |
| --- | --- | --- |
| `TOKENOVA_PORT` | `8080` | Listen port |
| `TOKENOVA_OPENAI_UPSTREAM` | `https://api.openai.com` | OpenAI base URL |
| `TOKENOVA_ANTHROPIC_UPSTREAM` | `https://api.anthropic.com` | Anthropic base URL |
| `TOKENOVA_LOG_FORMAT` | `json` | `json` or `pretty` |
| `TOKENOVA_LOG` | `info` | `tracing` filter directive |

---

## Classifier backends

The `Embedder` trait has two implementations:

1. **`HeuristicEmbedder`** (default). Deterministic FNV-1a hashed unigrams
   and bigrams into a 512-dim vector. No native deps, ~2.6 µs per
   classification on an M-series Mac (see `cargo bench -p
   tokenova-classifier`). Good enough for the 10-way argmax on short
   enterprise prompts — shipped as the Session 1 default so the build has
   zero external dependencies.

2. **`OnnxEmbedder`** (behind the `onnx` Cargo feature). Wraps an ONNX
   Runtime session over a MiniLM-L6 / L12 export, mean-pools the token
   embeddings, and L2-normalizes. PRD §6.2 specifies an 80M-param model;
   the `onnx` feature ships with the L6 variant for speed and supports a
   drop-in L12 swap.

To enable ONNX classification:

```bash
./scripts/fetch-model.sh                 # downloads MiniLM ONNX + tokenizer
brew install onnxruntime                  # macOS
export ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib
cargo build --release \
  --features "tokenova-classifier/onnx" -p tokenova-proxy
```

---

## Architecture (Session 1)

```
┌──────────────┐  attribution headers + prompt body
│   SDK call   │ ─────────────────────────────────┐
└──────────────┘                                    │
                                                    ▼
                 ┌───────────────────────────────────────────┐
                 │           tokenova-proxy (axum)           │
                 │                                           │
                 │   ┌─────────┐     ┌──────────────────┐    │
                 │   │ attrib  │     │   pricing table  │    │
                 │   └─────────┘     └──────────────────┘    │
                 │                                           │
                 │   ┌──────────────────────────────────┐    │
                 │   │ handlers::openai / ::anthropic   │───────► upstream
                 │   └──────────────────────────────────┘    │
                 │         │              ▲                  │
                 │         ▼              │                  │
                 │   ┌─────────────────────────────────┐     │
                 │   │ classifier (off the hot path)   │     │
                 │   │   Embedder + 10 prototypes      │     │
                 │   └─────────────────────────────────┘     │
                 │                                           │
                 │   structured LogRecord → stdout           │
                 └───────────────────────────────────────────┘
```

Classifier runs inside `tokio::task::spawn_blocking` after the response
body is collected, protecting the <5 ms p99 added-latency SLA from PRD
§5.1.2.

---

## PRD cross-reference

- Patent 1 / Semantic Token Clustering → `crates/classifier`
- Attribution tagging (§5.1.1) → `crates/proxy/src/attribution.rs`
- Universal proxy (§5.1) → `crates/proxy/src/handlers/`
- Static pricing table → `crates/proxy/src/pricing.rs`
- Structured log record (§5.1.1) → `tokenova_domain::LogRecord`

Roadmap (PRD §8) — not in this Session 1 build:

- Policy engine (OPA/Rego) — Phase 2
- Forecasting engine — Phase 3
- Prompt compression, semantic cache — Phase 3
- TimescaleDB persistence + Kafka bus — Phase 2+
- Control-plane dashboard — Session 2
