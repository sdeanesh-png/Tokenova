# Tokenova — Task Tracker

Active deferred work, crossed off as commits land on `main`. Source of truth for next-slice planning.

## Session 2 Slice 2 — Polish (✅ shipped)

Low/medium tech debt surfaced by QA rounds 1 & 2. Small surgical changes, same files, one commit.

- [x] **Handler dedupe** — extract shared `proxy_request()` pipeline from `handlers/openai.rs` + `handlers/anthropic.rs`. *(QA R1 Low #8)*
- [x] **Drop `spawn_blocking` for heuristic classifier** — run inline; 2.6µs doesn't justify the thread-pool hop. Restore it when ONNX lands. *(QA R1 Low #9)*
- [x] **Warn on dropped upstream headers** — `upstream::forward` silently skips invalid headers; add `tracing::warn!`. *(QA R1 Low #10)*
- [x] **Classifier threshold boundary test** — verify near-threshold prompts route to `Other` deterministically. *(QA R1 Medium #7)*
- [x] **Doc `StreamingAccumulator` Drop-emit idempotency** — comment the atomic-CAS invariant. *(QA R2 Low #1)*

## Session 2 Slice 3 — Schema + security (✅ shipped)

- [x] **Attribution tags → `Option<String>`** — distinguish missing header from empty-string value. *(QA R1 Medium #6)*
- [x] **Gate `test_sink` behind `test-utils` feature** — production builds no longer carry the log-sink API surface. *(QA R2 Low #2)*

## Session 2 Slice 4+ — Real scope (PRD §8 roadmap)

- [ ] TimescaleDB persistence for `LogRecord` stream
- [ ] Control-plane dashboard scaffold (Next.js 14, PRD §6.2)
- [x] ~~Sentry wiring across proxy~~ (shipped); dashboard wiring deferred until dashboard exists *(PRD §10.5)*
- [ ] PostHog wiring *(PRD §10.5)*
- [ ] Provider expansion: ~~Azure OpenAI~~ (shipped), Bedrock, Vertex, Cohere, Mistral *(PRD Phase 2)*
- [ ] Policy engine (OPA/Rego) *(PRD Phase 2)*

## Non-code (PRD §10.5)

- [ ] Stripe: 3 pricing tiers + metered overage
- [ ] Cloudflare DNS → `tokenova.ai`
- [ ] Patent filings (5 patents)

---

## Done

- [x] Session 1: universal proxy + Semantic Token Clustering v1 — [`31af3cc`](https://github.com/sdeanesh-png/Tokenova/commit/31af3cc)
- [x] Session 2 Slice 1: streaming token accounting + QA-round-1 blockers/highs — [`c780206`](https://github.com/sdeanesh-png/Tokenova/commit/c780206)
- [x] Session 2 Slice 2: handler dedupe, inline classifier, header-drop warn, threshold test, Drop-emit docs — [`1f0456e`](https://github.com/sdeanesh-png/Tokenova/commit/1f0456e)
- [x] Session 2 Slice 3: attribution `Option<String>` + `test_sink` feature gate — [`9a42958`](https://github.com/sdeanesh-png/Tokenova/commit/9a42958)
- [x] Session 2 Slice 4: Sentry wiring (DSN-gated, no-op when unset) — [`6a374b3`](https://github.com/sdeanesh-png/Tokenova/commit/6a374b3)
- [x] Session 2 Slice 5: Azure OpenAI provider (deployment-based URLs, path+query passthrough, streaming + buffered, tests) — [`32f54f9`](https://github.com/sdeanesh-png/Tokenova/commit/32f54f9)
