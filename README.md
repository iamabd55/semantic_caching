# Semantic Caching Gateway for LLM APIs

A gateway that sits between an application and an LLM provider, using vector similarity to detect semantically equivalent requests and avoid redundant LLM calls — instead of relying on exact-match caching, which fails on natural language.

> **Status:** 🚧 Under active development. See [Project Status](#project-status) for what's built vs. in progress. Benchmark numbers below are placeholders until real evaluation runs are complete — see [`AGENTS.md`](./AGENTS.md) Section 33: never claim a result without evidence.

---

## Problem

Applications calling LLM APIs repeatedly receive requests that mean the same thing but are worded differently:

```text
"What is machine learning?"
"Can you explain machine learning?"
"Please describe machine learning in simple words."
```

A traditional cache using exact string matching treats these as three different requests, causing unnecessary API costs, added latency, and repeated computation for requests that have effectively already been answered.

This project builds a **semantic caching gateway**: instead of matching text exactly, it converts each request into an embedding and uses vector similarity search to find whether a sufficiently similar request has already been processed — returning the cached response instead of calling the LLM again.

---

## What Makes This More Than a Vector-Search Demo

Most "semantic cache" side projects stop at *embedding → vector DB → similarity match*. This project treats that as the easy 20% and focuses the real engineering effort on the parts that make a cache trustworthy in practice:

- **Two-stage lookup** — a cheap exact-hash check before ever generating an embedding
- **Canonicalization vs. original request** — the text used for embedding is never the text sent to the LLM, preventing caching logic from silently altering what the model actually receives
- **Tenant + model/version isolation**, filtered *before* similarity ranking, not after — prevents cross-user leakage and stale cross-model responses
- **Request coalescing (single-flight locking)** — concurrent identical cache misses trigger exactly one upstream LLM call, not N
- **PII-aware cache bypass** — high-risk requests skip caching entirely rather than attempting redaction, which can make distinct users' requests semantically indistinguishable
- **Evidence-based threshold selection** — the similarity threshold is chosen from a labeled precision/recall/F1 sweep tracked in MLflow, not picked because "0.85 looks reasonable"
- **Explicit, documented limitations** — semantic similarity is not the same as response correctness or freshness, and this project says so rather than pretending otherwise

---

## Architecture

```text
                       Client
                         │
                         ▼
                 ┌───────────────┐
                 │ FastAPI       │
                 │ Gateway       │
                 └───────┬───────┘
                         │
                         ▼
                 Request Validation
                         │
                         ▼
                    PII Policy
                         │
                         ▼
                  Request Hash
                         │
                  ┌──────┴──────┐
                  │             │
                Hit           Miss
                  │             │
                  ▼             ▼
               Return      Canonicalize
                                │
                                ▼
                           Embedding
                                │
                                ▼
                       ┌────────────────┐
                       │ Qdrant         │
                       │ Tenant/Model   │
                       │ Filter         │
                       └───────┬────────┘
                               │
                               ▼
                         Similarity Search
                               │
                       ┌───────┴───────┐
                       │               │
                     Hit              Miss
                       │               │
                       ▼               ▼
                    Return        Redis Lock
                                      │
                                      ▼
                                Re-check Cache
                                      │
                                      ▼
                                Provider Layer
                                 /           \
                              Groq        OpenRouter
                                 \           /
                                  \         /
                                   ▼       ▼
                                  Response
                                      │
                                      ▼
                               Store in Qdrant
                                      │
                                      ▼
                               Release Lock
                                      │
                                      ▼
                                Return Response
```

Full request-flow and design rationale documented in [`PRD.md`](./semantic-cache-gateway-PRD-v2.md).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| API framework | FastAPI |
| LLM providers | OpenRouter (primary), Groq (fallback) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (local, free, no rate limits) |
| Vector database | Qdrant |
| Locking / concurrency | Redis |
| Experiment tracking | MLflow |
| Dashboard | Streamlit |
| Testing | pytest |
| HTTP client | httpx |
| Package/env management | uv |
| Containerization | Docker + Docker Compose |

---

## Project Status

Tracking against the phase order defined in [`AGENTS.md`](./AGENTS.md):

- [x] **Phase 0** — Project setup, `/health` endpoint, environment config
- [x] **Phase 1** — Basic LLM gateway (OpenRouter pass-through)
- [x] **Phase 2** — Exact request hashing
- [x] **Phase 3** — Canonicalization
- [x] **Phase 4** — Local embeddings
- [x] **Phase 5** — Qdrant integration
- [x] **Phase 6** — Semantic cache core
- [x] **Phase 7** — Cache correctness tests
- [x] **Phase 8** — Evaluation dataset
- [x] **Phase 9** — Threshold evaluation
- [x] **Phase 10** — MLflow integration
- [ ] **Phase 11** — Latency + cost benchmark
- [ ] **Phase 12** — Redis single-flight locking
- [ ] **Phase 13** — Tenant isolation
- [ ] **Phase 14** — Model/version isolation
- [ ] **Phase 15** — TTL
- [ ] **Phase 16** — PII bypass
- [ ] **Phase 17** — Provider abstraction
- [ ] **Phase 18** — Provider fallback
- [ ] **Phase 19** — Observability
- [ ] **Phase 20** — Cost analysis
- [ ] **Phase 21** — Dashboard
- [ ] **Phase 22** — Docker Compose
- [ ] **Final** — README, benchmarks, resume summary

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose (for Qdrant, Redis, MLflow — added in later phases)
- An [OpenRouter](https://openrouter.ai) API key
- A [Groq](https://groq.com) API key (used for provider fallback, added in a later phase)

### Installation

```bash
git clone https://github.com/iamabd55/semantic_caching.git
cd semantic_caching

uv python pin 3.11
uv sync
```

### Environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | API key for OpenRouter (primary LLM provider) |
| `GROQ_API_KEY` | API key for Groq (fallback provider) |
| `QDRANT_URL` | Qdrant instance URL (default: `http://localhost:6333`) |
| `REDIS_URL` | Redis instance URL (default: `redis://localhost:6379`) |
| `SIMILARITY_THRESHOLD` | Configurable cosine-similarity cutoff for cache hits |

Never commit `.env` — only `.env.example` is tracked.

---

## Running the System

```bash
uv run uvicorn app.main:app --reload
```

Check it's alive:

```bash
curl http://localhost:8000/health
```

Full Docker Compose stack (FastAPI + Qdrant + Redis + MLflow + Streamlit) will be available from Phase 22 onward — see [Project Status](#project-status).

---

## Running Tests

```bash
uv run pytest
```

---

## Running Evaluation

*(Available from Phase 8 onward.)*

```bash
uv run python evaluation/evaluate.py
uv run python evaluation/threshold_sweep.py
```

The evaluation dataset lives at `evaluation/dataset/cache_eval.json` — labeled `query_a` / `query_b` / `expected_match` pairs used to compute precision, recall, and F1 at multiple similarity thresholds.

---

## MLflow

Experiment tracking is integrated to make all threshold sweeps fully reproducible and auditable.

Start the MLflow UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://127.0.0.1:5000` in your browser.

Every threshold-sweep run in experiment `semantic-cache-threshold-evaluation` logs:
- **Parameters**: `threshold`, `embedding_model`, `dataset_version`, `use_polarity_guard`
- **Metrics**: `precision`, `recall`, `f1`, `hit_rate`, `average_latency`, `accuracy`, `tp`, `fp`, `tn`, `fn`
- **Artifacts**: Per-category confusion matrix breakdown (`category_breakdown.json`)

---

## Benchmark Results

*(Populated once Phases 9–11 are complete. Placeholder structure below — no numbers are invented per [`AGENTS.md`](./AGENTS.md) Section 33.)*

### Threshold Evaluation

Measured on 108 labeled query pairs (`evaluation/dataset/cache_eval.json`) with `all-MiniLM-L6-v2` and Polarity Guard:

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| 0.75 | 0.8889 | 0.7273 | 0.8000 | 0.8148 |
| 0.80 | 0.9189 | 0.6182 | 0.7391 | 0.7778 |
| 0.85 | 0.9259 | 0.4545 | 0.6098 | 0.7037 |
| 0.90 | 1.0000 | 0.2909 | 0.4507 | 0.6389 |
| 0.95 | 1.0000 | 0.0909 | 0.1667 | 0.5370 |

**Selected threshold:** `0.75` (or `0.80` for higher precision) — At `0.75`, the system maximizes overall F1 score (`0.8000`) while maintaining strong precision (`88.89%`). When combined with the Polarity Guard, adversarial false positives on negation and antonym conflicts drop by 16 pairs, making `0.75–0.80` the empirically backed operating range. Full report and ablation analysis available in [`evaluation/threshold_evaluation_report.md`](./evaluation/threshold_evaluation_report.md).

### Latency

| Scenario | Avg Latency | P50 | P95 |
|---|---:|---:|---:|
| Direct LLM call | — | — | — |
| Cache miss (embedding + search + LLM) | — | — | — |
| Cache hit | — | — | — |

### Cost / Efficiency

| Metric | Result |
|---|---:|
| Cache hit rate | — |
| LLM calls avoided | — |
| Estimated cost reduction | — |

---

## Known Limitations

- **Semantic match ≠ response correctness or freshness.** Two requests can be judged semantically equivalent by the embedding model while the *correct* answer has since changed (e.g. "What's the weather in Lahore?" asked a day apart). This version does not attempt to detect or handle time-sensitive queries — see [Future Work](#future-work).
- **Bi-encoder polarity blindness & heuristic post-filter.** Dense bi-encoder embeddings (`all-MiniLM-L6-v2`) embed opposing polarities (e.g. "good" vs "bad", or negated queries) very close together (~0.94 cosine similarity) due to shared vocabulary. We implemented a lightweight post-filter (`has_polarity_conflict`) checking common negation words and an explicit list of ~17 domain antonym pairs. This is a narrow patch for known cases found during testing, not comprehensive negation handling.
- **PII detection is a lightweight regex/pattern check, not a guarantee.** High-risk requests bypass the cache entirely rather than attempting redaction, since redaction can make distinct users' requests semantically indistinguishable to the embedding model (e.g. two different names both becoming `[PERSON]`) — but the detector itself is not claimed to catch every case.
- **No encryption at rest is implemented** for cached request/response data in this version. A production deployment should use a managed Qdrant/Redis instance with encryption at rest enabled.
- **No circuit breaker** on provider failures in v1 — retries and fallback ordering exist, but sustained upstream failure isn't yet handled with a breaker pattern.

---

## Future Work

- Cross-encoder / NLI re-ranking stage as a generalized solution for negation and polarity conflict detection
- Freshness-aware caching (detecting time-sensitive queries and adjusting TTL/cache eligibility accordingly)
- Circuit breaker for sustained provider failures
- Dynamic/adaptive similarity thresholds instead of a single global value
- Advanced cache eviction policies beyond TTL/LRU

---

## Project Documents

- [`PRD.md`](./semantic-cache-gateway-PRD-v2.md) — full product requirements, phase-by-phase build plan, and design rationale
- [`AGENTS.md`](./AGENTS.md) — instructions and constraints for AI coding agents working on this repository

---

## License

MIT
