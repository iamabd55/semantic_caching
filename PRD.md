# Product Requirements Document
## Semantic Caching Gateway for LLM APIs

**Status:** Draft v1
**Type:** Personal / Portfolio Infrastructure Project

---

# 1. Problem Statement

Applications that use LLM APIs can repeatedly send requests that are semantically similar but not textually identical.

For example:

```text
"What is machine learning?"
"Can you explain machine learning?"
"Please describe machine learning in simple words."
```

A traditional exact-match cache treats these as different requests and makes unnecessary LLM calls.

This results in:

- unnecessary API costs
- increased latency
- repeated computation
- unnecessary provider usage

The project will build a **semantic caching gateway** that sits between an application and an LLM provider.

Instead of relying only on exact string matching, the gateway will use embeddings and vector similarity to identify previously processed requests with similar meaning.

The project will focus on the engineering challenges around semantic caching:

- cache correctness
- semantic similarity thresholds
- exact-match optimization
- concurrency safety
- tenant isolation
- PII handling
- observability
- evaluation
- latency and cost measurement
- provider reliability

---

# 2. Project Goal

Build a provider-agnostic gateway that can act as an intermediate layer between an application and LLM APIs.

The gateway should:

1. Receive an LLM request.
2. Validate and identify the request.
3. Check for an exact cache match using a request hash.
4. If there is no exact match, generate an embedding.
5. Search the semantic cache using vector similarity.
6. Return a cached response when similarity exceeds the selected threshold.
7. If no suitable match exists, call the LLM provider.
8. Store the new request/response pair.
9. Prevent duplicate LLM calls during concurrent cache misses.
10. Protect cached responses between tenants.
11. Measure cache quality, latency, and estimated cost savings.
12. Provide experimental evidence for the selected similarity threshold.

---

# 3. Core Design Principle

The project must distinguish between the **original request** and the **normalized request used for semantic comparison**.

### Original request

The exact request received from the client.

It must remain unchanged when sent to the LLM.

### Canonicalized request

A normalized representation used only for embedding and semantic comparison.

Example:

```text
Original:
"  Explain   Machine Learning!! "

Canonicalized:
"explain machine learning"
```

The canonicalized version must **never replace the original request sent to the LLM**.

This prevents preprocessing performed for caching from accidentally changing the meaning, formatting, code, JSON, or other important content of the actual LLM request.

---

# 4. Main Request Flow

```text
                        Client Application
                               │
                               ▼
                      ┌─────────────────┐
                      │ FastAPI Gateway │
                      └────────┬────────┘
                               │
                               ▼
                       Request Validation
                               │
                               ▼
                       PII / Eligibility Check
                               │
                               ▼
                        Generate Request Hash
                               │
                               ▼
                      Exact Cache Lookup
                         /             \
                       HIT             MISS
                        │                │
                        ▼                ▼
                   Return Response   Canonicalize
                                         │
                                         ▼
                                  Generate Embedding
                                         │
                                         ▼
                                    Qdrant Search
                                         │
                                         ▼
                                Similarity Threshold
                                  /             \
                                HIT             MISS
                                 │                │
                                 ▼                ▼
                            Return Cache     Redis Lock
                            Response             │
                                                ▼
                                         Re-check Cache
                                                │
                                      ┌─────────┴─────────┐
                                      │                   │
                                   Found              Still Missing
                                      │                   │
                                      ▼                   ▼
                                Return Cache        LLM Provider
                                                        │
                                                        ▼
                                                  Store Response
                                                        │
                                                        ▼
                                                  Release Lock
                                                        │
                                                        ▼
                                                  Return Response
```

---

# 5. Cache Lookup Strategy

The gateway will use a two-stage lookup.

## Stage 1 — Exact Match

Generate a deterministic hash from the relevant original request representation.

For example:

```text
request
   ↓
canonical serialization
   ↓
SHA-256
   ↓
original_request_hash
```

Check the exact hash before generating an embedding.

This provides a cheaper fast path for genuinely identical requests.

---

## Stage 2 — Semantic Match

If there is no exact match:

```text
Original Request
       ↓
Canonicalization
       ↓
Embedding
       ↓
Qdrant
       ↓
Similarity Search
```

If:

```text
similarity >= threshold
```

return the cached response.

Otherwise:

```text
Cache Miss
   ↓
LLM Provider
   ↓
Store Result
```

---

# 6. Cache Entry Schema

Each cache entry should contain:

```text
CacheEntry
├── id
├── embedding
├── normalized_request
├── original_request_hash
├── response
├── model
├── model_version
├── tenant_id
├── created_at
├── expires_at
└── metadata
```

### Field purposes

| Field | Purpose |
|---|---|
| `embedding` | Semantic representation for vector search |
| `normalized_request` | Canonicalized text used for comparison |
| `original_request_hash` | Exact-match lookup |
| `response` | Previously generated LLM response |
| `model` | Provider/model associated with response |
| `model_version` | Prevent incompatible cached responses |
| `tenant_id` | Prevent cross-tenant leakage |
| `created_at` | Cache creation time |
| `expires_at` | TTL management |
| `metadata` | Additional request/cache information |

---

# 7. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11+ | Main development language |
| API | FastAPI | Gateway/API |
| LLM Provider | OpenRouter | Initial provider |
| Secondary Provider | Groq | Provider fallback |
| Embeddings | sentence-transformers | Local embeddings |
| Embedding Model | `all-MiniLM-L6-v2` | Fast/free development model |
| Vector Database | Qdrant | Semantic search |
| Locking | Redis | Single-flight/concurrency control |
| Experiment Tracking | MLflow | Evaluation tracking |
| Dashboard | Streamlit | Metrics visualization |
| Testing | Pytest | Automated tests |
| HTTP | httpx | Async provider requests |
| Containers | Docker + Docker Compose | Local deployment |
| Version Control | Git + GitHub | Source control |

---

# 8. MVP Scope

## Required

- FastAPI gateway
- OpenAI-compatible request structure
- OpenRouter integration
- Local embedding model
- Qdrant
- Exact request hashing
- Semantic similarity search
- Configurable similarity threshold
- Cache hit/miss logic
- Cache storage
- Basic evaluation dataset
- Precision/recall evaluation
- Threshold sweep
- MLflow tracking
- Latency benchmark
- Cost-saving estimation
- Docker Compose

## After Core MVP

- Redis single-flight locking
- Tenant isolation
- TTL
- PII bypass
- Groq provider
- Provider fallback
- Structured logging
- Dashboard

## Stretch

- Circuit breaker
- Advanced tracing
- Dynamic thresholds
- Freshness-aware caching
- Advanced eviction policies

---

# 9. Phase 0 — Project Setup

## Goal

Prepare the development environment.

### Tasks

1. Create GitHub repository.
2. Create Python virtual environment.
3. Create project structure.
4. Install initial dependencies.
5. Create `.env`.
6. Create `.env.example`.
7. Add `.gitignore`.
8. Initialize FastAPI.

Initial dependencies:

```text
fastapi
uvicorn
httpx
pydantic-settings
sentence-transformers
qdrant-client
```

### First endpoint

```text
GET /health
```

Expected:

```json
{
  "status": "healthy"
}
```

### Checkpoint

Do not continue until:

- FastAPI starts.
- `/health` works.
- Environment variables load correctly.
- Git repository is initialized.

---

# 10. Phase 1 — Basic LLM Gateway

## Goal

Make the gateway work as a simple LLM proxy.

### Step 1 — Request schema

Support a request similar to:

```json
{
  "model": "model-name",
  "messages": [
    {
      "role": "user",
      "content": "Explain machine learning."
    }
  ]
}
```

### Step 2 — Provider abstraction

Create:

```text
LLMProvider
    │
    └── complete(request)
```

Implement:

```text
OpenRouterProvider
```

### Step 3 — Forward request

The initial flow:

```text
Client
  ↓
FastAPI
  ↓
OpenRouter
  ↓
FastAPI
  ↓
Client
```

No caching yet.

### Checkpoint

A client should be able to send an LLM request to your gateway and receive a valid response.

---

# 11. Phase 2 — Exact Request Hashing

## Goal

Implement the cheapest cache lookup before semantic search.

### Step 1

Define how the relevant request is serialized consistently.

### Step 2

Generate:

```text
original_request_hash
```

using SHA-256.

### Step 3

Check the exact hash before embedding.

Flow:

```text
Request
  ↓
Hash
  ↓
Exact Cache Lookup
  ↓
Hit → Return
Miss → Continue
```

### Checkpoint

Send exactly the same request twice.

Expected:

```text
Request 1 → LLM call
Request 2 → Exact cache hit
```

The second request should not generate an embedding if the exact cache path is implemented as intended.

---

# 12. Phase 3 — Canonicalization

## Goal

Create a normalized representation specifically for semantic comparison.

### Canonicalization may include

- lowercasing
- whitespace normalization
- simple punctuation normalization

Example:

```text
"Explain   MACHINE learning!!"
```

becomes:

```text
"explain machine learning"
```

### Important rule

The canonicalized text is used for:

```text
Embedding
Semantic comparison
```

It is **not** used for:

```text
LLM request
```

The original request remains untouched.

### Checkpoint

Verify through logging/tests that:

```text
Original Request ≠ Modified Request
```

when the original contains meaningful formatting.

---

# 13. Phase 4 — Embedding Service

## Goal

Generate local embeddings.

Use:

```text
all-MiniLM-L6-v2
```

Create:

```text
EmbeddingService
    └── embed(text)
```

Input:

```text
canonicalized request
```

Output:

```text
embedding vector
```

### Measure

Record embedding generation time.

This is important because:

```text
Embedding latency + Qdrant latency
```

must still provide a useful advantage over making an LLM call.

### Checkpoint

Confirm:

- model loads successfully
- embeddings have consistent dimensions
- repeated requests work
- latency can be measured

---

# 14. Phase 5 — Qdrant Integration

## Goal

Store and search semantic cache entries.

### Step 1

Run Qdrant locally with Docker.

### Step 2

Create the cache collection.

### Step 3

Insert test embeddings.

### Step 4

Perform similarity search.

Example:

```text
Stored:
"What is machine learning?"

Query:
"Can you explain machine learning?"
```

The stored request should receive a high similarity score.

### Checkpoint

You should have:

```text
Text
 ↓
Embedding
 ↓
Qdrant
 ↓
Similarity Score
```

working independently.

---

# 15. Phase 6 — Semantic Cache Core

## Goal

Connect everything into the actual caching system.

### Cache miss

```text
Request
 ↓
Hash lookup
 ↓
No exact match
 ↓
Canonicalize
 ↓
Embedding
 ↓
Qdrant search
 ↓
No sufficiently similar result
 ↓
LLM call
 ↓
Store response
 ↓
Return response
```

### Cache hit

```text
Request
 ↓
Hash lookup
 ↓
No exact match
 ↓
Canonicalize
 ↓
Embedding
 ↓
Qdrant search
 ↓
Similarity >= threshold
 ↓
Return cached response
```

### Checkpoint

Test:

#### First request

```text
"What is machine learning?"
```

Expected:

```text
Cache Miss
LLM Called
Response Stored
```

#### Second request

```text
"Can you explain machine learning?"
```

Expected:

```text
Semantic Cache Hit
LLM Not Called
```

This is the first major project milestone.

---

# 16. Phase 7 — Cache Correctness

## Goal

Prevent incorrect semantic matches.

Create test cases containing:

### Positive pairs

```text
"What is machine learning?"
"Explain machine learning."
```

### Negative pairs

```text
"What is Python?"
"How do I install Python on Windows?"
```

### Borderline pairs

```text
"What is a neural network?"
"How do neural networks learn?"
```

The system should not assume that semantic similarity automatically means the responses are interchangeable.

---

# 17. Phase 8 — Evaluation Dataset

## Goal

Create a labeled dataset to measure semantic-cache quality.

Each example should contain:

```text
query_a
query_b
expected_match
```

Example:

| Query A | Query B | Expected |
|---|---|---|
| What is ML? | Explain machine learning | Match |
| What is Python? | How do I install Python? | No match |
| What is a neural network? | Explain neural networks | Match |

Create an initial dataset of approximately:

```text
100–200 labeled pairs
```

A small manually verified dataset is preferable to a large unreliable dataset.

---

# 18. Phase 9 — Threshold Evaluation

## Goal

Determine the similarity threshold using actual evaluation.

Test thresholds such as:

```text
0.75
0.80
0.85
0.90
0.95
```

For each threshold calculate:

```text
True Positives
False Positives
True Negatives
False Negatives
```

Then calculate:

```text
Precision
Recall
F1 Score
```

Example:

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.75 | X | X | X |
| 0.80 | X | X | X |
| 0.85 | X | X | X |
| 0.90 | X | X | X |
| 0.95 | X | X | X |

Select the threshold based on measured results.

Do not select it arbitrarily.

---

# 19. Phase 10 — MLflow

## Goal

Make threshold experiments reproducible.

Each MLflow run should record:

### Parameters

```text
threshold
embedding_model
dataset_version
```

### Metrics

```text
precision
recall
f1
hit_rate
average_latency
```

Example:

```text
Run 1
threshold = 0.80
precision = X
recall = X

Run 2
threshold = 0.85
precision = X
recall = X
```

The final README should show the evaluation results.

---

# 20. Phase 11 — Performance Benchmarking

## Goal

Demonstrate that semantic caching provides measurable benefit.

Compare:

### A. Direct LLM

```text
Client → LLM
```

### B. Cache Miss

```text
Client → Gateway → Embedding → Qdrant → LLM
```

### C. Cache Hit

```text
Client → Gateway → Embedding → Qdrant → Cached Response
```

Measure:

```text
Average latency
P50
P95
```

Also measure:

```text
LLM calls avoided
Estimated tokens avoided
Estimated cost reduction
```

---

# 21. Phase 12 — Redis Single-Flight Locking

## Goal

Prevent cache stampedes.

Without locking:

```text
Request A ─┐
Request B ─┼── Cache Miss
Request C ─┘

All call LLM
```

With single-flight:

```text
Request A → Lock → LLM
Request B → Wait
Request C → Wait

LLM Response
     ↓
Store Cache
     ↓
Release Lock
     ↓
B/C use cached result
```

Use Redis for:

- distributed lock
- lock timeout
- lock recovery

### Important behavior

After acquiring the lock, the request should **check the cache again**.

Why?

Another request may have populated the cache while this request was waiting for the lock.

### Checkpoint

Send several concurrent identical requests.

Verify that only one request reaches the LLM.

---

# 22. Phase 13 — Multi-Tenant Isolation

## Goal

Prevent cached responses from leaking between users/tenants.

Every entry must contain:

```text
tenant_id
```

Semantic search must filter using:

```text
tenant_id
model
model_version
```

**before similarity ranking.**

Correct:

```text
Tenant filter
     ↓
Similarity search
     ↓
Best result
```

Incorrect:

```text
Similarity search across everyone
     ↓
Best result
     ↓
Check tenant afterward
```

### Checkpoint

Create:

```text
Tenant A
Tenant B
```

Verify that Tenant A cannot retrieve Tenant B's cached response.

---

# 23. Phase 14 — Model and Version Isolation

## Goal

Prevent incompatible responses from being returned.

Cache entries must include:

```text
model
model_version
```

For example:

```text
GPT-model-A
```

should not automatically satisfy a request intended for:

```text
GPT-model-B
```

unless explicitly supported by the cache policy.

---

# 24. Phase 15 — TTL

## Goal

Prevent stale responses from remaining indefinitely.

Add:

```text
created_at
expires_at
```

Before returning a cached response:

```text
Is expired?
   │
   ├── Yes → Cache Miss
   │
   └── No → Continue
```

Make TTL configurable through environment variables.

---

# 25. Phase 16 — PII Cache Policy

## Goal

Reduce the risk of storing sensitive requests and responses.

Implement a lightweight PII eligibility check.

Potential patterns include:

- email addresses
- phone numbers
- API keys
- obvious sensitive identifiers

### Policy

If high-risk PII is detected:

```text
PII detected
     ↓
Bypass cache
     ↓
Call LLM
     ↓
Do not store cache entry
```

### Why bypass instead of redact-and-cache?

Redaction can cause different users' requests to become semantically indistinguishable.

For example:

```text
"Write an email to John Smith about his account."
```

and:

```text
"Write an email to Ahmed Khan about his account."
```

could both become:

```text
"Write an email to [PERSON] about his account."
```

A cached response generated for one person could then become eligible for another request.

Therefore, v1 uses the safer policy:

> **If high-risk PII is detected, bypass caching rather than attempting to redact and cache the request.**

This is a conservative v1 policy, not a claim that the detector provides complete PII protection.

---

# 26. Phase 17 — Provider Abstraction

## Goal

Support more than one LLM provider.

Create:

```text
LLMProvider
    │
    ├── OpenRouterProvider
    │
    └── GroqProvider
```

Both should implement the same interface.

The rest of the application should not need to know which provider is being used.

---

# 27. Phase 18 — Provider Fallback

## Goal

Provide graceful behavior when a provider fails or becomes unavailable.

Example:

```text
Primary Provider
      │
      ├── Success → Return
      │
      └── Failure
            ↓
      Fallback Provider
            │
            ▼
          Return
```

Initial configuration:

```text
Primary: Groq
Fallback: OpenRouter
```

or the reverse depending on current availability/configuration.

The order should be configurable rather than hardcoded.

### Handle

- timeout
- provider errors
- temporary failures
- exhausted quota

Do not implement a complex circuit breaker in the first version.

---

# 28. Phase 19 — Observability

## Goal

Understand what the gateway is doing.

Every request should have a:

```text
request_id
```

Where applicable, track:

```text
request_id
cache_hit
cache_entry_id
tenant_id
model
similarity_score
latency
provider
```

Core metrics:

```text
total requests
cache hits
cache misses
hit rate
cache hit latency
cache miss latency
LLM latency
embedding latency
Qdrant latency
```

Later:

```text
P50
P95
P99
```

---

# 29. Phase 20 — Cost Analysis

## Goal

Estimate the economic value of caching.

Track:

```text
LLM calls avoided
tokens avoided
estimated cost avoided
```

For a test workload:

```text
Without cache:
100 LLM calls

With cache:
40 LLM calls
```

Then calculate:

```text
Calls avoided = 60%
```

Use actual provider pricing/configuration when calculating cost.

Clearly label estimates as estimates.

---

# 30. Phase 21 — Streamlit Dashboard

## Goal

Create a lightweight visualization layer.

Dashboard should show:

### Request metrics

- total requests
- cache hits
- cache misses
- cache hit rate

### Performance

- cache hit latency
- cache miss latency
- LLM latency
- P50/P95

### Economics

- LLM calls avoided
- estimated tokens avoided
- estimated cost savings

### Evaluation

- selected threshold
- precision
- recall
- F1
- threshold comparison

The dashboard is secondary to the gateway itself.

---

# 31. Phase 22 — Docker Compose

## Goal

Make the complete project easy to run locally.

Services:

```text
FastAPI
Qdrant
Redis
MLflow
Streamlit
```

Target:

```bash
docker compose up --build
```

Create:

```text
Dockerfile
docker-compose.yml
.env.example
```

The project should work without requiring complicated manual setup.

---

# 32. Testing Strategy

## Unit Tests

Test:

- request hashing
- canonicalization
- threshold logic
- cache hit logic
- cache miss logic
- TTL
- PII detection
- provider selection
- provider fallback

## Integration Tests

Test:

```text
FastAPI
 ↓
Embedding
 ↓
Qdrant
 ↓
LLM Provider
```

## Security Tests

Test:

```text
Tenant A → Tenant A cache only
Tenant B → Tenant B cache only
```

## Concurrency Tests

Send multiple simultaneous requests for the same uncached request.

Expected:

```text
1 LLM call
N responses
```

rather than:

```text
N LLM calls
```

---

# 33. Important Edge Cases

The system should handle:

### Exact duplicate

```text
Same request
→ exact hash hit
```

### Semantic duplicate

```text
Different wording
→ semantic hit
```

### Unrelated request

```text
Low similarity
→ cache miss
```

### Expired cache

```text
Expired
→ cache miss
```

### PII request

```text
PII detected
→ bypass cache
```

### Provider failure

```text
Provider A fails
→ Provider B
```

### Concurrent miss

```text
Multiple identical misses
→ single LLM call
```

### Different tenants

```text
Tenant A
≠
Tenant B
```

### Different models

```text
Model A
≠
Model B
```

---

# 34. Evaluation Limitations

The project must explicitly acknowledge:

### Semantic similarity does not guarantee correctness.

For example:

```text
"What is the weather in Lahore?"
```

asked at two different times may be semantically identical but require different answers.

Therefore:

```text
Semantic match ≠ response correctness
Semantic match ≠ freshness
```

The v1 system will not attempt to solve freshness automatically.

Freshness-aware caching can be future work.

---

# 35. Final Architecture

The completed system should resemble:

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

---

# 36. Recommended Build Order

Follow this order strictly:

```text
PHASE 0
Project Setup
      ↓
PHASE 1
Basic LLM Gateway
      ↓
PHASE 2
Exact Request Hashing
      ↓
PHASE 3
Canonicalization
      ↓
PHASE 4
Local Embeddings
      ↓
PHASE 5
Qdrant
      ↓
PHASE 6
Semantic Cache Core
      ↓
PHASE 7
Cache Correctness Tests
      ↓
PHASE 8
Evaluation Dataset
      ↓
PHASE 9
Threshold Evaluation
      ↓
PHASE 10
MLflow
      ↓
PHASE 11
Latency + Cost Benchmark
      ↓
PHASE 12
Redis Single-Flight
      ↓
PHASE 13
Tenant Isolation
      ↓
PHASE 14
Model/Version Isolation
      ↓
PHASE 15
TTL
      ↓
PHASE 16
PII Bypass
      ↓
PHASE 17
Provider Abstraction
      ↓
PHASE 18
Provider Fallback
      ↓
PHASE 19
Observability
      ↓
PHASE 20
Cost Analysis
      ↓
PHASE 21
Dashboard
      ↓
PHASE 22
Docker Compose
      ↓
FINAL
README + Benchmarks + Resume
```

---

# 37. Definition of Done

The project is complete when:

## Core System

- [ ] FastAPI gateway works.
- [ ] LLM provider integration works.
- [ ] Exact request hashing works.
- [ ] Canonicalization works.
- [ ] Original request remains unchanged.
- [ ] Local embeddings work.
- [ ] Qdrant semantic search works.
- [ ] Semantic cache hits work.
- [ ] Cache misses call the LLM.
- [ ] Responses are stored correctly.

## Correctness

- [ ] Similarity threshold is configurable.
- [ ] Positive/negative semantic pairs are evaluated.
- [ ] Precision is measured.
- [ ] Recall is measured.
- [ ] F1 is measured.
- [ ] Threshold selection is evidence-based.
- [ ] Model/version isolation works.
- [ ] TTL works.

## Reliability

- [ ] Redis locking works.
- [ ] Concurrent requests are coalesced.
- [ ] Provider fallback works.
- [ ] Provider timeout handling works.

## Security

- [ ] Tenant isolation works.
- [ ] PII requests bypass caching.
- [ ] Secrets are stored through environment variables.
- [ ] No API keys are committed to Git.

## Evaluation

- [ ] Evaluation dataset exists.
- [ ] Threshold sweep is completed.
- [ ] MLflow tracks experiments.
- [ ] Latency benchmark is completed.
- [ ] Cost-saving estimate is completed.

## Deployment

- [ ] Dockerfile works.
- [ ] Docker Compose works.
- [ ] `.env.example` exists.
- [ ] Health check exists.

## Documentation

- [ ] README explains architecture.
- [ ] Setup instructions work.
- [ ] Evaluation instructions work.
- [ ] Benchmark results are documented.
- [ ] Architecture diagram is included.
- [ ] Limitations are documented.
- [ ] MLflow evaluation screenshot is included.

---

# 38. Final Resume Outcome

The final project should allow a result-based resume description such as:

> **Built a semantic caching gateway for LLM APIs using FastAPI, Qdrant, Redis, and local embeddings, implementing exact-match and semantic caching, tenant isolation, request coalescing, and provider fallback; evaluated similarity thresholds using MLflow and achieved [X]% cache precision, [Y]% cache hit rate, and [Z]% estimated reduction in LLM API costs.**

The values must come from the project's actual experiments.

---

# 39. Success Criteria

The project is successful if it demonstrates three things:

### 1. It works

A real application can send requests through the gateway and receive cached or newly generated responses.

### 2. It is correct

The system demonstrates controlled semantic matching, tenant isolation, concurrency protection, and documented limitations.

### 3. It has measurable value

The final evaluation provides actual evidence for:

```text
Cache Quality
+
Latency Reduction
+
LLM Calls Avoided
+
Estimated Cost Savings
```

The final deliverable is therefore **not simply a vector database wrapper**.

It is a measured, testable **LLM infrastructure component**.
