# AGENTS.md

## Project: Semantic Caching Gateway for LLM APIs

You are an AI coding agent working on this repository.

Your job is to help build a production-minded portfolio project: a semantic caching gateway that sits between an application and an LLM provider.

The project is an engineering project, not a simple LLM demo.

The primary goals are:

- Correct semantic caching
- Measurable cache quality
- Low latency
- Reduced LLM calls and estimated cost
- Concurrency safety
- Tenant isolation
- Provider abstraction and fallback
- Reproducible evaluation
- Clean, maintainable Python code

---

# 1. Source of Truth

The project requirements are defined in the project PRD.

Before implementing a feature:

1. Read the relevant PRD section.
2. Understand its dependencies.
3. Check the current repository state.
4. Implement only the next appropriate step.
5. Run tests/checks.
6. Verify the implementation before moving forward.

Do not silently change the project's architecture or requirements.

If a requirement is ambiguous, make the smallest reasonable assumption and clearly state it.

Do not invent functionality that is not required.

---

# 2. Critical Development Rule

## Build incrementally.

DO NOT implement the entire project in one pass.

The project must be developed in phases.

Follow this order:

```text
Phase 0  → Project Setup
Phase 1  → Basic LLM Gateway
Phase 2  → Exact Request Hashing
Phase 3  → Canonicalization
Phase 4  → Local Embeddings
Phase 5  → Qdrant
Phase 6  → Semantic Cache Core
Phase 7  → Cache Correctness Tests
Phase 8  → Evaluation Dataset
Phase 9  → Threshold Evaluation
Phase 10 → MLflow
Phase 11 → Latency + Cost Benchmark
Phase 12 → Redis Single-Flight
Phase 13 → Tenant Isolation
Phase 14 → Model/Version Isolation
Phase 15 → TTL
Phase 16 → PII Bypass
Phase 17 → Provider Abstraction
Phase 18 → Provider Fallback
Phase 19 → Observability
Phase 20 → Cost Analysis
Phase 21 → Dashboard
Phase 22 → Docker Compose
Phase 23 → Documentation + Finalization
```

Do not skip phases unless explicitly instructed.

Do not implement advanced infrastructure while the core semantic cache is broken.

---

# 3. Current Phase Protocol

At the beginning of every coding task:

1. Determine the current project phase from the repository.
2. Inspect existing files before modifying anything.
3. Identify what has already been implemented.
4. Identify the smallest next milestone.
5. Implement that milestone.
6. Run appropriate tests.
7. Report:
   - what changed
   - files changed
   - tests/checks performed
   - any remaining issue

Never assume that a feature exists just because the PRD describes it.

---

# 4. Do Not Over-Engineer

This is a portfolio project.

Prefer:

```text
simple + correct + measurable
```

over:

```text
complex + impressive-looking + fragile
```

Do not add:

- unnecessary frameworks
- unnecessary abstractions
- unnecessary microservices
- Kubernetes
- Kafka
- Celery
- service meshes
- complex event-driven architecture
- custom vector database
- custom authentication system
- custom encryption
- unnecessary frontend infrastructure

unless explicitly requested.

The core system should remain understandable to one developer.

---

# 5. Technology Stack

Use the following technologies unless there is a strong technical reason to change them:

### Backend

- Python 3.11+
- FastAPI
- Pydantic
- httpx

### LLM

Initial provider:

- OpenRouter

Secondary provider:

- Groq

Provider access must be abstracted behind a common interface.

### Embeddings

Default:

```text
sentence-transformers
all-MiniLM-L6-v2
```

The embedding model runs locally.

Do not introduce a paid embedding API as a runtime dependency.

### Vector Database

```text
Qdrant
```

Run locally during development.

### Redis

Use Redis for:

- single-flight locks
- concurrency control
- future cache-related coordination

Do not use Redis as the semantic vector database.

### Experiment Tracking

```text
MLflow
```

Use MLflow for threshold and evaluation experiments.

### Dashboard

```text
Streamlit
```

Only implement the dashboard after the core system and evaluation are working.

### Deployment

```text
Docker
Docker Compose
```

### Testing

```text
pytest
```

---

# 6. Architecture

The target architecture is:

```text
Client
   │
   ▼
FastAPI Gateway
   │
   ▼
Request Validation
   │
   ▼
PII / Cache Eligibility
   │
   ▼
Request Hash
   │
   ▼
Exact Cache Lookup
   │
   ├──────── HIT ────────► Return Cached Response
   │
   ▼ MISS
Canonicalization
   │
   ▼
Embedding Service
   │
   ▼
Qdrant
   │
   ▼
Similarity Search
   │
   ├──── Similar ────────► Return Cached Response
   │
   ▼ No Suitable Match
Redis Single-Flight Lock
   │
   ▼
Re-check Cache
   │
   ▼
LLM Provider
   │
   ├──── Primary Provider
   │
   └──── Fallback Provider
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

# 7. Critical Correctness Rule: Original vs Canonicalized Request

This is one of the most important requirements.

There must be two separate representations.

## Original Request

The original request received from the client.

It must be preserved.

It is the request sent to the LLM.

## Canonicalized Request

A normalized representation used only for semantic cache comparison.

Example:

```text
Original:
"  Explain   MACHINE Learning!! "

Canonicalized:
"explain machine learning"
```

The canonicalized request MUST NOT replace the original request.

Never send the canonicalized request to the LLM simply because it was generated for embedding.

This distinction must remain explicit in the code.

Prefer variable names such as:

```python
original_request
normalized_request
embedding_text
```

rather than a generic variable such as:

```python
request_text
```

when the distinction matters.

---

# 8. Exact Hash Fast Path

Before generating an embedding, attempt an exact cache lookup.

Conceptually:

```text
Incoming Request
       │
       ▼
Canonical Request Serialization
       │
       ▼
SHA-256 Hash
       │
       ▼
Exact Cache Lookup
       │
       ├── HIT → Return
       │
       └── MISS → Semantic Search
```

The cache entry should contain:

```text
original_request_hash
```

The exact hash should be deterministic.

Do not hash Python object representations that may be unstable.

Use a deterministic serialization of the relevant request fields.

---

# 9. Cache Entry

The target cache entry is:

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

Do not add fields without a reason.

---

# 10. Semantic Matching

The semantic cache uses cosine similarity.

Initial flow:

```text
Request
   ↓
Canonicalize
   ↓
Embedding
   ↓
Qdrant Search
   ↓
Top Result
   ↓
Similarity Score
   ↓
Threshold Decision
```

If:

```text
similarity >= threshold
```

then:

```text
cache hit
```

Otherwise:

```text
cache miss
```

The threshold must be configurable.

Do not permanently hardcode a threshold such as `0.85`.

The final threshold must be selected using evaluation.

---

# 11. Cache Correctness

Never assume:

```text
high semantic similarity = identical answer
```

Semantic similarity is only evidence that two requests are related.

The evaluation must contain:

### Positive examples

Requests that should reuse the same response.

### Negative examples

Requests that are related but should not reuse the same response.

### Borderline examples

Requests where similarity alone may be ambiguous.

The system should prioritize avoiding harmful false cache hits.

---

# 12. Model Isolation

Cache lookup must consider the requested model.

At minimum:

```text
tenant_id
model
model_version
```

must be part of cache filtering.

Do not search globally and filter afterward.

Correct:

```text
Filter
  ↓
Similarity Search
  ↓
Best Match
```

Incorrect:

```text
Similarity Search
  ↓
Best Match
  ↓
Check Tenant/Model
```

---

# 13. Tenant Isolation

Every cache entry must be associated with a tenant.

All cache reads and writes must be scoped to the correct tenant.

Never allow:

```text
Tenant A
   ↓
retrieve
   ↓
Tenant B response
```

The tenant filter must be applied before semantic ranking.

Tenant isolation is a correctness and security requirement, not an optional optimization.

---

# 14. PII Policy

The initial PII policy is:

```text
PII detected
    ↓
Bypass cache
    ↓
Call LLM
    ↓
Do not store cache entry
```

Do not implement sophisticated redaction/reconstruction unless explicitly requested.

### Reason

Redaction can make distinct requests semantically indistinguishable.

Example:

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

This could cause a response generated for one person to be reused for another.

Therefore, v1 prefers bypassing cache for high-risk PII rather than attempting to redact and cache.

The PII detector is not assumed to be perfect.

---

# 15. Concurrency / Single-Flight

A cache miss must not cause a cache stampede.

Without protection:

```text
Request A ─┐
Request B ─┼── Cache Miss → LLM
Request C ─┘              → LLM
```

With single-flight:

```text
Request A → Lock → LLM → Store
Request B → Wait
Request C → Wait
                 ↓
             Cache Result
```

Use Redis.

After acquiring the lock, ALWAYS re-check the cache.

Reason:

Another request may have populated the cache before the lock was acquired.

Implement:

- lock acquisition
- lock timeout
- lock release
- crash recovery
- second cache check

---

# 16. Provider Abstraction

Create a common provider interface.

Conceptually:

```python
class LLMProvider:
    async def complete(...):
        ...
```

Implement:

```text
OpenRouterProvider
GroqProvider
```

The gateway should not contain provider-specific logic.

Provider-specific code belongs in the provider layer.

---

# 17. Provider Fallback

Provider failures should be handled gracefully.

Example:

```text
Primary Provider
      │
      ├── Success → Response
      │
      └── Failure
            ↓
      Fallback Provider
            │
            ▼
         Response
```

Provider order must be configurable.

Do not hardcode assumptions about free-tier availability.

Handle:

- timeout
- temporary errors
- unavailable provider
- exhausted quota where detectable

Do not implement a circuit breaker unless explicitly requested.

---

# 18. Freshness

The v1 system does NOT solve freshness.

A semantically identical request may require a new answer.

Example:

```text
"What is the weather in Lahore?"
```

A cached response from yesterday may be incorrect today.

Document:

```text
semantic match ≠ correctness
semantic match ≠ freshness
```

Do not silently implement freshness logic unless it is explicitly added as a future feature.

---

# 19. Evaluation

Evaluation is a core part of the project.

It is not optional polish.

Create a labeled dataset containing:

```text
query_a
query_b
expected_match
```

Evaluate:

```text
true positives
false positives
true negatives
false negatives
```

Calculate:

```text
precision
recall
F1
```

Run threshold sweeps.

Example:

```text
0.75
0.80
0.85
0.90
0.95
```

The final threshold must be supported by evaluation results.

Never invent benchmark numbers.

---

# 20. MLflow

Every evaluation run should record:

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

Evaluation should be reproducible.

---

# 21. Benchmarking

Measure:

### Direct LLM

```text
Client → LLM
```

### Cache Miss

```text
Client → Gateway → Embedding → Qdrant → LLM
```

### Cache Hit

```text
Client → Gateway → Embedding → Qdrant → Cached Response
```

Measure:

```text
average latency
P50
P95
```

Also measure:

```text
LLM calls avoided
estimated tokens avoided
estimated cost savings
```

Never claim a performance improvement without measurement.

---

# 22. Testing Requirements

Use pytest.

At minimum, tests should cover:

### Hashing

- deterministic hash
- same request → same hash
- different request → different hash

### Canonicalization

- whitespace normalization
- case normalization
- original request remains unchanged

### Cache

- exact hit
- semantic hit
- semantic miss
- threshold behavior

### TTL

- valid entry
- expired entry

### Tenant isolation

- tenant A cannot retrieve tenant B's entry

### Model isolation

- model A cannot retrieve model B's response

### PII

- detected PII bypasses cache

### Provider

- successful provider call
- provider failure
- fallback provider

### Concurrency

- concurrent identical misses produce one upstream LLM call

---

# 23. Code Quality Rules

Prefer:

- small functions
- clear names
- type hints
- async I/O where appropriate
- dependency injection where useful
- configuration through environment variables
- explicit error handling
- testable components

Avoid:

- giant files
- giant functions
- hidden global state
- hardcoded API keys
- duplicated provider logic
- unnecessary abstractions
- unexplained magic numbers

Similarity thresholds and TTL values should be configuration values.

---

# 24. Security Rules

Never:

- commit API keys
- hardcode secrets
- log API keys
- log sensitive request content unnecessarily
- bypass tenant filtering
- return cached data across tenants

Use:

```text
.env
```

for local secrets.

Commit:

```text
.env.example
```

but never `.env`.

---

# 25. Dependency Management

Only add a dependency when it provides clear value.

Before adding a library:

1. Check whether existing dependencies can solve the problem.
2. Prefer standard library functionality for simple tasks.
3. Avoid adding large frameworks for small features.
4. Update dependency files consistently.

Do not install random libraries during debugging without understanding why they are needed.

---

# 26. Error Handling

The gateway should fail gracefully.

Possible failures:

- embedding failure
- Qdrant unavailable
- Redis unavailable
- LLM provider timeout
- LLM provider error
- malformed request

Do not allow internal stack traces or secrets to leak through API responses.

For optional infrastructure, define sensible fallback behavior where appropriate.

Do not silently hide serious failures.

---

# 27. Logging

Use structured logs where practical.

Important identifiers:

```text
request_id
cache_entry_id
tenant_id
model
provider
cache_hit
similarity_score
```

Do not log sensitive request/response content by default.

---

# 28. Docker

The final local stack should include:

```text
FastAPI
Qdrant
Redis
MLflow
Streamlit
```

The target experience is:

```bash
docker compose up --build
```

The project should be reproducible on another machine after environment configuration.

---

# 29. Dashboard

The dashboard is secondary.

Do not build it before the gateway and evaluation work.

Display:

- total requests
- cache hits
- cache misses
- hit rate
- cache latency
- LLM latency
- P50/P95
- calls avoided
- estimated cost savings
- threshold evaluation results

---

# 30. Documentation

The README must eventually contain:

1. Project overview
2. Problem
3. Architecture
4. Request flow
5. Technology stack
6. Setup instructions
7. Environment variables
8. Running the system
9. Running tests
10. Running evaluation
11. MLflow usage
12. Benchmark results
13. Limitations
14. Future work

Do not write fake benchmark results.

Use placeholders until real experiments exist.

---

# 31. Git Practices

Make small, logical commits.

Good examples:

```text
feat: add FastAPI gateway
feat: add OpenRouter provider
feat: add request hashing
feat: add canonicalization
feat: integrate Qdrant semantic search
test: add semantic cache tests
feat: add threshold evaluation
feat: add MLflow tracking
feat: add Redis single-flight locking
```

Avoid giant commits such as:

```text
finished entire project
```

Do not rewrite unrelated code while implementing a feature.

---

# 32. Agent Behavior

When asked to implement a feature:

### First

Inspect the repository.

### Then

Explain briefly:

```text
Current state
Target change
Files likely affected
```

### Then

Implement.

### Then

Run tests.

### Then

Report:

```text
Implemented:
- ...

Tests:
- ...

Result:
- ...

Next logical step:
- ...
```

Do not ask unnecessary questions.

If the requirement is clear, proceed.

If a decision materially affects architecture or correctness, ask before making a large irreversible change.

---

# 33. Do Not Fake Completion

Never say:

```text
implemented
```

unless the code actually exists and has been checked.

Never claim:

```text
tests pass
```

without running the tests.

Never claim:

```text
X% latency reduction
```

without benchmark evidence.

Never claim:

```text
secure
```

as an absolute statement.

Use evidence-based language.

---

# 34. Phase Completion Rule

A phase is complete only when:

1. Its implementation exists.
2. Relevant tests pass.
3. The feature can be demonstrated.
4. No obvious regression exists.
5. The next phase can build on it cleanly.

If a phase is broken, fix it before moving forward.

---

# 35. First Task

When starting from an empty repository, DO NOT build the complete project.

Start with:

```text
Phase 0 — Project Setup
```

Perform only:

1. Create project structure.
2. Create Python environment/dependency configuration.
3. Create FastAPI application.
4. Add `/health`.
5. Add configuration management.
6. Add `.env.example`.
7. Add `.gitignore`.
8. Add basic pytest setup.
9. Run the application.
10. Run the test suite.

After that, stop and report the result.

The next task should be Phase 1.

---

# 36. Definition of Done

The final project should provide evidence for:

```text
Correctness
+
Performance
+
Reliability
+
Security
+
Evaluation
```

The final resume claims must be based on actual measurements from the completed system.

The project should be understandable by another engineer who reads the repository without needing the AI agent to explain hidden decisions.

---

# 37. Final Principle

The goal is NOT:

> Build as many technologies as possible.

The goal is:

> Build a correct semantic caching system, measure whether it works, identify its limitations, and demonstrate the engineering decisions with evidence.

Prefer a smaller working system over a larger unfinished system.
