import asyncio
import time
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    Usage,
)
from app.cache.exact_cache import InMemoryExactCache
from app.cache.semantic_cache import SemanticCache, get_semantic_cache
from app.core.security import get_tenant_id
from app.db.qdrant import QdrantService
from app.embeddings.service import get_embedding_service
from app.main import app
from app.providers.base import LLMProvider


class TenantAwareMockLLMProvider(LLMProvider):
    """
    Mock LLM provider that returns custom responses keyed by the caller's tenant
    and counts total upstream calls.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.history = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.call_count += 1
        prompt = request.messages[-1].content
        self.history.append((self.call_count, prompt))

        return ChatCompletionResponse(
            id=f"chatcmpl-call-{self.call_count}",
            model=request.model,
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=f"LLM Response #{self.call_count} for: '{prompt}'",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=15, completion_tokens=25, total_tokens=40),
        )


@pytest.mark.asyncio
async def test_qdrant_pre_filtering_tenant_isolation():
    """
    Direct test on QdrantService verifying that vector search with tenant_id pre-filtering
    strictly partitions vector matches.
    """
    qdrant = QdrantService(location=":memory:")
    qdrant.ensure_collection("test_isolation")
    embedder = get_embedding_service()

    vector_a = embedder.embed("How do I cancel my subscription?")
    vector_b = embedder.embed("How do I cancel my subscription?")

    import uuid
    point_id_a = str(uuid.uuid4())

    # Store entry for tenant_a
    qdrant.upsert_entry(
        point_id=point_id_a,
        vector=vector_a,
        payload={"tenant_id": "tenant_a", "normalized_request": "how do i cancel my subscription", "data": "A"},
        collection_name="test_isolation",
    )

    # Search as tenant_b for identical vector
    results_b = qdrant.search_nearest(
        query_vector=vector_b,
        limit=5,
        tenant_id="tenant_b",
        collection_name="test_isolation",
    )
    # Must NOT find tenant_a's point
    assert len(results_b) == 0

    # Search as tenant_a for the same vector
    results_a = qdrant.search_nearest(
        query_vector=vector_a,
        limit=5,
        tenant_id="tenant_a",
        collection_name="test_isolation",
    )
    # Must find tenant_a's point
    assert len(results_a) == 1
    assert results_a[0]["payload"]["tenant_id"] == "tenant_a"


@pytest.mark.asyncio
async def test_prd_phase13_checkpoint_multi_tenant_cache_isolation():
    """
    PRD Phase 13 Checkpoint:
    Create:
      Tenant A
      Tenant B
    Verify that Tenant A cannot retrieve Tenant B's cached response,
    and Tenant B cannot retrieve Tenant A's cached response.
    """
    mock_provider = TenantAwareMockLLMProvider()

    isolated_exact_cache = InMemoryExactCache()
    isolated_qdrant = QdrantService(location=":memory:")
    isolated_qdrant.ensure_collection()

    isolated_semantic_cache = SemanticCache(
        exact_cache=isolated_exact_cache,
        embedding_service=get_embedding_service(),
        qdrant_service=isolated_qdrant,
        threshold=0.75,
    )

    from app.api.routes import get_llm_provider
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    app.dependency_overrides[get_semantic_cache] = lambda: isolated_semantic_cache

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            prompt_original = "What is the corporate refund policy?"
            prompt_paraphrase = "Explain the corporate refund policy to me."

            payload_original = {
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [{"role": "user", "content": prompt_original}],
            }
            payload_paraphrase = {
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [{"role": "user", "content": prompt_paraphrase}],
            }

            # -------------------------------------------------------------
            # STEP 1: Tenant A sends original request -> Cache MISS
            # -------------------------------------------------------------
            resp_a1 = await client.post(
                "/v1/chat/completions",
                json=payload_original,
                headers={"X-Tenant-ID": "tenant_a"},
            )
            assert resp_a1.status_code == 200
            assert resp_a1.headers["X-Cache"] == "MISS"
            assert resp_a1.headers["X-Tenant-ID"] == "tenant_a"
            assert mock_provider.call_count == 1
            body_a1 = resp_a1.json()["choices"][0]["message"]["content"]
            assert "LLM Response #1" in body_a1

            # -------------------------------------------------------------
            # STEP 2: Tenant A sends paraphrase -> Cache HIT-SEMANTIC
            # -------------------------------------------------------------
            resp_a2 = await client.post(
                "/v1/chat/completions",
                json=payload_paraphrase,
                headers={"X-Tenant-ID": "tenant_a"},
            )
            assert resp_a2.status_code == 200
            assert resp_a2.headers["X-Cache"] == "HIT-SEMANTIC"
            assert resp_a2.headers["X-Tenant-ID"] == "tenant_a"
            # No new LLM call made!
            assert mock_provider.call_count == 1
            body_a2 = resp_a2.json()["choices"][0]["message"]["content"]
            # Tenant A gets back Response #1
            assert body_a2 == body_a1

            # -------------------------------------------------------------
            # STEP 3: Tenant B sends EXACT SAME prompt as Tenant A -> MUST MISS!
            # -------------------------------------------------------------
            resp_b1 = await client.post(
                "/v1/chat/completions",
                json=payload_original,
                headers={"X-Tenant-ID": "tenant_b"},
            )
            assert resp_b1.status_code == 200
            # CRITICAL CHECKPOINT: Must be MISS even though exact text was cached for Tenant A
            assert resp_b1.headers["X-Cache"] == "MISS"
            assert resp_b1.headers["X-Tenant-ID"] == "tenant_b"
            assert mock_provider.call_count == 2
            body_b1 = resp_b1.json()["choices"][0]["message"]["content"]
            # Tenant B gets Response #2, NOT Tenant A's cached response!
            assert "LLM Response #2" in body_b1
            assert body_b1 != body_a1

            # -------------------------------------------------------------
            # STEP 4: Tenant B sends paraphrase -> Cache HIT for Tenant B's data
            # -------------------------------------------------------------
            resp_b2 = await client.post(
                "/v1/chat/completions",
                json=payload_paraphrase,
                headers={"X-Tenant-ID": "tenant_b"},
            )
            assert resp_b2.status_code == 200
            assert resp_b2.headers["X-Cache"] == "HIT-SEMANTIC"
            assert resp_b2.headers["X-Tenant-ID"] == "tenant_b"
            assert mock_provider.call_count == 2
            body_b2 = resp_b2.json()["choices"][0]["message"]["content"]
            # Tenant B gets back Response #2
            assert body_b2 == body_b1

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_default_tenant_fallback():
    """Verify that requests without X-Tenant-ID header gracefully fall back to 'default'."""
    isolated_exact_cache = InMemoryExactCache()
    isolated_qdrant = QdrantService(location=":memory:")
    isolated_qdrant.ensure_collection()

    isolated_semantic_cache = SemanticCache(
        exact_cache=isolated_exact_cache,
        embedding_service=get_embedding_service(),
        qdrant_service=isolated_qdrant,
        threshold=0.75,
    )
    mock_provider = TenantAwareMockLLMProvider()

    from app.api.routes import get_llm_provider
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    app.dependency_overrides[get_semantic_cache] = lambda: isolated_semantic_cache

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta-llama/llama-3-8b-instruct",
                    "messages": [{"role": "user", "content": "Test default tenant"}],
                },
            )
            assert resp.status_code == 200
            # Header check: defaults to 'default'
            assert resp.headers["X-Tenant-ID"] == "default"
    finally:
        app.dependency_overrides.clear()

