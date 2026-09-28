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
from app.db.qdrant import QdrantService
from app.embeddings.service import get_embedding_service
from app.main import app
from app.providers.base import LLMProvider


class TTLCountingMockLLMProvider(LLMProvider):
    """
    Mock LLM provider that increments a counter on every call to track
    when upstream queries happen versus cache hits.
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.call_count += 1
        prompt = request.messages[-1].content
        return ChatCompletionResponse(
            id=f"chatcmpl-ttl-{self.call_count}",
            model=request.model,
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=f"LLM Generation #{self.call_count} for: '{prompt}'",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=15, completion_tokens=25, total_tokens=40),
        )


@pytest.mark.asyncio
async def test_prd_phase15_checkpoint_exact_cache_ttl_expiration():
    """
    PRD Phase 15 Checkpoint:
    Verify that an exact cache entry is served while within its TTL,
    and cleanly expires (producing a cache MISS and re-fetching from LLM) once TTL elapses.
    """
    mock_provider = TTLCountingMockLLMProvider()

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
            payload = {
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [{"role": "user", "content": "What is the current Bitcoin price?"}],
            }

            # 1. Request with 1-second TTL -> Cache MISS, calls LLM (Call #1)
            resp1 = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"X-Cache-TTL": "1"},
            )
            assert resp1.status_code == 200
            assert resp1.headers["X-Cache"] == "MISS"
            assert resp1.headers["X-Cache-TTL"] == "1"
            assert mock_provider.call_count == 1
            body1 = resp1.json()["choices"][0]["message"]["content"]
            assert "Generation #1" in body1

            # 2. Immediate Repeat (within 1-second TTL) -> Cache HIT-EXACT (No LLM call)
            resp2 = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"X-Cache-TTL": "1"},
            )
            assert resp2.status_code == 200
            assert resp2.headers["X-Cache"] == "HIT-EXACT"
            assert mock_provider.call_count == 1  # Still 1
            body2 = resp2.json()["choices"][0]["message"]["content"]
            assert body2 == body1

            # 3. Wait for TTL to expire (> 1.1s)
            await asyncio.sleep(1.2)

            # 4. Repeat after expiration -> MUST be Cache MISS! Calls LLM (Call #2)
            resp3 = await client.post(
                "/v1/chat/completions",
                json=payload,
                headers={"X-Cache-TTL": "1"},
            )
            assert resp3.status_code == 200
            assert resp3.headers["X-Cache"] == "MISS"
            assert mock_provider.call_count == 2  # New LLM call triggered!
            body3 = resp3.json()["choices"][0]["message"]["content"]
            assert "Generation #2" in body3
            assert body3 != body1

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_prd_phase15_checkpoint_semantic_cache_ttl_expiration():
    """
    Verify that semantic similarity lookup respects TTL expiration and ignores stale vectors.
    """
    mock_provider = TTLCountingMockLLMProvider()

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
            prompt_base = "What is the latest stable release of Python?"
            prompt_paraphrase = "Tell me the latest stable release of Python."

            payload_base = {
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [{"role": "user", "content": prompt_base}],
            }
            payload_paraphrase = {
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [{"role": "user", "content": prompt_paraphrase}],
            }

            # 1. Base query with 1-second TTL -> Cache MISS (Call #1)
            resp1 = await client.post(
                "/v1/chat/completions",
                json=payload_base,
                headers={"X-Cache-TTL": "1"},
            )
            assert resp1.status_code == 200
            assert resp1.headers["X-Cache"] == "MISS"
            assert mock_provider.call_count == 1

            # 2. Immediate paraphrase (within 1s) -> Cache HIT-SEMANTIC
            resp2 = await client.post(
                "/v1/chat/completions",
                json=payload_paraphrase,
                headers={"X-Cache-TTL": "1"},
            )
            assert resp2.status_code == 200
            assert resp2.headers["X-Cache"] == "HIT-SEMANTIC"
            assert mock_provider.call_count == 1

            # 3. Wait for TTL to expire
            await asyncio.sleep(1.2)

            # 4. Paraphrase query after expiration -> MUST be Cache MISS!
            resp3 = await client.post(
                "/v1/chat/completions",
                json=payload_paraphrase,
                headers={"X-Cache-TTL": "1"},
            )
            assert resp3.status_code == 200
            assert resp3.headers["X-Cache"] == "MISS"
            assert mock_provider.call_count == 2  # Re-fetched from upstream!

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_cache_control_header_parsing():
    """
    Verify that Cache-Control max-age header is correctly parsed into TTL seconds.
    """
    mock_provider = TTLCountingMockLLMProvider()

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
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta-llama/llama-3-8b-instruct",
                    "messages": [{"role": "user", "content": "Cache control test"}],
                },
                headers={"Cache-Control": "max-age=7200"},
            )
            assert resp.status_code == 200
            assert resp.headers["X-Cache-TTL"] == "7200"

    finally:
        app.dependency_overrides.clear()
