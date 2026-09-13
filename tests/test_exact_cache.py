import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_exact_cache, get_llm_provider
from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)
from app.cache.exact_cache import InMemoryExactCache
from app.cache.models import CacheEntry
from app.cache.semantic_cache import SemanticCache, get_semantic_cache
from app.db.qdrant import QdrantService
from app.embeddings.service import get_embedding_service
from app.main import app
from app.providers.base import LLMProvider

client = TestClient(app)

TEST_MODEL = "dots-studio/dots-3-note-preview:free"


class CountingMockProvider(LLMProvider):
    def __init__(self) -> None:
        self.call_count = 0

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.call_count += 1
        return ChatCompletionResponse(
            id=f"test-id-{self.call_count}",
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=f"Generated response #{self.call_count} for: {request.messages[-1].content}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )


@pytest.mark.asyncio
async def test_in_memory_exact_cache_operations():
    cache = InMemoryExactCache()
    assert len(cache) == 0

    sample_entry = CacheEntry(
        original_request_hash="abc123hash",
        response=ChatCompletionResponse(
            id="sample-id",
            model=TEST_MODEL,
            choices=[],
        ),
        model=TEST_MODEL,
    )

    # Miss
    assert await cache.get("abc123hash") is None

    # Set and Hit
    await cache.set("abc123hash", sample_entry)
    assert len(cache) == 1
    retrieved = await cache.get("abc123hash")
    assert retrieved is not None
    assert retrieved.original_request_hash == "abc123hash"
    assert retrieved.model == TEST_MODEL

    # Clear
    await cache.clear()
    assert len(cache) == 0
    assert await cache.get("abc123hash") is None


def test_phase2_exact_cache_checkpoint():
    """
    PRD Phase 2 Checkpoint:
    Send exactly the same request twice.
    Request 1 -> LLM call, X-Cache: MISS
    Request 2 -> Exact cache hit, X-Cache: HIT-EXACT (no second LLM call)
    Request 3 -> Different request, X-Cache: MISS, second LLM call
    """
    mock_provider = CountingMockProvider()
    isolated_exact_cache = InMemoryExactCache()
    isolated_qdrant = QdrantService(location=":memory:")
    isolated_semantic_cache = SemanticCache(
        exact_cache=isolated_exact_cache,
        embedding_service=get_embedding_service(),
        qdrant_service=isolated_qdrant,
        threshold=0.99,  # Ensures exact-cache testing without semantic false positives
    )

    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    app.dependency_overrides[get_exact_cache] = lambda: isolated_exact_cache
    app.dependency_overrides[get_semantic_cache] = lambda: isolated_semantic_cache

    try:
        payload_1 = {
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Explain machine learning."}],
        }

        # Request 1: Cache MISS
        res1 = client.post("/v1/chat/completions", json=payload_1)
        assert res1.status_code == 200
        assert res1.headers.get("X-Cache") == "MISS"
        data1 = res1.json()
        assert data1["choices"][0]["message"]["content"] == "Generated response #1 for: Explain machine learning."
        assert mock_provider.call_count == 1
        assert len(isolated_exact_cache) == 1

        # Request 2: Identical payload -> Exact Cache HIT
        res2 = client.post("/v1/chat/completions", json=payload_1)
        assert res2.status_code == 200
        assert res2.headers.get("X-Cache") == "HIT-EXACT"
        data2 = res2.json()
        # Verifies the exact same cached response is returned
        assert data2["choices"][0]["message"]["content"] == "Generated response #1 for: Explain machine learning."
        assert data2["id"] == data1["id"]
        # Upstream provider was NOT called again
        assert mock_provider.call_count == 1

        # Request 3: Different payload -> Cache MISS
        payload_3 = {
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "What is deep learning?"}],
        }
        res3 = client.post("/v1/chat/completions", json=payload_3)
        assert res3.status_code == 200
        assert res3.headers.get("X-Cache") == "MISS"
        data3 = res3.json()
        assert data3["choices"][0]["message"]["content"] == "Generated response #2 for: What is deep learning?"
        assert mock_provider.call_count == 2
        assert len(isolated_exact_cache) == 2

    finally:
        app.dependency_overrides.clear()
