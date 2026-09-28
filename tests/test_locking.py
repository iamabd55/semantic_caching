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
from app.cache.locking import (
    InMemoryLockManager,
    LockAcquisitionError,
    RedisLockManager,
    get_lock_manager,
)
from app.cache.semantic_cache import SemanticCache, get_semantic_cache
from app.core.config import get_settings
from app.db.qdrant import QdrantService
from app.embeddings.service import get_embedding_service
from app.main import app
from app.providers.base import LLMProvider


class DelayedCountingLLMProvider(LLMProvider):
    """
    Mock LLM provider that simulates 80ms network generation latency
    and increments an invocation counter on every call.
    """

    def __init__(self, delay_seconds: float = 0.08) -> None:
        self.call_count = 0
        self.delay_seconds = delay_seconds

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.call_count += 1
        await asyncio.sleep(self.delay_seconds)
        return ChatCompletionResponse(
            id=f"chatcmpl-mock-{self.call_count}",
            model=request.model,
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=f"Coalesced response to: {request.messages[-1].content}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


@pytest.mark.asyncio
async def test_in_memory_lock_manager_basic():
    mgr = InMemoryLockManager()

    # Normal lock acquisition and release
    async with mgr.lock("test_key"):
        pass

    # Verify key can be re-locked after exit
    async with mgr.lock("test_key"):
        pass


@pytest.mark.asyncio
async def test_in_memory_lock_manager_timeout():
    mgr = InMemoryLockManager()

    async def hold_lock():
        async with mgr.lock("block_key", timeout=2.0):
            await asyncio.sleep(0.5)

    async def try_acquire():
        await asyncio.sleep(0.05)
        # Attempt acquire with tiny timeout of 0.1s while hold_lock is sleeping for 0.5s
        with pytest.raises(LockAcquisitionError):
            async with mgr.lock("block_key", timeout=0.1):
                pass

    await asyncio.gather(hold_lock(), try_acquire())


@pytest.mark.asyncio
async def test_redis_lock_manager_live():
    """Verify live RedisLockManager if Redis is available in Docker."""
    settings = get_settings()
    lock_mgr = await get_lock_manager()

    if isinstance(lock_mgr, RedisLockManager):
        assert await lock_mgr.ping() is True

        order = []

        async def worker(worker_id: int):
            async with lock_mgr.lock("redis_test_sync", timeout=5.0):
                order.append(f"start_{worker_id}")
                await asyncio.sleep(0.05)
                order.append(f"end_{worker_id}")

        await asyncio.gather(worker(1), worker(2))
        # Ensure sequential execution (no overlapping critical sections)
        assert len(order) == 4
        assert order[0].startswith("start_") and order[1].startswith("end_")
        assert order[2].startswith("start_") and order[3].startswith("end_")


@pytest.mark.asyncio
async def test_prd_phase12_checkpoint_concurrent_request_coalescing():
    """
    PRD Phase 12 Checkpoint:
    Send 10 concurrent identical requests to the gateway during a cold cache miss.
    Verify:
    1. Exactly ONE call reaches the upstream LLM provider.
    2. All 10 requests return 200 OK with the exact same completion.
    3. The first request returns X-Cache: MISS; subsequent requests return X-Cache: HIT-EXACT.
    """
    mock_provider = DelayedCountingLLMProvider(delay_seconds=0.1)

    # Use isolated in-memory exact cache and in-memory Qdrant
    isolated_exact_cache = InMemoryExactCache()
    isolated_qdrant = QdrantService(location=":memory:")
    isolated_qdrant.ensure_collection()

    isolated_semantic_cache = SemanticCache(
        exact_cache=isolated_exact_cache,
        embedding_service=get_embedding_service(),
        qdrant_service=isolated_qdrant,
        threshold=0.75,
    )

    # Dependency overrides
    from app.api.routes import get_llm_provider
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    app.dependency_overrides[get_semantic_cache] = lambda: isolated_semantic_cache

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [
                    {"role": "user", "content": "Explain single-flight locking in distributed systems."}
                ],
                "temperature": 0.0,
            }

            # Fire 10 concurrent requests simultaneously
            tasks = [
                client.post("/v1/chat/completions", json=payload)
                for _ in range(10)
            ]
            responses = await asyncio.gather(*tasks)

        # 1. Assert all 10 requests succeeded
        assert len(responses) == 10
        for resp in responses:
            assert resp.status_code == 200
            data = resp.json()
            assert "Coalesced response to: Explain single-flight locking" in data["choices"][0]["message"]["content"]

        # 2. Checkpoint: Upstream LLM provider called EXACTLY ONCE
        assert mock_provider.call_count == 1, (
            f"Expected exactly 1 LLM call due to single-flight locking, but got {mock_provider.call_count}"
        )

        # 3. Cache headers check:
        # At least one request was MISS, and the waiting concurrent requests were HIT-EXACT
        cache_headers = [resp.headers.get("X-Cache") for resp in responses]
        assert "MISS" in cache_headers
        assert "HIT-EXACT" in cache_headers
        assert cache_headers.count("HIT-EXACT") == 9

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_prd_phase12_checkpoint_in_memory_coalescing():
    """Verify concurrent request coalescing works identically with in-memory lock manager."""
    mock_provider = DelayedCountingLLMProvider(delay_seconds=0.1)

    isolated_exact_cache = InMemoryExactCache()
    isolated_qdrant = QdrantService(location=":memory:")
    isolated_qdrant.ensure_collection()

    isolated_semantic_cache = SemanticCache(
        exact_cache=isolated_exact_cache,
        embedding_service=get_embedding_service(),
        qdrant_service=isolated_qdrant,
        threshold=0.75,
    )
    in_memory_lock = InMemoryLockManager()

    from app.api.routes import get_llm_provider
    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    app.dependency_overrides[get_semantic_cache] = lambda: isolated_semantic_cache
    app.dependency_overrides[get_lock_manager] = lambda: in_memory_lock

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [
                    {"role": "user", "content": "Explain concurrency control in async Python."}
                ],
                "temperature": 0.0,
            }

            tasks = [
                client.post("/v1/chat/completions", json=payload)
                for _ in range(8)
            ]
            responses = await asyncio.gather(*tasks)

        assert len(responses) == 8
        for resp in responses:
            assert resp.status_code == 200

        # Exact coalescing assertion
        assert mock_provider.call_count == 1
        cache_headers = [resp.headers.get("X-Cache") for resp in responses]
        assert "MISS" in cache_headers
        assert cache_headers.count("HIT-EXACT") == 7

    finally:
        app.dependency_overrides.clear()

