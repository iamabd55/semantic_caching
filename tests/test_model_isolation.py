import asyncio
import time
import uuid
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


class ModelAwareMockLLMProvider(LLMProvider):
    """
    Mock LLM provider that formats responses using the requested model and version,
    and tracks call counts per model.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.calls_by_model = {}

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.call_count += 1
        model_key = f"{request.model}:{request.model_version or 'latest'}"
        self.calls_by_model[model_key] = self.calls_by_model.get(model_key, 0) + 1

        prompt = request.messages[-1].content
        return ChatCompletionResponse(
            id=f"chatcmpl-{self.call_count}",
            model=request.model,
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=f"[{request.model} (ver: {request.model_version or 'latest'})] Response to: '{prompt}'",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        )


@pytest.mark.asyncio
async def test_qdrant_model_and_version_pre_filtering():
    """
    Verify that QdrantService strictly filters vectors by model and model_version
    before calculating similarity.
    """
    qdrant = QdrantService(location=":memory:")
    qdrant.ensure_collection("test_model_isolation")
    embedder = get_embedding_service()

    vector = embedder.embed("How do neural networks work?")

    point_llama = str(uuid.uuid4())
    point_claude = str(uuid.uuid4())

    # 1. Upsert Llama vector
    qdrant.upsert_entry(
        point_id=point_llama,
        vector=vector,
        payload={
            "tenant_id": "default",
            "model": "meta-llama/llama-3-8b-instruct",
            "model_version": "v1",
            "data": "llama_data",
        },
        collection_name="test_model_isolation",
    )

    # 2. Upsert Claude vector
    qdrant.upsert_entry(
        point_id=point_claude,
        vector=vector,
        payload={
            "tenant_id": "default",
            "model": "anthropic/claude-3-haiku",
            "model_version": "v1",
            "data": "claude_data",
        },
        collection_name="test_model_isolation",
    )

    # 3. Query for Llama -> must return ONLY Llama point
    llama_results = qdrant.search_nearest(
        query_vector=vector,
        limit=5,
        tenant_id="default",
        model="meta-llama/llama-3-8b-instruct",
        collection_name="test_model_isolation",
    )
    assert len(llama_results) == 1
    assert llama_results[0]["id"] == point_llama

    # 4. Query for Claude -> must return ONLY Claude point
    claude_results = qdrant.search_nearest(
        query_vector=vector,
        limit=5,
        tenant_id="default",
        model="anthropic/claude-3-haiku",
        collection_name="test_model_isolation",
    )
    assert len(claude_results) == 1
    assert claude_results[0]["id"] == point_claude

    # 5. Query for non-existent model -> must return empty
    gpt_results = qdrant.search_nearest(
        query_vector=vector,
        limit=5,
        tenant_id="default",
        model="openai/gpt-4o",
        collection_name="test_model_isolation",
    )
    assert len(gpt_results) == 0


@pytest.mark.asyncio
async def test_prd_phase14_checkpoint_cross_model_cache_isolation():
    """
    PRD Phase 14 Checkpoint:
    Verify that requests to Model A do NOT return responses cached from Model B,
    even for semantically identical prompts.
    """
    mock_provider = ModelAwareMockLLMProvider()

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
            prompt_base = "What is distributed consensus?"
            prompt_paraphrase = "Explain distributed consensus algorithms."

            # -------------------------------------------------------------
            # STEP 1: Query Model A (Llama) -> MISS, stores Llama response
            # -------------------------------------------------------------
            resp_llama_1 = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta-llama/llama-3-8b-instruct",
                    "messages": [{"role": "user", "content": prompt_base}],
                },
            )
            assert resp_llama_1.status_code == 200
            assert resp_llama_1.headers["X-Cache"] == "MISS"
            assert mock_provider.call_count == 1
            body_llama_1 = resp_llama_1.json()["choices"][0]["message"]["content"]
            assert "[meta-llama/llama-3-8b-instruct" in body_llama_1

            # -------------------------------------------------------------
            # STEP 2: Query Model B (Claude) with EXACT SAME PROMPT -> MUST MISS!
            # -------------------------------------------------------------
            resp_claude_1 = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": prompt_base}],
                },
            )
            assert resp_claude_1.status_code == 200
            # CRITICAL CHECKPOINT: Must NOT return Llama's response!
            assert resp_claude_1.headers["X-Cache"] == "MISS"
            assert mock_provider.call_count == 2
            body_claude_1 = resp_claude_1.json()["choices"][0]["message"]["content"]
            assert "[anthropic/claude-3-haiku" in body_claude_1
            assert body_claude_1 != body_llama_1

            # -------------------------------------------------------------
            # STEP 3: Paraphrase query for Model A -> HIT with Llama response
            # -------------------------------------------------------------
            resp_llama_2 = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta-llama/llama-3-8b-instruct",
                    "messages": [{"role": "user", "content": prompt_paraphrase}],
                },
            )
            assert resp_llama_2.status_code == 200
            assert resp_llama_2.headers["X-Cache"] == "HIT-SEMANTIC"
            assert mock_provider.call_count == 2  # No new call
            body_llama_2 = resp_llama_2.json()["choices"][0]["message"]["content"]
            assert body_llama_2 == body_llama_1

            # -------------------------------------------------------------
            # STEP 4: Paraphrase query for Model B -> HIT with Claude response
            # -------------------------------------------------------------
            resp_claude_2 = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "anthropic/claude-3-haiku",
                    "messages": [{"role": "user", "content": prompt_paraphrase}],
                },
            )
            assert resp_claude_2.status_code == 200
            assert resp_claude_2.headers["X-Cache"] == "HIT-SEMANTIC"
            assert mock_provider.call_count == 2  # No new call
            body_claude_2 = resp_claude_2.json()["choices"][0]["message"]["content"]
            assert body_claude_2 == body_claude_1

    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_model_version_isolation():
    """
    Verify that requests with different model_version fields are strictly isolated.
    """
    mock_provider = ModelAwareMockLLMProvider()

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
            prompt = "Optimize this SQL query."

            # Query Version 2024-05-13
            resp_v1 = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o",
                    "model_version": "2024-05-13",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            assert resp_v1.status_code == 200
            assert resp_v1.headers["X-Cache"] == "MISS"
            assert "ver: 2024-05-13" in resp_v1.json()["choices"][0]["message"]["content"]

            # Query Version 2024-08-06 -> MUST MISS
            resp_v2 = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o",
                    "model_version": "2024-08-06",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            assert resp_v2.status_code == 200
            assert resp_v2.headers["X-Cache"] == "MISS"
            assert "ver: 2024-08-06" in resp_v2.json()["choices"][0]["message"]["content"]

            # Re-query Version 2024-05-13 -> MUST HIT
            resp_v1_repeat = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4o",
                    "model_version": "2024-05-13",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            assert resp_v1_repeat.status_code == 200
            assert resp_v1_repeat.headers["X-Cache"] == "HIT-EXACT"
            assert "ver: 2024-05-13" in resp_v1_repeat.json()["choices"][0]["message"]["content"]

    finally:
        app.dependency_overrides.clear()
