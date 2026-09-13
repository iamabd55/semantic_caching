from fastapi.testclient import TestClient
import pytest

from app.api.routes import get_llm_provider
from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
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

client = TestClient(app)

TEST_MODEL = "dots-studio/dots-3-note-preview:free"


class CountingMockProvider(LLMProvider):
    def __init__(self) -> None:
        self.call_count = 0

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.call_count += 1
        return ChatCompletionResponse(
            id=f"gen-id-{self.call_count}",
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=f"LLM Answer #{self.call_count} for: {request.messages[-1].content}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=15,
                completion_tokens=25,
                total_tokens=40,
            ),
        )


@pytest.fixture
def isolated_semantic_cache():
    """
    Creates an isolated SemanticCache with an in-memory exact store
    and an in-memory Qdrant database for isolated test execution.
    """
    exact = InMemoryExactCache()
    embedding = get_embedding_service()
    qdrant = QdrantService(location=":memory:")
    qdrant.ensure_collection()
    return SemanticCache(
        exact_cache=exact,
        embedding_service=embedding,
        qdrant_service=qdrant,
        threshold=0.75,  # 0.75 threshold for robust semantic matching
    )


def test_prd_phase6_checkpoint_full_flow(isolated_semantic_cache):
    """
    PRD Phase 6 Checkpoint:
    1. First Request: "What is machine learning?" -> Cache Miss -> LLM Called -> Stored.
    2. Second Request: "Tell me what is machine learning?" -> Semantic Cache Hit -> LLM Not Called!
    3. Third Request: "What is machine learning?" -> Exact Cache Hit -> LLM Not Called!
    4. Fourth Request: "How do I bake sourdough bread?" -> Cache Miss -> LLM Called!
    """
    mock_provider = CountingMockProvider()

    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    app.dependency_overrides[get_semantic_cache] = lambda: isolated_semantic_cache

    try:
        # 1. Request 1: Initial Query -> Cache MISS
        payload_1 = {
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "What is machine learning?"}],
        }
        res1 = client.post("/v1/chat/completions", json=payload_1)
        assert res1.status_code == 200
        assert res1.headers.get("X-Cache") == "MISS"
        data1 = res1.json()
        assert data1["id"] == "gen-id-1"
        assert "LLM Answer #1" in data1["choices"][0]["message"]["content"]
        assert mock_provider.call_count == 1

        # 2. Request 2: Semantically Equivalent Query -> Semantic Cache HIT!
        payload_2 = {
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Tell me what is machine learning?"}],
        }
        res2 = client.post("/v1/chat/completions", json=payload_2)
        assert res2.status_code == 200
        assert res2.headers.get("X-Cache") == "HIT-SEMANTIC"
        assert res2.headers.get("X-Cache-Score") is not None
        score = float(res2.headers["X-Cache-Score"])
        assert score >= 0.75, f"Expected similarity score >= 0.75, got {score}"

        data2 = res2.json()
        # Returns previously cached response
        assert data2["id"] == "gen-id-1"
        assert "LLM Answer #1" in data2["choices"][0]["message"]["content"]
        # LLM was NOT called again!
        assert mock_provider.call_count == 1

        # 3. Request 3: Exact repeat of Request 1 -> Exact Cache HIT!
        res3 = client.post("/v1/chat/completions", json=payload_1)
        assert res3.status_code == 200
        assert res3.headers.get("X-Cache") == "HIT-EXACT"
        data3 = res3.json()
        assert data3["id"] == "gen-id-1"
        # LLM still not called
        assert mock_provider.call_count == 1

        # 4. Request 4: Completely unrelated query -> Cache MISS
        payload_4 = {
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "How do I bake sourdough bread?"}],
        }
        res4 = client.post("/v1/chat/completions", json=payload_4)
        assert res4.status_code == 200
        assert res4.headers.get("X-Cache") == "MISS"
        data4 = res4.json()
        assert data4["id"] == "gen-id-2"
        assert "LLM Answer #2" in data4["choices"][0]["message"]["content"]
        # LLM was called a second time
        assert mock_provider.call_count == 2

    finally:
        app.dependency_overrides.clear()
