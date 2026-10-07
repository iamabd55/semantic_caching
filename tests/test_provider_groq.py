import json
import pytest
import httpx
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient

from app.api.routes import get_llm_provider
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
from app.providers.base import (
    ProviderAuthenticationError,
    ProviderException,
)
from app.providers.factory import get_provider
from app.providers.groq import GroqProvider
from app.providers.openrouter import OpenRouterProvider


# ==============================================================================
# Unit Tests for Factory & GroqProvider
# ==============================================================================

def test_provider_factory_resolution():
    p1 = get_provider("openrouter")
    assert isinstance(p1, OpenRouterProvider)
    assert p1.name == "openrouter"

    p2 = get_provider("groq")
    assert isinstance(p2, GroqProvider)
    assert p2.name == "groq"

    p3 = get_provider("  GrOq  ")
    assert isinstance(p3, GroqProvider)

    p_default = get_provider(None)
    assert isinstance(p_default, OpenRouterProvider)

    with pytest.raises(ProviderException, match="Unsupported LLM provider 'anthropic'"):
        get_provider("anthropic")


@pytest.mark.asyncio
async def test_groq_missing_api_key():
    provider = GroqProvider(api_key=None)
    # Ensure settings.GROQ_API_KEY is also cleared if present
    provider.api_key = None
    req = ChatCompletionRequest(
        model="llama-3.1-8b-instant",
        messages=[ChatMessage(role="user", content="Hello Groq")],
    )
    with pytest.raises(ProviderAuthenticationError, match="Groq API key is not configured"):
        await provider.complete(req)


@pytest.mark.asyncio
async def test_groq_successful_response():
    provider = GroqProvider(api_key="gsk_dummy_test_key")
    req = ChatCompletionRequest(
        model="llama-3.1-8b-instant",
        messages=[ChatMessage(role="user", content="What is latency?")],
        temperature=0.7,
        max_tokens=50,
    )

    fake_groq_payload = {
        "id": "chatcmpl-groq-mock-123",
        "model": "llama-3.1-8b-instant",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Latency is the delay before a transfer of data begins.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 18,
            "total_tokens": 30,
        },
    }

    mock_resp = httpx.Response(
        status_code=200,
        json=fake_groq_payload,
        request=httpx.Request("POST", GroqProvider.API_URL),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        result = await provider.complete(req)

        assert result.id == "chatcmpl-groq-mock-123"
        assert result.model == "llama-3.1-8b-instant"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Latency is the delay before a transfer of data begins."
        assert result.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_groq_http_error_handling():
    provider = GroqProvider(api_key="gsk_dummy_test_key")
    req = ChatCompletionRequest(
        model="llama-3.1-8b-instant",
        messages=[ChatMessage(role="user", content="Test Error")],
    )

    # 401 Unauthorized
    mock_401 = httpx.Response(
        status_code=401,
        text="Invalid API Key",
        request=httpx.Request("POST", GroqProvider.API_URL),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_401
        with pytest.raises(ProviderAuthenticationError, match="Groq authentication failed"):
            await provider.complete(req)

    # 500 Server Error
    mock_500 = httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("POST", GroqProvider.API_URL),
    )
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_500
        with pytest.raises(ProviderException, match="Groq request failed with status 500"):
            await provider.complete(req)


# ==============================================================================
# Integration Tests: Gateway Multi-Provider Route
# ==============================================================================

@pytest.mark.asyncio
async def test_prd_phase17_gateway_provider_routing():
    """
    PRD Phase 17 Checkpoint:
    Verify that the gateway supports the Provider Abstraction:
    - Default provider resolves to OpenRouter (X-Provider: openrouter)
    - X-Provider: groq header routes to Groq (X-Provider: groq)
    - Unsupported provider returns HTTP 400 Bad Request
    """
    fake_groq_payload = {
        "id": "chatcmpl-groq-phase17",
        "model": "llama-3.1-8b-instant",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Ultra fast response powered by Groq LPU.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
    }

    clean_exact = InMemoryExactCache()
    qdrant_service = QdrantService(location=":memory:")
    embed_service = get_embedding_service()
    clean_semantic = SemanticCache(
        exact_cache=clean_exact,
        qdrant_service=qdrant_service,
        embedding_service=embed_service,
    )

    app.dependency_overrides[get_semantic_cache] = lambda: clean_semantic

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. Invalid provider header -> 400 Bad Request
            res_bad = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers={"X-Provider": "unknown-llm"},
            )
            assert res_bad.status_code == 400
            assert "Unsupported LLM provider 'unknown-llm'" in res_bad.json()["detail"]

            # 2. Call with X-Provider: groq
            groq_response = ChatCompletionResponse(
                id="chatcmpl-groq-phase17",
                model="llama-3.1-8b-instant",
                choices=[
                    Choice(
                        index=0,
                        message=ChoiceMessage(
                            role="assistant",
                            content="Ultra fast response powered by Groq LPU.",
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(prompt_tokens=8, completion_tokens=12, total_tokens=20),
            )
            with patch.object(GroqProvider, "complete", new_callable=AsyncMock) as mock_groq_complete:
                mock_groq_complete.return_value = groq_response
                res_groq = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": "Explain Groq LPUs"}],
                    },
                    headers={
                        "X-Provider": "groq",
                        "X-Tenant-ID": "tenant-groq-test",
                    },
                )
                assert res_groq.status_code == 200
                assert res_groq.headers["x-provider"] == "groq"
                assert res_groq.headers["x-cache"] == "MISS"
                assert res_groq.json()["choices"][0]["message"]["content"] == "Ultra fast response powered by Groq LPU."
                assert mock_groq_complete.call_count == 1

            # 3. Second call with identical prompt hits cache seamlessly regardless of provider
            res_hit = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": "Explain Groq LPUs"}],
                },
                headers={
                    "X-Provider": "groq",
                    "X-Tenant-ID": "tenant-groq-test",
                },
            )
            assert res_hit.status_code == 200
            assert res_hit.headers["x-cache"] == "HIT-EXACT"
            assert res_hit.json()["choices"][0]["message"]["content"] == "Ultra fast response powered by Groq LPU."

    finally:
        app.dependency_overrides.clear()
