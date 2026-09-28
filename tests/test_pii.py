import asyncio
import time
import pytest
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
from app.cache.pii import check_request_pii, contains_pii
from app.cache.semantic_cache import SemanticCache, get_semantic_cache
from app.db.qdrant import QdrantService
from app.embeddings.service import get_embedding_service
from app.main import app
from app.providers.base import LLMProvider


class PIICountingMockLLMProvider(LLMProvider):
    """
    Mock LLM provider that increments a counter on every call to track
    when upstream queries happen versus cache hits or bypasses.
    """

    def __init__(self) -> None:
        self.call_count = 0

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.call_count += 1
        prompt = request.messages[-1].content
        return ChatCompletionResponse(
            id=f"chatcmpl-pii-{self.call_count}",
            model=request.model,
            created=int(time.time()),
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=f"LLM Response #{self.call_count} to: '{prompt}'",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


# ==============================================================================
# Unit Tests for PII Detector
# ==============================================================================

def test_unit_email_detection():
    has_pii, pii_type = contains_pii("Send billing details to finance-dept@example.org today.")
    assert has_pii is True
    assert pii_type == "EMAIL"


def test_unit_phone_detection():
    has_pii, pii_type = contains_pii("Please call my direct line at +1 (555) 345-6789.")
    assert has_pii is True
    assert pii_type == "PHONE"


def test_unit_ssn_detection():
    has_pii, pii_type = contains_pii("Patient SSN record on file: 000-12-3456.")
    assert has_pii is True
    assert pii_type == "SSN"


def test_unit_credit_card_detection():
    has_pii, pii_type = contains_pii("Charge to Visa card 4111-2222-3333-4444 expiring 12/28.")
    assert has_pii is True
    assert pii_type == "CREDIT_CARD"


def test_unit_api_key_detection():
    has_pii, pii_type = contains_pii("Use this secret key: sk-proj-abcdef1234567890abcdef1234567890")
    assert has_pii is True
    assert pii_type == "API_KEY"


def test_unit_clean_text_no_pii():
    has_pii, pii_type = contains_pii("What are the advantages of convolutional neural networks?")
    assert has_pii is False
    assert pii_type is None


def test_unit_check_request_pii():
    clean_req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="How do I sort a list in Python?"),
        ],
    )
    assert check_request_pii(clean_req) == (False, None)

    dirty_req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Reset password for user@company.com"),
        ],
    )
    has_pii, pii_type = check_request_pii(dirty_req)
    assert has_pii is True
    assert pii_type == "EMAIL"


# ==============================================================================
# Integration Tests: Gateway PII Bypass Policy
# ==============================================================================

@pytest.mark.asyncio
async def test_prd_phase16_pii_bypass_and_no_storage():
    """
    PRD Phase 16 Checkpoint:
    Verify that requests containing high-risk PII bypass the cache (X-Cache: BYPASS-PII),
    invoke the LLM provider directly, and are never stored in cache (so repeated requests
    also trigger fresh LLM calls).
    """
    mock_provider = PIICountingMockLLMProvider()
    clean_exact = InMemoryExactCache()
    qdrant_service = QdrantService(location=":memory:")
    embed_service = get_embedding_service()
    clean_semantic = SemanticCache(
        exact_cache=clean_exact,
        qdrant_service=qdrant_service,
        embedding_service=embed_service,
    )

    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    app.dependency_overrides[get_semantic_cache] = lambda: clean_semantic

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            pii_payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": "Reset the credentials for support@corporate-enterprise.com immediately.",
                    }
                ],
            }

            # 1. First request with PII
            res1 = await client.post("/v1/chat/completions", json=pii_payload)
            assert res1.status_code == 200
            assert res1.headers["x-cache"] == "BYPASS-PII"
            assert "EMAIL" in res1.headers["x-cache-bypass-reason"]
            assert mock_provider.call_count == 1

            # 2. Second identical request with PII
            # Must NOT be served from cache; must bypass and trigger call_count == 2
            res2 = await client.post("/v1/chat/completions", json=pii_payload)
            assert res2.status_code == 200
            assert res2.headers["x-cache"] == "BYPASS-PII"
            assert mock_provider.call_count == 2

            # 3. Verify clean non-PII requests continue normal caching
            clean_payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain the concept of semantic caching in distributed systems.",
                    }
                ],
            }

            # Clean request 1 -> Cache MISS -> LLM call #3
            res3 = await client.post("/v1/chat/completions", json=clean_payload)
            assert res3.status_code == 200
            assert res3.headers["x-cache"] == "MISS"
            assert mock_provider.call_count == 3

            # Clean request 2 -> Cache HIT -> no new LLM call
            res4 = await client.post("/v1/chat/completions", json=clean_payload)
            assert res4.status_code == 200
            assert res4.headers["x-cache"] == "HIT-EXACT"
            assert mock_provider.call_count == 3

    finally:
        app.dependency_overrides.clear()
