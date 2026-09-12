from fastapi.testclient import TestClient

from app.api.routes import get_exact_cache, get_llm_provider
from app.api.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
)
from app.cache.canonicalization import canonicalize_request, canonicalize_text
from app.cache.exact_cache import InMemoryExactCache
from app.cache.hashing import compute_request_hash
from app.main import app
from app.providers.base import LLMProvider

client = TestClient(app)

TEST_MODEL = "dots-studio/dots-3-note-preview:free"


def test_canonicalize_text_lowercasing():
    assert canonicalize_text("HELLO WORLD") == "hello world"


def test_canonicalize_text_whitespace_normalization():
    raw = "What   is\n\nsemantic    caching?  "
    assert canonicalize_text(raw) == "what is semantic caching"


def test_canonicalize_text_punctuation_normalization():
    # PRD Section 12 explicit example
    raw = "Explain   MACHINE learning!!"
    assert canonicalize_text(raw) == "explain machine learning"


def test_canonicalize_request_single_message():
    req = ChatCompletionRequest(
        model=TEST_MODEL,
        messages=[ChatMessage(role="user", content="  Explain   MACHINE learning!! ")],
    )
    assert canonicalize_request(req) == "explain machine learning"


def test_canonicalize_request_multi_turn():
    req = ChatCompletionRequest(
        model=TEST_MODEL,
        messages=[
            ChatMessage(role="system", content="You are a helpful ASSISTANT!"),
            ChatMessage(role="user", content="  Tell me a JOKE?! "),
        ],
    )
    expected = "system: you are a helpful assistant\nuser: tell me a joke"
    assert canonicalize_request(req) == expected


class InspectingMockProvider(LLMProvider):
    """Mock provider that records the exact request it received."""
    def __init__(self):
        self.received_request: ChatCompletionRequest | None = None

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.received_request = request
        return ChatCompletionResponse(
            id="test-id",
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content="Response"),
                    finish_reason="stop",
                )
            ],
        )


def test_prd_phase3_checkpoint_original_vs_canonicalized():
    """
    PRD Phase 3 Checkpoint:
    Verify that:
      1. Original Request != Modified/Canonicalized Request when formatting is present.
      2. The LLM provider receives the unmodified Original Request.
      3. The CacheEntry stores the canonicalized text in `normalized_request`.
    """
    mock_provider = InspectingMockProvider()
    isolated_cache = InMemoryExactCache()

    app.dependency_overrides[get_llm_provider] = lambda: mock_provider
    app.dependency_overrides[get_exact_cache] = lambda: isolated_cache

    try:
        raw_prompt = "  Explain   MACHINE Learning!! "
        payload = {
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": raw_prompt}],
            "temperature": 0.7,
        }

        # Send through gateway
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200

        # Checkpoint 1: Original text != Canonicalized text
        canonical_text = canonicalize_text(raw_prompt)
        assert canonical_text == "explain machine learning"
        assert raw_prompt != canonical_text

        # Checkpoint 2: LLM provider received the ORIGINAL untouched request
        assert mock_provider.received_request is not None
        assert mock_provider.received_request.messages[0].content == raw_prompt
        assert mock_provider.received_request.messages[0].content != canonical_text

        # Checkpoint 3: Cache entry contains normalized_request for semantic matching
        req_obj = ChatCompletionRequest(**payload)
        req_hash = compute_request_hash(req_obj)
        cached_entry = isolated_cache._store.get(req_hash)

        assert cached_entry is not None
        assert cached_entry.original_request_hash == req_hash
        assert cached_entry.normalized_request == "explain machine learning"

    finally:
        app.dependency_overrides.clear()
