from app.api.schemas import ChatCompletionRequest, ChatMessage
from app.cache.hashing import compute_request_hash, serialize_request_deterministically


def test_deterministic_serialization_same_hash():
    req1 = ChatCompletionRequest(
        model="dots-studio/dots-3-note-preview:free",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Explain machine learning."),
        ],
        temperature=0.7,
        max_tokens=100,
    )

    req2 = ChatCompletionRequest(
        model="dots-studio/dots-3-note-preview:free",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Explain machine learning."),
        ],
        temperature=0.7,
        max_tokens=100,
    )

    assert serialize_request_deterministically(req1) == serialize_request_deterministically(req2)
    assert compute_request_hash(req1) == compute_request_hash(req2)


def test_different_content_produces_different_hash():
    req1 = ChatCompletionRequest(
        model="dots-studio/dots-3-note-preview:free",
        messages=[ChatMessage(role="user", content="What is python?")],
    )
    req2 = ChatCompletionRequest(
        model="dots-studio/dots-3-note-preview:free",
        messages=[ChatMessage(role="user", content="What is rust?")],
    )
    assert compute_request_hash(req1) != compute_request_hash(req2)


def test_different_model_produces_different_hash():
    req1 = ChatCompletionRequest(
        model="dots-studio/dots-3-note-preview:free",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    req2 = ChatCompletionRequest(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    assert compute_request_hash(req1) != compute_request_hash(req2)


def test_different_temperature_produces_different_hash():
    req1 = ChatCompletionRequest(
        model="dots-studio/dots-3-note-preview:free",
        messages=[ChatMessage(role="user", content="Hello")],
        temperature=0.2,
    )
    req2 = ChatCompletionRequest(
        model="dots-studio/dots-3-note-preview:free",
        messages=[ChatMessage(role="user", content="Hello")],
        temperature=0.9,
    )
    assert compute_request_hash(req1) != compute_request_hash(req2)


def test_different_role_produces_different_hash():
    req1 = ChatCompletionRequest(
        model="dots-studio/dots-3-note-preview:free",
        messages=[ChatMessage(role="user", content="Hello")],
    )
    req2 = ChatCompletionRequest(
        model="dots-studio/dots-3-note-preview:free",
        messages=[ChatMessage(role="system", content="Hello")],
    )
    assert compute_request_hash(req1) != compute_request_hash(req2)
