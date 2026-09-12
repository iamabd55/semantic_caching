from fastapi.testclient import TestClient

from app.api.routes import get_llm_provider
from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)
from app.main import app
from app.providers.base import (
    LLMProvider,
    ProviderAuthenticationError,
    ProviderException,
)

client = TestClient(app)

TEST_MODEL = "dots-studio/dots-3-note-preview:free"


class MockSuccessProvider(LLMProvider):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="test-completion-id",
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(
                        role="assistant",
                        content=f"Echo: {request.messages[-1].content}",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=5,
                completion_tokens=10,
                total_tokens=15,
            ),
        )


class MockFailingProvider(LLMProvider):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise ProviderException("Simulated upstream provider timeout")


class MockAuthFailingProvider(LLMProvider):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise ProviderAuthenticationError("Missing or invalid API key")


def test_chat_completions_validation_error():
    # Empty messages list should fail validation
    response = client.post(
        "/v1/chat/completions",
        json={"model": TEST_MODEL, "messages": []},
    )
    assert response.status_code == 422


def test_chat_completions_missing_model_error():
    # Missing required model field
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 422


def test_chat_completions_success():
    app.dependency_overrides[get_llm_provider] = lambda: MockSuccessProvider()
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": "What is semantic caching?"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-completion-id"
        assert data["model"] == TEST_MODEL
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "Echo: What is semantic caching?"
        assert data["usage"]["total_tokens"] == 15
    finally:
        app.dependency_overrides.clear()


def test_chat_completions_provider_failure():
    app.dependency_overrides[get_llm_provider] = lambda: MockFailingProvider()
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 502
        assert "Simulated upstream provider timeout" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_chat_completions_auth_failure():
    app.dependency_overrides[get_llm_provider] = lambda: MockAuthFailingProvider()
    try:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 401
        assert "Missing or invalid API key" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()

