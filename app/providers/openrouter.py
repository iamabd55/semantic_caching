import httpx
from typing import Optional

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
)
from app.core.config import settings
from app.providers.base import LLMProvider, ProviderAuthenticationError, ProviderException


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider implementation."""

    name: str = "openrouter"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.timeout = timeout

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ProviderAuthenticationError(
                "OpenRouter API key is not configured. Please set OPENROUTER_API_KEY in .env"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/semantic-caching",
            "X-Title": settings.PROJECT_NAME,
        }

        payload = {
            "model": request.model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.API_URL, json=payload, headers=headers)

            if response.status_code in (401, 403):
                raise ProviderAuthenticationError(
                    f"OpenRouter authentication failed: {response.text}"
                )
            if response.status_code >= 400:
                raise ProviderException(
                    f"OpenRouter request failed with status {response.status_code}: {response.text}"
                )

            data = response.json()
            return self._format_response(data)

        except httpx.TimeoutException as exc:
            raise ProviderException("OpenRouter request timed out.") from exc
        except httpx.RequestError as exc:
            raise ProviderException(f"OpenRouter network error: {str(exc)}") from exc

    def _format_response(self, data: dict) -> ChatCompletionResponse:
        choices = []
        for choice in data.get("choices", []):
            msg = choice.get("message", {})
            choices.append(
                Choice(
                    index=choice.get("index", 0),
                    message=ChoiceMessage(
                        role=msg.get("role", "assistant"),
                        content=msg.get("content", ""),
                    ),
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            )

        usage_data = data.get("usage")
        usage = None
        if usage_data:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

        return ChatCompletionResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
        )
