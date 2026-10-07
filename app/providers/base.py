from abc import ABC, abstractmethod
from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse


class ProviderException(Exception):
    """Base exception for provider errors."""
    pass


class ProviderAuthenticationError(ProviderException):
    """Raised when provider authentication fails (e.g. missing or invalid API key)."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    name: str = "custom"

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Send chat completion request to the LLM provider and return response."""
        pass
