from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.providers.base import (
    LLMProvider,
    ProviderAuthenticationError,
    ProviderException,
)
from app.providers.openrouter import OpenRouterProvider

router = APIRouter(prefix="/v1", tags=["Chat Completions"])


def get_llm_provider() -> LLMProvider:
    """Dependency provider returning the default LLM provider."""
    return OpenRouterProvider()


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    provider: LLMProvider = Depends(get_llm_provider),
) -> ChatCompletionResponse:
    """Forward chat completion request to the upstream LLM provider."""
    try:
        return await provider.complete(request)
    except ProviderAuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )
    except ProviderException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        )
