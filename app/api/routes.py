from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.cache.exact_cache import get_exact_cache
from app.cache.semantic_cache import SemanticCache, get_semantic_cache
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
    response: Response,
    provider: LLMProvider = Depends(get_llm_provider),
    cache: SemanticCache = Depends(get_semantic_cache),
) -> ChatCompletionResponse:
    """
    Forward chat completion request to semantic cache or upstream LLM provider.
    1. Tier 1 & 2: Check ExactCache and Qdrant via SemanticCache.
    2. On HIT: return cached response with X-Cache: HIT-EXACT or HIT-SEMANTIC.
    3. On MISS: forward untouched request to LLM, store in cache, and return response with X-Cache: MISS.
    """
    cache_result = await cache.lookup(request)
    if cache_result.hit and cache_result.response is not None:
        response.headers["X-Cache"] = f"HIT-{cache_result.hit_type}"
        if cache_result.similarity_score is not None:
            response.headers["X-Cache-Score"] = f"{cache_result.similarity_score:.4f}"
        return cache_result.response

    # Cache Miss: Query upstream provider with ORIGINAL request untouched
    response.headers["X-Cache"] = "MISS"
    try:
        llm_response = await provider.complete(request)
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

    # Store result in both ExactCache and Qdrant
    await cache.store(request, llm_response)

    return llm_response



