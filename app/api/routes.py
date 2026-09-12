from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.cache.canonicalization import canonicalize_request
from app.cache.exact_cache import ExactCache, get_exact_cache
from app.cache.hashing import compute_request_hash
from app.cache.models import CacheEntry
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
    cache: ExactCache = Depends(get_exact_cache),
) -> ChatCompletionResponse:
    """
    Forward chat completion request to exact cache or upstream LLM provider.
    1. Deterministically hash original request and check exact cache (fast-path).
    2. On HIT: return cached response with X-Cache: HIT-EXACT.
    3. On MISS: forward original request to LLM, populate cache with both original
       hash and canonicalized text (for semantic search), and return response with X-Cache: MISS.
    """
    request_hash = compute_request_hash(request)

    # 1. Exact Cache Fast-Path Lookup
    cached_entry = await cache.get(request_hash)
    if cached_entry is not None:
        response.headers["X-Cache"] = "HIT-EXACT"
        return cached_entry.response

    # 2. Cache Miss: Query upstream provider with ORIGINAL request untouched
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

    # 3. Canonicalize request for semantic cache storage
    normalized_request = canonicalize_request(request)

    # 4. Store result in exact cache
    cache_entry = CacheEntry(
        original_request_hash=request_hash,
        response=llm_response,
        model=request.model,
        normalized_request=normalized_request,
    )
    await cache.set(request_hash, cache_entry)

    return llm_response


