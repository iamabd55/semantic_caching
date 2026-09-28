from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.cache.exact_cache import get_exact_cache
from app.cache.hashing import compute_request_hash
from app.cache.locking import LockManager, get_lock_manager
from app.cache.pii import check_request_pii
from app.cache.semantic_cache import SemanticCache, get_semantic_cache
from app.core.security import get_cache_ttl, get_tenant_id
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
    lock_manager: LockManager = Depends(get_lock_manager),
    tenant_id: str = Depends(get_tenant_id),
    ttl_seconds: int = Depends(get_cache_ttl),
) -> ChatCompletionResponse:
    """
    Forward chat completion request to semantic cache or upstream LLM provider.
    0. PII Policy: If sensitive PII is detected, bypass caching completely (do not read/write cache).
    1. Tier 1 & 2: Check ExactCache and Qdrant via SemanticCache within tenant partition.
    2. On HIT: return cached response with X-Cache: HIT-EXACT or HIT-SEMANTIC.
    3. On MISS: Acquire single-flight lock, re-check cache (double-checked locking),
       query upstream provider, store in cache with tenant_id and TTL, and return response with X-Cache: MISS.
    """
    response.headers["X-Tenant-ID"] = tenant_id
    response.headers["X-Cache-TTL"] = str(ttl_seconds)

    # 0. PII Guard: Bypass cache completely to avoid persisting or cross-contaminating sensitive data
    has_pii, pii_type = check_request_pii(request)
    if has_pii:
        response.headers["X-Cache"] = "BYPASS-PII"
        response.headers["X-Cache-Bypass-Reason"] = f"PII_DETECTED:{pii_type}"
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
        return llm_response

    # 1. Initial Cache Lookup within tenant boundary
    cache_result = await cache.lookup(request, tenant_id=tenant_id)
    if cache_result.hit and cache_result.response is not None:
        response.headers["X-Cache"] = f"HIT-{cache_result.hit_type}"
        if cache_result.similarity_score is not None:
            response.headers["X-Cache-Score"] = f"{cache_result.similarity_score:.4f}"
        return cache_result.response

    # 2. Cache Miss: Acquire single-flight lock namespaced by tenant and model
    request_hash = compute_request_hash(request)
    model_version = getattr(request, "model_version", None) or "latest"
    lock_key = f"{tenant_id}:{request.model}:{model_version}:{request_hash}"
    async with lock_manager.lock(lock_key):
        # 3. Double-Checked Locking: Check if earlier lock holder populated the cache
        recheck_result = await cache.lookup(request, tenant_id=tenant_id)
        if recheck_result.hit and recheck_result.response is not None:
            response.headers["X-Cache"] = f"HIT-{recheck_result.hit_type}"
            if recheck_result.similarity_score is not None:
                response.headers["X-Cache-Score"] = f"{recheck_result.similarity_score:.4f}"
            return recheck_result.response

        # 4. Query upstream provider with ORIGINAL request untouched
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

        # 5. Store result in both ExactCache and Qdrant under tenant_id with TTL
        await cache.store(request, llm_response, tenant_id=tenant_id, ttl_seconds=ttl_seconds)

        return llm_response



