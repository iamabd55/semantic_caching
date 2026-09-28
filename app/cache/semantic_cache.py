from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.cache.canonicalization import canonicalize_request
from app.cache.exact_cache import ExactCache, get_exact_cache
from app.cache.hashing import compute_request_hash
from app.cache.models import CacheEntry
from app.cache.polarity_guard import has_polarity_conflict
from app.core.config import settings
from app.db.qdrant import QdrantService, get_qdrant_service
from app.embeddings.service import EmbeddingService, get_embedding_service


@dataclass
class CacheResult:
    """Represents the outcome of a cache lookup."""
    hit: bool
    hit_type: Optional[str] = None  # "EXACT" or "SEMANTIC"
    response: Optional[ChatCompletionResponse] = None
    similarity_score: Optional[float] = None
    embedding_latency_ms: Optional[float] = None


class SemanticCache:
    """
    Unified Semantic Caching Engine (Phase 6).
    1. Checks ExactCache (SHA-256 hash fast-path, O(1), zero embedding overhead).
    2. On miss, canonicalizes text, computes local embedding, and queries Qdrant for semantic neighbors.
    3. Evaluates cosine similarity against SIMILARITY_THRESHOLD.
    4. On full miss, stores responses in both ExactCache and Qdrant.
    """

    def __init__(
        self,
        exact_cache: ExactCache,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        threshold: Optional[float] = None,
    ) -> None:
        self.exact_cache = exact_cache
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

    async def lookup(
        self,
        request: ChatCompletionRequest,
        tenant_id: str = "default",
    ) -> CacheResult:
        """
        Execute tiered cache lookup: Exact Match -> Semantic Match.
        Enforces tenant and model/version isolation across both lookup tiers.
        """
        now = datetime.now(timezone.utc)
        request_hash = compute_request_hash(request)
        model_version = getattr(request, "model_version", None)
        exact_cache_key = f"{tenant_id}:{request.model}:{model_version or 'latest'}:{request_hash}"

        # 1. Tier 1: Exact Hash Fast-Path Lookup (O(1), < 1ms)
        exact_entry = await self.exact_cache.get(exact_cache_key)
        if exact_entry is not None:
            if exact_entry.is_expired(now):
                await self.exact_cache.delete(exact_cache_key)
            else:
                return CacheResult(
                    hit=True,
                    hit_type="EXACT",
                    response=exact_entry.response,
                    similarity_score=1.0,
                    embedding_latency_ms=0.0,
                )

        # 2. Tier 2: Semantic Similarity Lookup via Qdrant with Tenant and Model Pre-Filtering
        canonical_text = canonicalize_request(request)
        query_vector, latency_ms = self.embedding_service.embed_with_latency(canonical_text)

        matches = self.qdrant_service.search_nearest(
            query_vector=query_vector,
            limit=1,
            tenant_id=tenant_id,
            model=request.model,
            model_version=model_version,
        )

        if matches:
            top_match = matches[0]
            score = top_match["score"]

            if score >= self.threshold:
                payload = top_match.get("payload", {})

                # Check TTL expiration on semantic match
                is_expired = False
                expires_at_str = payload.get("expires_at")
                if expires_at_str:
                    try:
                        exp_dt = datetime.fromisoformat(expires_at_str)
                        if now > exp_dt:
                            is_expired = True
                    except Exception:
                        pass

                if not is_expired:
                    cached_text = payload.get("normalized_request", "")

                    # Post-filter: prevent false hits on explicit negation or antonym polarity conflict
                    if not has_polarity_conflict(canonical_text, cached_text):
                        cached_response_data = payload.get("response")
                        if cached_response_data:
                            cached_response = ChatCompletionResponse.model_validate(cached_response_data)
                            return CacheResult(
                                hit=True,
                                hit_type="SEMANTIC",
                                response=cached_response,
                                similarity_score=score,
                                embedding_latency_ms=latency_ms,
                            )

        # 3. Cache Miss
        top_score = matches[0]["score"] if matches else None
        return CacheResult(
            hit=False,
            hit_type=None,
            response=None,
            similarity_score=top_score,
            embedding_latency_ms=latency_ms,
        )

    async def store(
        self,
        request: ChatCompletionRequest,
        response: ChatCompletionResponse,
        tenant_id: str = "default",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Store a new completion response in both ExactCache and Qdrant.
        Enforces tenant, model/version isolation, and TTL expiration.
        """
        request_hash = compute_request_hash(request)
        model_version = getattr(request, "model_version", None)
        exact_cache_key = f"{tenant_id}:{request.model}:{model_version or 'latest'}:{request_hash}"
        canonical_text = canonicalize_request(request)
        vector = self.embedding_service.embed(canonical_text)

        now = datetime.now(timezone.utc)
        effective_ttl = ttl_seconds if ttl_seconds is not None else settings.DEFAULT_CACHE_TTL_SECONDS
        expires_at = now + timedelta(seconds=effective_ttl) if effective_ttl > 0 else None

        entry = CacheEntry(
            original_request_hash=request_hash,
            response=response,
            model=request.model,
            model_version=model_version,
            normalized_request=canonical_text,
            embedding=vector,
            tenant_id=tenant_id,
            created_at=now,
            expires_at=expires_at,
        )

        # 1. Store in Exact Cache (namespaced by tenant_id, model, and version)
        await self.exact_cache.set(exact_cache_key, entry)

        # 2. Store in Qdrant Vector Store (payload tagged with tenant_id, model, and TTL for pre-filtering & freshness)
        payload = {
            "original_request_hash": request_hash,
            "normalized_request": canonical_text,
            "response": response.model_dump(),
            "model": request.model,
            "model_version": model_version,
            "tenant_id": tenant_id,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat() if expires_at else None,
        }
        self.qdrant_service.upsert_entry(
            point_id=entry.id,
            vector=vector,
            payload=payload,
        )


from fastapi import Depends


def get_semantic_cache(
    exact_cache: ExactCache = Depends(get_exact_cache),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    qdrant_service: QdrantService = Depends(get_qdrant_service),
) -> SemanticCache:
    """
    FastAPI dependency accessor for the SemanticCache engine.
    Supports dependency overrides for exact_cache, embedding_service, and qdrant_service.
    """
    return SemanticCache(
        exact_cache=exact_cache,
        embedding_service=embedding_service,
        qdrant_service=qdrant_service,
    )

