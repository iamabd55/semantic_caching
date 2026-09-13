from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.cache.canonicalization import canonicalize_request
from app.cache.exact_cache import ExactCache, get_exact_cache
from app.cache.hashing import compute_request_hash
from app.cache.models import CacheEntry
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

    async def lookup(self, request: ChatCompletionRequest) -> CacheResult:
        """
        Execute tiered cache lookup: Exact Match -> Semantic Match.
        """
        request_hash = compute_request_hash(request)

        # 1. Tier 1: Exact Hash Fast-Path Lookup (O(1), < 1ms)
        exact_entry = await self.exact_cache.get(request_hash)
        if exact_entry is not None:
            return CacheResult(
                hit=True,
                hit_type="EXACT",
                response=exact_entry.response,
                similarity_score=1.0,
                embedding_latency_ms=0.0,
            )

        # 2. Tier 2: Semantic Similarity Lookup via Qdrant
        canonical_text = canonicalize_request(request)
        query_vector, latency_ms = self.embedding_service.embed_with_latency(canonical_text)

        matches = self.qdrant_service.search_nearest(query_vector=query_vector, limit=1)

        if matches:
            top_match = matches[0]
            score = top_match["score"]

            if score >= self.threshold:
                payload = top_match.get("payload", {})
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
    ) -> None:
        """
        Store a new completion response in both ExactCache and Qdrant.
        """
        request_hash = compute_request_hash(request)
        canonical_text = canonicalize_request(request)
        vector = self.embedding_service.embed(canonical_text)

        entry = CacheEntry(
            original_request_hash=request_hash,
            response=response,
            model=request.model,
            normalized_request=canonical_text,
            embedding=vector,
        )

        # 1. Store in Exact Cache
        await self.exact_cache.set(request_hash, entry)

        # 2. Store in Qdrant Vector Store
        payload = {
            "original_request_hash": request_hash,
            "normalized_request": canonical_text,
            "response": response.model_dump(),
            "model": request.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
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

