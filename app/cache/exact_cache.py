from abc import ABC, abstractmethod
from typing import Dict, Optional
from app.cache.models import CacheEntry


class ExactCache(ABC):
    """Abstract interface for exact-match request caching."""

    @abstractmethod
    async def get(self, request_hash: str) -> Optional[CacheEntry]:
        """Retrieve a cached entry by its original request SHA-256 hash."""
        pass

    @abstractmethod
    async def set(self, request_hash: str, entry: CacheEntry) -> None:
        """Store a cache entry indexed by its original request SHA-256 hash."""
        pass

    @abstractmethod
    async def delete(self, request_hash: str) -> None:
        """Remove a cached entry by its key."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries from the cache."""
        pass


class InMemoryExactCache(ExactCache):
    """
    In-memory implementation of the exact cache.
    Provides O(1) exact-match lookup before semantic embeddings.
    """

    def __init__(self) -> None:
        self._store: Dict[str, CacheEntry] = {}

    async def get(self, request_hash: str) -> Optional[CacheEntry]:
        return self._store.get(request_hash)

    async def set(self, request_hash: str, entry: CacheEntry) -> None:
        self._store[request_hash] = entry

    async def delete(self, request_hash: str) -> None:
        self._store.pop(request_hash, None)

    async def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# Global singleton instance for the in-memory cache
_default_exact_cache = InMemoryExactCache()


def get_exact_cache() -> ExactCache:
    """FastAPI dependency for accessing the exact cache instance."""
    return _default_exact_cache
