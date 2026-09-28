import asyncio
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator, Dict, Optional
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LockAcquisitionError(Exception):
    """Raised when acquiring a distributed or local lock times out."""
    pass


class LockManager:
    """Base interface for single-flight concurrency lock managers."""

    @asynccontextmanager
    async def lock(
        self,
        key: str,
        timeout: float = 15.0,
        lease_time: float = 30.0,
    ) -> AsyncGenerator[None, None]:
        """
        Acquire a lock for the given key.
        :param key: Unique resource identifier to lock (e.g. lock:request:<hash>)
        :param timeout: Maximum seconds to wait to acquire the lock before raising LockAcquisitionError.
        :param lease_time: Lock expiration (TTL) to prevent deadlocks if holder crashes.
        """
        yield


class RedisLockManager(LockManager):
    """
    Distributed lock manager using Redis.
    Guarantees cross-process and cross-worker single-flight request coalescing.
    """

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self.client = redis_client

    @asynccontextmanager
    async def lock(
        self,
        key: str,
        timeout: float = 15.0,
        lease_time: float = 30.0,
    ) -> AsyncGenerator[None, None]:
        # redis-py lock handles atomic acquisition (NX PX) and token ownership
        redis_lock = self.client.lock(
            name=f"lock:{key}",
            timeout=lease_time,
            blocking_timeout=timeout,
        )
        acquired = False
        try:
            acquired = await redis_lock.acquire()
            if not acquired:
                raise LockAcquisitionError(
                    f"Timed out waiting {timeout}s to acquire Redis lock for '{key}'"
                )
            yield
        except (RedisConnectionError, RedisTimeoutError) as exc:
            logger.warning("Redis lock error for key '%s': %s", key, exc)
            raise LockAcquisitionError(f"Redis connection error during lock: {exc}") from exc
        finally:
            if acquired:
                try:
                    await redis_lock.release()
                except Exception as exc:
                    logger.debug("Failed releasing Redis lock '%s': %s", key, exc)

    async def ping(self) -> bool:
        """Verify Redis connectivity."""
        try:
            return bool(await self.client.ping())
        except Exception:
            return False

    async def close(self) -> None:
        """Close connection pool."""
        await self.client.aclose()


class InMemoryLockManager(LockManager):
    """
    In-memory single-flight lock manager using asyncio.Lock.
    Used as an automatic fallback when Redis is unavailable, or during lightweight local tests.
    """

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._ref_counts: Dict[str, int] = {}
        self._guard: Optional[asyncio.Lock] = None

    def _get_guard(self) -> asyncio.Lock:
        if self._guard is None:
            self._guard = asyncio.Lock()
        return self._guard

    @asynccontextmanager
    async def lock(
        self,
        key: str,
        timeout: float = 15.0,
        lease_time: float = 30.0,
    ) -> AsyncGenerator[None, None]:
        guard = self._get_guard()
        async with guard:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
                self._ref_counts[key] = 0
            self._ref_counts[key] += 1
            lock = self._locks[key]

        acquired = False
        try:
            try:
                await asyncio.wait_for(lock.acquire(), timeout=timeout)
                acquired = True
            except asyncio.TimeoutError:
                raise LockAcquisitionError(
                    f"Timed out waiting {timeout}s for in-memory lock on key '{key}'"
                )
            yield
        finally:
            if acquired:
                lock.release()
            async with guard:
                self._ref_counts[key] -= 1
                if self._ref_counts[key] <= 0:
                    self._locks.pop(key, None)
                    self._ref_counts.pop(key, None)


# Global instances and loop-tracking
_redis_lock_manager: Optional[RedisLockManager] = None
_redis_loop: Optional[asyncio.AbstractEventLoop] = None
_in_memory_lock_manager = InMemoryLockManager()


async def reset_lock_manager() -> None:
    """Reset global lock manager instances (used in tests and shutdown)."""
    global _redis_lock_manager, _redis_loop, _in_memory_lock_manager
    if _redis_lock_manager is not None:
        try:
            await _redis_lock_manager.close()
        except Exception:
            pass
        _redis_lock_manager = None
    _redis_loop = None
    _in_memory_lock_manager = InMemoryLockManager()


async def get_lock_manager() -> LockManager:
    """
    FastAPI dependency / factory.
    Returns RedisLockManager if Redis is available and reachable;
    otherwise gracefully falls back to InMemoryLockManager.
    Recreates connection if asyncio event loop has changed (e.g. across test cases).
    """
    global _redis_lock_manager, _redis_loop
    settings = get_settings()

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _redis_lock_manager is not None and _redis_loop != current_loop:
        _redis_lock_manager = None
        _redis_loop = None

    if settings.REDIS_URL:
        if _redis_lock_manager is None:
            try:
                client = aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=2.0,
                )
                manager = RedisLockManager(client)
                if await manager.ping():
                    _redis_lock_manager = manager
                    _redis_loop = current_loop
                    logger.info("Connected to Redis single-flight lock manager at %s", settings.REDIS_URL)
                else:
                    logger.warning("Redis at %s did not respond to PING; falling back to InMemoryLockManager", settings.REDIS_URL)
            except Exception as exc:
                logger.warning("Could not initialize Redis lock manager (%s); using in-memory fallback.", exc)

        if _redis_lock_manager is not None:
            return _redis_lock_manager

    return _in_memory_lock_manager

