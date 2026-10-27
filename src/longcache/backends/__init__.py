"""Cache storage backends."""

from longcache.backends.base import AbstractCacheBackend
from longcache.backends.sqlite import SQLiteCacheBackend
from longcache.backends.memory import InMemoryCacheBackend

__all__ = ["AbstractCacheBackend", "SQLiteCacheBackend", "InMemoryCacheBackend"]


# Lazy import for Redis backend — only available when redis extra is installed
def get_redis_backend():
    """Import and return RedisCacheBackend. Raises ImportError if redis is not installed."""
    try:
        from longcache.backends.redis import RedisCacheBackend

        return RedisCacheBackend
    except ImportError:
        raise ImportError(
            "Redis backend requires the 'redis' extra. "
            "Install with: uv pip install longcache[redis]"
        )
