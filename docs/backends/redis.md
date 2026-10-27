# Redis Backend

LongCache can persist responses in Redis. In v0.1.1, use one LongCache process per Redis-backed cache: each process owns a local FAISS index, and Redis does not synchronize indexes between workers.

## Installation

Ensure the `redis` extra is installed:

```bash
uv pip install "longcache[redis]"
# or
pip install "longcache[redis]"
```

## Synchronous Redis

```python
from longcache import LongCache, CacheConfig

config = CacheConfig(
    backend="redis",
    redis_url="redis://:secret@redis-cluster.internal:6379/0",
    namespace="production_rag",
    ttl_seconds=7200,
)

cache = LongCache(config)

@cache.cached
def ask(query: str) -> str:
    return call_model(query)
```

## Async support

`LongCache(CacheConfig(backend="async_redis"))` raises a clear error in v0.1.1 because the async backend is not connected to the complete cache/index lifecycle. `AsyncRedisCacheBackend` remains available for standalone experimental use. Use `backend="redis"` for the supported public cache path.

Close cache instances with `cache.close()` or a `with LongCache(config) as cache:` block to stop background maintenance and release backend resources.

## Local Development with Docker

Run Redis locally with Docker Compose:

```bash
docker compose up -d redis
LONGCACHE_TEST_REDIS_URL=redis://localhost:6379/0 uv run --no-sync pytest tests/test_redis_backend.py -q
```
