# Redis Backend

For multi-container deployments, Kubernetes clusters, or horizontally scaled workers, LongCache provides high-throughput Redis backends.

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

## Asynchronous Redis (`async_redis`)

For `asyncio`-first frameworks like FastAPI, Starlette, or Sanic:

```python
import asyncio
from longcache import LongCache, CacheConfig

config = CacheConfig(
    backend="async_redis",
    redis_url="redis://localhost:6379/0",
    namespace="async_api"
)

cache = LongCache(config)

@cache.cached
async def ask_async(query: str) -> str:
    return await async_llm_client.generate(query)
```

## Local Development with Docker

Run Redis locally with Docker Compose:

```bash
docker compose up -d redis
```
