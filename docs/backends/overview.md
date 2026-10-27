# Storage Backends Overview

LongCache separates vector similarity search (handled by FAISS) from payload and metadata persistence (handled by pluggable backends).

## Supported Backends

| Backend | Persistence | Multi-Process / Cluster | Setup Required | Best For |
|---|---|---|---|---|
| **SQLite** (Default) | Disk | Single-node / Thread-safe | None (Automatic) | CLI tools, local apps, single containers |
| **Redis** | In-Memory / AOF | One LongCache process per cache | Redis service | Shared response persistence with a local index |
| **Memory** | None (RAM) | Single Process | None | Unit tests, CI/CD, ephemeral workers |

All FAISS indexes are local to the LongCache process. Redis response storage does not synchronize those indexes between workers. `backend="async_redis"` is rejected by `LongCache` in v0.1.1; the standalone async backend remains experimental.

---

## Configuring the Backend

The backend is configured via `CacheConfig`:

```python
from longcache import LongCache, CacheConfig

# 1. SQLite (default)
cache_sqlite = LongCache(CacheConfig(backend="sqlite", db_path=".longcache/cache.db"))

# 2. Redis
cache_redis = LongCache(CacheConfig(backend="redis", redis_url="redis://localhost:6379/0"))

# 3. Memory
cache_memory = LongCache(CacheConfig(backend="memory"))
```
