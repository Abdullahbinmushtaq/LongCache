# Changelog

All notable changes are recorded here using [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-10-27

### Fixed

- Similarity lookup now considers other eligible entries when the closest vector is in another namespace, expired, or missing.
- Rebuilds restore all live namespaces after restart and invalidation, and preserve configured HNSW indexes.
- Cache inspection no longer records a hit. In-memory entry hit counts now increase once per served hit.
- The documented OpenAI `chat.completions.create` path works for supported sync and async clients. Supported requests are scoped by model, instructions, and temperature, and SDK response objects can survive a SQLite restart.
- Additional function arguments bypass decorator caching until those arguments can be represented in a stable cache identity.
- The ASGI middleware bypasses unsupported chat shapes and requests bearing Authorization, Cookie, or X-API-Key headers.
- Cache instances can be closed explicitly or used as context managers to stop background work.
- The package, CLI, and wheel report the same version.

### Compatibility and migration

- The OpenAI and ASGI integrations use new internal namespaces. Entries written by the old under-scoped integrations are not reused; the first eligible request after upgrade is a miss.
- Unsupported `backend="async_redis"` through `LongCache` now fails immediately. `AsyncRedisCacheBackend` remains available as a standalone backend while the full orchestration contract is developed.
- Redis stores responses but does not synchronize in-memory FAISS indexes between processes. Use one LongCache process per Redis-backed cache until multi-instance consistency is implemented.
- Python 3.10 remains supported in 0.1.1. The planned minimum of Python 3.11 is scheduled for 0.2.0.

### Validation

- Run the unit and integration suite with `uv run pytest -q`; Redis integration tests require a reachable service at `LONGCACHE_TEST_REDIS_URL`.
- Run `uv run ruff check src tests` and `uv build` from the package directory.
- Run real-model integration tests explicitly with `LONGCACHE_TEST_MODEL=1 uv run pytest tests/test_embedder.py -q`; these may download model weights.

## [0.1.0] - 2026-10-26

### Added

- **Core Semantic Cache Engine:**
  - Fast local embeddings using `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors) running fully offline.
  - FAISS index integration with support for exact flat cosine similarity and scalable HNSW indexing.
  - Transparent sync and async decorator `@cache.cached` with parameter hashing, TTL expiry, and namespace scoping.
  - Composite cache keys to prevent cross-model poisoning (scoped by model, system prompt, temperature, and query).
- **Pluggable Storage Backends:**
  - `SQLiteBackend`: Zero-configuration local disk persistence with transactional safety and WAL mode.
  - `RedisBackend` & `AsyncRedisBackend`: High-throughput distributed caching for multi-container and cluster deployments.
  - `MemoryBackend`: High-speed in-memory storage for unit tests, ephemeral workers, and local experiments.
- **Framework Integrations:**
  - `langchain`: Semantic caching adapter for LangChain LLMs and runnables.
  - `langgraph`: Contextual state cache node for multi-agent workflows.
  - `llamaindex`: Semantic cache wrapper for LlamaIndex query engines.
  - `openai`: Drop-in wrapper for `openai.OpenAI` and `openai.AsyncOpenAI` chat completions.
  - `fastapi`: ASGI HTTP middleware for automatic semantic caching of REST / SSE endpoints.
- **CLI & Diagnostics:**
  - `longcache stats`: Real-time cache hit rate, token savings, and cost metrics.
  - `longcache clear`: Scoped namespace and total cache eviction.
  - `longcache inspect`: Query lookup, similarity breakdown, and payload inspection.
  - `longcache export`: JSON and CSV dataset exporting for cache analytics.
- **Developer Tools & CI/CD:**
  - CI workflow with Python 3.10, 3.11, and 3.12 test matrix.
  - Automated GitHub Pages documentation deployment with Material for MkDocs.
  - Auto-tagging and GitHub release generation on main branch merge.
  - Automated PyPI publishing pipeline with Trusted Publisher authentication.
