<p align="center">
  <img src="https://raw.githubusercontent.com/ENDEVSOLS/LongCache/main/assets/logo.png" alt="LongCache Logo" width="320"/>
</p>

<p align="center"><strong>Production-grade semantic similarity cache for LLM and RAG pipelines — cut repeat query costs by 40%+.</strong></p>

<p align="center">
<a href="https://pypi.org/project/longcache/"><img src="https://img.shields.io/pypi/v/longcache" alt="PyPI Version"></a>
<a href="https://pepy.tech/project/longcache"><img src="https://static.pepy.tech/badge/longcache" alt="Total Downloads"></a>
<a href="https://pepy.tech/project/longcache"><img src="https://static.pepy.tech/badge/longcache/month" alt="Monthly Downloads"></a>
<a href="https://github.com/ENDEVSOLS/LongCache/stargazers"><img src="https://img.shields.io/github/stars/ENDEVSOLS/LongCache?style=flat" alt="GitHub Stars"></a>
<a href="https://github.com/ENDEVSOLS/LongCache/actions/workflows/ci.yml"><img src="https://github.com/ENDEVSOLS/LongCache/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/pypi/pyversions/longcache" alt="Python Versions">
<a href="https://github.com/ENDEVSOLS/LongCache/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ENDEVSOLS/LongCache" alt="License"></a>
</p>

<p align="center">
<a href="https://endevsols.github.io/LongCache/"><strong>📖 Documentation</strong></a> &nbsp;·&nbsp;
<a href="https://endevsols.github.io/LongCache/getting-started/quickstart/"><strong>Quick Start</strong></a> &nbsp;·&nbsp;
<a href="https://endevsols.github.io/LongCache/api-reference/"><strong>API Reference</strong></a> &nbsp;·&nbsp;
<a href="https://github.com/ENDEVSOLS/LongCache/blob/main/CHANGELOG.md"><strong>Changelog</strong></a>
</p>

> *"What is your refund policy?" costs $0.003 the first time.*
> *It should cost $0.0000 every time after that.*

**LongCache** is a production-grade semantic similarity cache for LLM and RAG pipelines. Unlike string-match caches, it understands that *"How do I return a product?"* and *"What's your return process?"* deserve the same cached answer — and serves it in milliseconds.

---

## Why LongCache?

| Problem | Solution |
|---------|----------|
| Duplicate queries burn 30–60% of your LLM budget | Semantic cache intercepts similar queries before they reach the model |
| GPTCache is unmaintained with 50+ dependencies | LongCache has 4 core deps and is actively maintained |
| String-match caches miss paraphrased queries | FAISS + BGE-small understands semantic meaning, not just exact tokens |
| Most caches are coupled to one LLM provider | Works with LangChain, LangGraph, LlamaIndex, OpenAI, or raw functions |
| Cross-model cache poisoning is a silent bug | Composite key scoping isolates entries per model, prompt, and temperature |

**Result: Cut repeat query costs by 40%+ in production RAG applications.**

---

## Quick Start

### Installation

```bash
# Core — SQLite backend + FAISS + BGE-small (no external services needed)
uv pip install longcache

# With LangChain + LangGraph integration
uv pip install longcache[langchain]

# With Redis backend for multi-instance production
uv pip install longcache[redis]

# Everything
uv pip install longcache[all]
```

> **Offline-friendly**: LongCache embeds queries locally. After the BGE-small model weights are downloaded on first run, no internet connection is required.

### 30-Second Example

```python
from longcache import LongCache

cache = LongCache()

@cache.cached
def my_rag_chain(query: str) -> str:
    return expensive_llm_call(query)   # Your retrieval + LLM pipeline

# First call: MISS → runs full pipeline (~2 000 ms)
result1 = my_rag_chain("What is your refund policy?")

# Second call — exact query: HIT → instant (~5 ms)
result2 = my_rag_chain("What is your refund policy?")

# Third call — paraphrase: HIT → instant
result3 = my_rag_chain("How do I get a refund?")
```

Async functions are detected automatically — no additional setup needed:

```python
@cache.cached(namespace="faq", ttl=7200)
async def my_async_chain(query: str) -> str:
    return await async_llm_call(query)
```

---

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                        Query Lifecycle                       │
│                                                              │
│  User Query                                                  │
│       │                                                      │
│       ▼                                                      │
│  ┌───────────────────┐                                       │
│  │   LocalEmbedder   │  BAAI/bge-small-en-v1.5 — no API     │
│  │   (~5 ms, local)  │  call, runs fully offline            │
│  └────────┬──────────┘                                       │
│           │ 384-dim vector                                   │
│           ▼                                                  │
│  ┌───────────────────┐    HIT (similarity ≥ threshold)       │
│  │  SimilarityIndex  │ ─────────────────────────────────▶   │
│  │  FAISS + Backend  │         Return cached response        │
│  └────────┬──────────┘                                       │
│           │    MISS                                          │
│           ▼                                                  │
│  ┌───────────────────┐                                       │
│  │  Full RAG Chain   │  Retrieval + LLM (~1–3 s)            │
│  └────────┬──────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌───────────────────┐                                       │
│  │   CacheWriter     │  Stores vector + response + TTL       │
│  │                   │  + namespace + tags + context         │
│  └───────────────────┘                                       │
└──────────────────────────────────────────────────────────────┘
```

1. **Intercept** — queries are checked before they reach the LLM
2. **Embed** — the query is embedded locally in ~5 ms (BGE-small, no API call)
3. **Search** — FAISS searches for cosine similarity against cached vectors
4. **HIT** (similarity ≥ threshold): return cached response instantly
5. **MISS**: run the full pipeline, cache the result, and return it

---

## Configuration

### CacheConfig

All settings are controlled via the `CacheConfig` dataclass:

```python
from longcache import LongCache, CacheConfig

config = CacheConfig(
    similarity_threshold=0.92,           # 0.0–1.0; higher = stricter matching
    ttl_seconds=3600,                    # Entry lifetime in seconds (0 = never expire)
    max_entries=10_000,                  # Max entries per namespace (LRU eviction)
    namespace="default",                 # Logical isolation group
    embedder_model="BAAI/bge-small-en-v1.5",  # Local embedding model
    backend="sqlite",                    # "sqlite" | "redis" | "async_redis" | "memory"
    db_path=".longcache/cache.db",       # SQLite path (auto-creates parent dirs)
    redis_url="redis://localhost:6379/0",# Redis connection URL
    track_stats=True,                    # Track hits, misses, and cost savings
    faiss_index_type="flat",             # "flat" (exact) | "hnsw" (approximate, faster at scale)
    hnsw_m=32,                           # HNSW connections per node
    hnsw_ef_search=64,                   # HNSW search depth
    # Composite key context (isolates entries per model/persona)
    model="",                            # LLM model identifier (e.g. "gpt-4o")
    system_prompt="",                    # System prompt string
    temperature=-1.0,                    # Sampling temperature (-1.0 = unset)
)

cache = LongCache(config)
```

### From a Dictionary

```python
config = CacheConfig.from_dict({
    "similarity_threshold": 0.90,
    "backend": "redis",
    "ttl_seconds": 7200,
})
```

### From Environment Variables

```bash
export LONGCACHE_SIMILARITY_THRESHOLD=0.90
export LONGCACHE_BACKEND=redis
export LONGCACHE_TTL_SECONDS=7200
export LONGCACHE_REDIS_URL=redis://prod-redis:6379/0
export LONGCACHE_MODEL=gpt-4o
```

```python
config = CacheConfig.from_env()   # Reads all LONGCACHE_* env vars automatically
cache = LongCache(config)
```

### Similarity Threshold Guide

| Threshold | Behavior | Recommended Use Case |
|-----------|----------|----------------------|
| 0.99 | Near-exact match only | Precise FAQ lookup |
| 0.95 | Very similar phrasing | Customer support |
| **0.92** | **Default — balanced** | **General RAG pipelines** |
| 0.85 | Looser matching | Exploratory search |
| 0.75 | Topic-level clustering | Broad domain matching |

**Tip:** Start at 0.92 and adjust based on your observed hit rate vs. false-positive rate.

---

## Features ✨

### Core

- ✅ **Semantic Similarity Cache:** Understands paraphrased queries — “How do I return a product?” hits the same cache entry as “What’s your return process?”
- ✅ **Local Embedding:** BAAI/bge-small-en-v1.5 runs entirely on-device (~5 ms per query, no API call, fully offline after first download)
- ✅ **`@cache.cached` Decorator:** One-line caching for any sync or async function — auto-detects coroutines
- ✅ **`@cache.cached_stream` Decorator:** Caches streaming generator responses (sync and async); yields live on MISS, replays from cache on HIT
- ✅ **Composite Cache Keys:** Scopes entries per `model`, `system_prompt`, and `temperature` via SHA-256 fingerprint — prevents cross-model cache poisoning
- ✅ **Tag-Based Invalidation:** Attach tags to entries at write time; selectively invalidate by tag without clearing entire namespaces
- ✅ **Per-Entry TTL:** Override TTL per `set()` call or decorator — supports never-expire (`ttl=0`) and short-lived entries
- ✅ **LRU Eviction:** Configurable `max_entries` per namespace; oldest entries evicted automatically when the limit is reached

### Reliability

- ✅ **Thread-Safe Statistics:** Hit/miss counts protected by a dedicated lock; safe under 8+ concurrent threads
- ✅ **Persistent Stats:** Hit/miss/savings data written to SQLite and survives process restarts
- ✅ **Rebuild-Safe FAISS Index:** Index is fully rebuilt after TTL sweeps or evictions — prevents stale vector accumulation
- ✅ **Background TTL Sweeper:** Daemon thread runs every 5 minutes to delete expired entries; shuts down cleanly on GC
- ✅ **Correct Miss Counting:** `get()`, `aget()`, and all decorators record misses accurately in all code paths
- ✅ **Config Validation:** `CacheConfig` validates all fields on construction with clear error messages

### Integrations & Backends

- ✅ **SQLite Backend:** Zero external dependencies, WAL-mode thread safety, persistent stats, default for all deployments
- ✅ **Redis Backend:** Shared cache across multiple instances/pods, atomic TTL via `EXPIRE` (sync)
- ✅ **Async Redis Backend:** Non-blocking Redis — safe inside `asyncio` event loops (FastAPI, etc.)
- ✅ **In-Memory Backend:** Zero I/O, ideal for unit tests and CI/CD pipelines
- ✅ **LangChain Integration:** `CachedLLM` and `CachedChain` — drop-in wrappers with sync, async, and streaming support
- ✅ **LangGraph Integration:** `CachedState` node for graph-based agents; async variants included
- ✅ **LlamaIndex Integration:** `CachedQueryEngine` wraps any LlamaIndex query engine
- ✅ **OpenAI Integration:** `CachedOpenAI` wraps the OpenAI client directly
- ✅ **FastAPI Middleware:** `LongCacheMiddleware` caches entire HTTP routes; `X-LongCache: HIT` header on cached responses
- ✅ **CLI:** `longcache stats`, `clear`, `inspect`, `export` — manage the cache without writing code

---

## Backends

### SQLite (Default — Zero External Dependencies)

```python
cache = LongCache(CacheConfig(
    backend="sqlite",
    db_path=".longcache/cache.db",   # Auto-creates parent directories
))
```

| | |
|---|---|
| **Pros** | Zero external deps, persistent, WAL-mode thread safety, stats survive restarts |
| **Cons** | Single-machine only |
| **Best for** | Development, single-server production, CLI tools |

### Redis (Production Multi-Instance)

```bash
uv pip install longcache[redis]
```

```python
cache = LongCache(CacheConfig(
    backend="redis",
    redis_url="redis://localhost:6379/0",
    max_entries=100_000,
))
```

| | |
|---|---|
| **Pros** | Shared across instances, atomic TTL via `EXPIRE`, highly scalable |
| **Cons** | Requires Redis; blocking I/O (not suited for `async` event loops) |
| **Best for** | Multi-instance Kubernetes deployments, microservices (sync) |

### Async Redis (Production Async Apps)

```python
cache = LongCache(CacheConfig(
    backend="async_redis",
    redis_url="redis://localhost:6379/0",
))
```

| | |
|---|---|
| **Pros** | Non-blocking — safe inside `async` event loops (FastAPI, etc.) |
| **Best for** | Fully async production deployments |

### In-Memory (Testing & CI)

```python
cache = LongCache(CacheConfig(backend="memory"))
```

| | |
|---|---|
| **Pros** | Fastest possible, zero I/O, no files created |
| **Cons** | Lost on process restart, single-process only |
| **Best for** | Unit tests, CI/CD pipelines, benchmarks |

---

## Integrations

### LangChain

```bash
uv pip install longcache[langchain]
```

#### `CachedLLM` — Wrap Any LLM

```python
from langchain_openai import ChatOpenAI
from longcache.integrations.langchain import CachedLLM

llm = CachedLLM(ChatOpenAI(model="gpt-4o"), cache=LongCache())

response = llm.invoke("What is the refund policy?")       # sync
response = await llm.ainvoke("What is the refund policy?") # async

for chunk in llm.stream("What is RAG?"):                  # sync streaming
    print(chunk, end="")

async for chunk in llm.astream("What is RAG?"):           # async streaming
    print(chunk, end="")
```

#### `CachedChain` — Wrap Any LCEL Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from longcache.integrations.langchain import CachedChain

chain = ChatPromptTemplate.from_template("Answer: {question}") | ChatOpenAI() | StrOutputParser()
cached_chain = CachedChain(chain, cache=LongCache(), query_key="question")

result = cached_chain.invoke({"question": "What is RAG?"})
result = await cached_chain.ainvoke({"question": "What is RAG?"})
```

### LangGraph

```python
from langgraph.graph import StateGraph
from longcache import LongCache
from longcache.integrations.langgraph import CachedState

cache = LongCache()

def cache_node(state: dict) -> dict:
    return CachedState.check(state, cache, query_key="messages")

def llm_node(state: dict) -> dict:
    response = my_llm_call(state["messages"])
    CachedState.save(state, response, cache)
    return {"response": response}

def should_use_llm(state: dict) -> str:
    return "llm" if not state.get("__cache_hit") else "end"

graph = StateGraph(dict)
graph.add_node("cache", cache_node)
graph.add_node("llm", llm_node)
graph.add_conditional_edges("cache", should_use_llm)
graph.set_entry_point("cache")
```

Async variants: `CachedState.acheck()` and `CachedState.asave()`.

### LlamaIndex

```bash
uv pip install longcache[llamaindex]
```

```python
from llama_index.core import VectorStoreIndex
from longcache.integrations.llamaindex import CachedQueryEngine

engine = VectorStoreIndex.from_documents(documents).as_query_engine()
cached_engine = CachedQueryEngine(engine, cache=LongCache())
response = cached_engine.query("What is the refund policy?")
```

### OpenAI (Direct)

```bash
uv pip install longcache[openai]
```

```python
from openai import OpenAI
from longcache.integrations.openai import CachedOpenAI

client = CachedOpenAI(OpenAI(api_key="sk-..."), cache=LongCache())
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is RAG?"}],
)
```

---

## FastAPI Middleware

```bash
uv pip install longcache[server]
```

```python
from fastapi import FastAPI
from longcache import LongCache
from longcache.middleware import LongCacheMiddleware

app = FastAPI()
app.add_middleware(
    LongCacheMiddleware,
    cache=LongCache(),
    cached_routes={"/v1/chat/completions"},
)

# All POST requests to /v1/chat/completions are now semantically cached.
# X-LongCache: HIT  →  cached response served
# X-LongCache: (absent) →  fresh LLM response
```

### Streaming Cache Adapter

```python
from longcache.middleware import StreamingCacheAdapter

def my_stream(query: str):
    for chunk in llm.stream(query):
        yield chunk

adapter = StreamingCacheAdapter(cache, my_stream)

# MISS — streams from LLM, saves full response in cache
for token in adapter.stream("What is RAG?"):
    print(token, end="")

# HIT — replays cached response with streaming pacing
for token in adapter.stream("What is RAG?"):
    print(token, end="")
```

---

## CLI

LongCache ships a command-line tool for cache management and inspection:

```bash
# View cache statistics (hits, misses, hit rate, estimated savings)
longcache stats
longcache stats --namespace faq

# Clear a namespace
longcache clear --namespace faq --yes

# Clear the entire cache
longcache clear --all --yes

# Inspect the nearest cached entry for a query
longcache inspect --query "refund policy"

# Export cache entries as JSON
longcache export --format json
longcache export --namespace faq -o faq_cache.json
```

---

## API Reference

### `LongCache(config=None)`

Main facade class. Accepts an optional `CacheConfig`; uses production-safe defaults otherwise.

#### Core Methods

| Method | Description |
|--------|-------------|
| `get(query, namespace, model, system_prompt, temperature)` | Check cache. Returns cached response or `None`. Increments miss count on MISS. |
| `set(query, response, namespace, ttl, tags, model, system_prompt, temperature)` | Store a query-response pair with optional tags and context scope. |
| `aget(...)` | Async version of `get()`. Non-blocking. |
| `aset(...)` | Async version of `set()`. Non-blocking. |
| `cached(func, *, namespace, ttl, tags, model, system_prompt, temperature)` | Decorator for sync **and** async functions. |
| `cached_stream(func, *, namespace, ttl, tags, model, system_prompt, temperature)` | Decorator for sync/async **generator** functions (streaming LLMs). |
| `invalidate(namespace, model, system_prompt, temperature)` | Delete all entries in a namespace. Returns count deleted. |
| `invalidate_by_tag(tag, namespace, ...)` | Delete all entries carrying a given tag. Returns count deleted. |
| `clear()` | Clear the entire cache across all namespaces. Returns count. |
| `stats()` | Returns hit rate, miss count, estimated tokens saved, estimated cost saved, and total entries. |
| `inspect(query, namespace, ...)` | Find nearest cached entry without returning it. Useful for debugging. |
| `record_hit()` | Manually record a cache hit (thread-safe). For custom integrations. |
| `record_miss(elapsed_ms)` | Manually record a cache miss with elapsed generation time (thread-safe). |

#### `@cache.cached` Decorator Options

```python
@cache.cached(
    namespace="faq",         # Logical isolation group
    ttl=7200,                # Entry lifetime override (seconds)
    tags=["billing"],        # Tags for selective invalidation
    model="gpt-4o",          # Scopes entries to this model
    system_prompt="...",     # Scopes entries to this system prompt
    temperature=0.0,         # Scopes entries to this temperature
)
def my_chain(query: str) -> str:
    ...
```

Works identically for `@cache.cached_stream`.

#### `stats()` Return Value

```python
{
    "hits":                    420,
    "misses":                  180,
    "hit_rate":                0.70,
    "total_saved_ms":          840_000.0,
    "estimated_tokens_saved":  630_000,
    "estimated_cost_saved_usd": 18.90,
    "total_entries":           95,
}
```

Statistics are protected by a reentrant threading lock and persisted to the SQLite backend — they survive process restarts.

---

### `CacheConfig` Field Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `similarity_threshold` | `float` | `0.92` | Cosine similarity threshold for cache hits (0.0–1.0) |
| `ttl_seconds` | `int` | `3600` | Entry lifetime in seconds. `0` = never expire |
| `max_entries` | `int` | `10_000` | Max entries per namespace (LRU eviction) |
| `namespace` | `str` | `"default"` | Default namespace |
| `embedder_model` | `str` | `"BAAI/bge-small-en-v1.5"` | Local sentence-transformers model |
| `backend` | `str` | `"sqlite"` | `"sqlite"`, `"redis"`, `"async_redis"`, `"memory"` |
| `db_path` | `str` | `".longcache/cache.db"` | SQLite database path |
| `redis_url` | `str` | `"redis://localhost:6379/0"` | Redis connection URL |
| `track_stats` | `bool` | `True` | Track hit/miss stats (thread-safe, persisted in SQLite) |
| `faiss_index_type` | `str` | `"flat"` | `"flat"` (exact) or `"hnsw"` (approximate) |
| `hnsw_m` | `int` | `32` | HNSW connections per node (higher = more accurate, more memory) |
| `hnsw_ef_search` | `int` | `64` | HNSW search depth (higher = more accurate, slower) |
| `model` | `str` | `""` | Global model identifier for composite key scoping |
| `system_prompt` | `str` | `""` | Global system prompt for composite key scoping |
| `temperature` | `float` | `-1.0` | Global temperature for composite key scoping (`-1.0` = unset) |

---

### `CacheEntry` Field Reference

Returned by `inspect()` and used internally by all backends.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int \| None` | Backend-assigned unique identifier |
| `namespace` | `str` | Effective namespace (includes composite key suffix if set) |
| `query` | `str` | Original query text |
| `vector` | `np.ndarray` | 384-dim L2-normalized float32 embedding |
| `response` | `Any` | Cached LLM response (must be JSON-serializable) |
| `created_at` | `float` | Unix timestamp of entry creation |
| `expires_at` | `float \| None` | Unix timestamp of expiry (`None` = no expiry) |
| `hit_count` | `int` | Number of times this entry has been served |
| `model` | `str` | Model identifier at write time |
| `system_prompt` | `str` | System prompt at write time |
| `temperature` | `float` | Temperature at write time (`-1.0` = unset) |
| `tags` | `list[str]` | Tags for selective invalidation |

---

## Advanced Usage

### Per-Namespace Threshold Tuning

```python
faq_cache = LongCache(CacheConfig(namespace="faq",     similarity_threshold=0.95))
chat_cache = LongCache(CacheConfig(namespace="general", similarity_threshold=0.85))
```

### Per-Entry TTL Override

```python
cache.set("volatile_query", response, ttl=300)      # 5-minute TTL
cache.set("stable_query",   response, ttl=86400)    # 24-hour TTL
cache.set("permanent",      response, ttl=0)        # Never expires
```

### Cache Invalidation Patterns

```python
# Clear a specific namespace
cache.invalidate("faq")

# Clear entries scoped to a specific model
cache.invalidate("faq", model="gpt-4o")

# Invalidate by tag (e.g., after a document update)
cache.invalidate_by_tag("doc:refund_v2")

# Nuclear option — clear everything
cache.clear()
```

### Async Manual Control

```python
cached = await cache.aget("What is your refund policy?")
if cached is None:
    result = await my_pipeline("What is your refund policy?")
    await cache.aset("What is your refund policy?", result, tags=["policy"])
```

### HNSW Index for Large Caches (100K+ Entries)

```python
cache = LongCache(CacheConfig(
    faiss_index_type="hnsw",
    hnsw_m=32,           # More connections → more accurate, more memory
    hnsw_ef_search=64,   # Higher → more accurate, slower
    max_entries=500_000,
))
```

### Inspecting the Cache

```python
entry = cache.inspect("refund policy")
# {
#   'id': 42,
#   'query': 'What is your refund policy?',
#   'namespace': 'default',
#   'hit_count': 7,
#   'tags': ['billing', 'policy'],
#   'model': 'gpt-4o',
#   'created_at': 1700000000.0,
#   'expires_at': None,
#   'is_expired': False,
# }
```

### Custom Integration: Manual Hit/Miss Recording

For custom wrappers that call `get()`/`set()` manually:

```python
result = cache.get(query, track_miss=False)
if result is not None:
    cache.record_hit()
    return result

t0 = time.time()
result = my_pipeline(query)
cache.record_miss(elapsed_ms=(time.time() - t0) * 1000)
cache.set(query, result)
return result
```

---

## Architecture

### Component Overview

```
src/longcache/
├── __init__.py              # Public API: LongCache, CacheConfig
├── config.py                # CacheConfig dataclass + from_dict/from_env
├── cli.py                   # CLI: stats, clear, inspect, export
├── middleware.py             # FastAPI/Starlette ASGI middleware + StreamingCacheAdapter
├── core/
│   ├── entry.py             # CacheEntry dataclass (query, vector, response, tags, context)
│   ├── embedder.py          # LocalEmbedder — BGE-small singleton, lazy-loaded
│   ├── index.py             # SimilarityIndex — FAISS Flat/HNSW, rebuild-safe
│   └── cache.py             # LongCache — main orchestrator, thread-safe stats
├── backends/
│   ├── base.py              # AbstractCacheBackend ABC
│   ├── sqlite.py            # SQLiteCacheBackend — WAL mode, tag table, stats persistence
│   ├── redis.py             # RedisCacheBackend — sync, tag support, atomic TTL
│   ├── async_redis.py       # AsyncRedisCacheBackend — async, non-blocking
│   └── memory.py            # InMemoryCacheBackend — testing, zero I/O
└── integrations/
    ├── langchain.py         # CachedLLM, CachedChain
    ├── langgraph.py         # CachedState
    ├── llamaindex.py        # CachedQueryEngine
    └── openai.py            # CachedOpenAI
```

### Embedding Model

LongCache uses **BAAI/bge-small-en-v1.5** by default:

- **384-dimensional** vectors — compact and fast
- **Sub-5ms** embedding on modern CPU hardware
- **Fully offline** after initial weight download — no API calls ever
- **L2-normalized** — inner product equals cosine similarity in FAISS
- **Singleton pattern** — the model is loaded once and reused across all instances

### FAISS Index

| Index | Config Value | Characteristics |
|-------|-------------|-----------------|
| `IndexFlatIP` | `"flat"` | Exact search, deterministic, best for ≤ 100K entries |
| `IndexHNSWFlat` | `"hnsw"` | Approximate search, sub-ms at any scale, best for 100K+ entries |

The FAISS index is **rebuild-safe**: after TTL sweeps, namespace invalidations, or evictions, the index is fully rebuilt from the backend rather than incrementally patched — preventing stale vector accumulation.

### Thread Safety

- **Stats** (`hits`, `misses`, `total_saved_ms`) are protected by a dedicated `threading.Lock()`.
- **SQLite backend** uses WAL mode and a per-connection `threading.RLock()`.
- **The TTL sweeper** runs as a daemon thread and signals shutdown cleanly via `threading.Event`.
- **`get()`/`aget()`** correctly record misses in a thread-safe manner.

### Background TTL Sweeper

A lightweight daemon thread runs every 5 minutes to delete expired entries and rebuild the FAISS index. Starts automatically when `ttl_seconds > 0`. Shuts down gracefully when the `LongCache` instance is garbage-collected.

### Persistent Statistics

Hit/miss statistics are persisted to the SQLite backend and survive process restarts. All stat mutations go through a reentrant lock to prevent race conditions under concurrent load.

---

## Performance

### Expected Cache Hit Rates

| Scenario | Hit Rate | Cost Savings |
|----------|----------|--------------|
| FAQ chatbot | 60–80% | 40–60% |
| Customer support | 40–60% | 25–40% |
| Internal knowledge base | 50–70% | 30–50% |
| General RAG pipeline | 30–50% | 20–35% |

### Latency Comparison

| Operation | Typical Latency |
|-----------|----------------|
| Full RAG pipeline (retrieval + LLM) | 1,000–3,000 ms |
| LongCache HIT (embed + FAISS search) | 5–15 ms |
| LongCache MISS overhead | 5–10 ms |

---

## Project Structure

```
longcache/
├── pyproject.toml
├── README.md
├── src/
│   └── longcache/
│       ├── __init__.py
│       ├── config.py
│       ├── cli.py
│       ├── middleware.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── entry.py
│       │   ├── embedder.py
│       │   ├── index.py
│       │   └── cache.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── sqlite.py
│       │   ├── redis.py
│       │   ├── async_redis.py
│       │   └── memory.py
│       └── integrations/
│           ├── __init__.py
│           ├── langchain.py
│           ├── langgraph.py
│           ├── llamaindex.py
│           └── openai.py
├── tests/
│   ├── conftest.py
│   ├── test_embedder.py
│   ├── test_index.py
│   ├── test_cache.py
│   ├── test_backends.py
│   ├── test_langchain.py
│   ├── test_middleware.py
│   └── test_cli.py
└── examples/
    ├── basic_langchain.py
    ├── fastapi_middleware.py
    └── redis_production.py
```

---

## Dependencies

### Core (Required)

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | `>=2.3.0` | Local embedding — BGE-small |
| `faiss-cpu` | `>=1.7.4` | Similarity search (Flat & HNSW) |
| `numpy` | `>=1.24.0,<2.0.0` | Vector math |
| `torch` | CPU (via uv) | sentence-transformers runtime |
| `click` | `>=8.1.0` | CLI framework |

> **Note**: `numpy<2.0.0` is required — numpy 2.0 breaks ABI compatibility with `faiss-cpu` and `torch`.

### Optional Extras

| Extra | Package(s) | Purpose |
|-------|-----------|---------|
| `[redis]` | `redis>=5.0.0,<6.0.0` | Redis & async Redis backends |
| `[langchain]` | `langchain-core>=1.0.0`, `langgraph>=1.0.0` | LangChain + LangGraph |
| `[llamaindex]` | `llama-index-core>=0.10.0` | LlamaIndex integration |
| `[openai]` | `openai>=1.0.0,<2.0.0` | Direct OpenAI integration |
| `[server]` | `fastapi>=0.109.0`, `starlette>=0.36.0`, `uvicorn>=0.27.0` | ASGI middleware |
| `[all]` | All of the above | Everything |

### Development

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | `>=8.0.0` | Test runner |
| `pytest-asyncio` | `>=0.23.0` | Async test support |
| `pytest-cov` | `>=5.0.0` | Coverage reporting |
| `httpx` | `>=0.27.0` | ASGI test client |
| `ruff` | `>=0.4.0` | Linter / formatter |

---

## Running Tests

```bash
# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=longcache --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_cache.py

# Verbose output
uv run pytest -v
```

---

## Contributing & Community

We welcome contributions! Please see:
- [CONTRIBUTING.md](CONTRIBUTING.md) — Setup, testing, and pull request guidelines.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — Contributor Covenant v2.1 standards.
- [SECURITY.md](SECURITY.md) — Vulnerability reporting and coordinated disclosure.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 ENDEVSOLS
