# API Reference

## `LongCache`

The core semantic cache orchestrator.

```python
from longcache import LongCache, CacheConfig

cache = LongCache(config=CacheConfig())
```

### Constructor

`LongCache(config: Optional[CacheConfig] = None)`

* **`config`**: An instance of `CacheConfig`. If omitted, default SQLite configuration is used.

---

### Methods

#### `get(query: str, namespace: Optional[str] = None) -> Optional[str]`
Checks the cache for a semantically similar query matching or exceeding `similarity_threshold`.

* **Parameters:**
    * `query` (`str`): The query string to look up.
    * `namespace` (`str`, optional): Override the default namespace.
* **Returns:**
    * Cached response string if HIT, or `None` if MISS.

---

#### `set(query: str, response: str, namespace: Optional[str] = None, ttl: Optional[int] = None) -> None`
Stores a query vector, text, and response payload in the cache.

* **Parameters:**
    * `query` (`str`): Original user query.
    * `response` (`str`): Generated LLM response.
    * `namespace` (`str`, optional): Specific namespace to store under.
    * `ttl` (`int`, optional): Time to live in seconds. Overrides global `ttl_seconds`.

---

#### `cached(fn=None, *, namespace=None, ttl=None)`
Decorator for synchronous and asynchronous functions. Automatically extracts query string, handles parameter hashing, and returns cached values.

```python
@cache.cached(namespace="support", ttl=3600)
def ask_question(question: str) -> str:
    return call_llm(question)
```

---

#### `stats(namespace: Optional[str] = None) -> dict`
Returns real-time cache statistics:

```python
{
    "hits": 1420,
    "misses": 850,
    "hit_rate": 0.625,
    "total_entries": 850,
    "estimated_tokens_saved": 710000,
    "estimated_cost_saved_usd": 2.13,
}
```

---

#### `clear(namespace: Optional[str] = None, all_namespaces: bool = False) -> None`
Evicts entries from the cache.

---

## `CacheConfig`

Configuration dataclass for `LongCache`.

```python
from longcache import CacheConfig

config = CacheConfig(
    similarity_threshold=0.92,
    ttl_seconds=3600,
    max_entries=10000,
    namespace="default",
    embedder_model="BAAI/bge-small-en-v1.5",
    backend="sqlite",
    db_path=".longcache/cache.db",
    redis_url="redis://localhost:6379/0",
    track_stats=True,
    faiss_index_type="flat",
    hnsw_m=32,
    hnsw_ef_search=64,
    model="",
    system_prompt="",
    temperature=-1.0,
)
```

### Parameter Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `similarity_threshold` | `float` | `0.92` | Minimum cosine similarity (0.0 to 1.0) required for a cache hit |
| `ttl_seconds` | `int` | `3600` | Expiration time in seconds (0 = no expiry) |
| `max_entries` | `int` | `10000` | Maximum entries stored per namespace before eviction |
| `namespace` | `str` | `"default"` | Logical partition namespace |
| `embedder_model` | `str` | `"BAAI/bge-small-en-v1.5"` | HuggingFace sentence-transformers model name |
| `backend` | `str` | `"sqlite"` | Supported storage backend: `"sqlite"`, `"redis"`, or `"memory"`; `"async_redis"` fails fast in v0.1.1 |
| `db_path` | `str` | `".longcache/cache.db"` | Path to SQLite database file |
| `redis_url` | `str` | `"redis://localhost:6379/0"` | Redis connection URL |
| `track_stats` | `bool` | `True` | Track hits, misses, and cost statistics |
| `faiss_index_type` | `str` | `"flat"` | Index type: `"flat"` (exact) or `"hnsw"` (approximate) |

---

## `LocalEmbedder`

Singleton embedding generator.

```python
from longcache.core.embedder import LocalEmbedder

embedder = LocalEmbedder(model_name="BAAI/bge-small-en-v1.5")
vector = embedder.embed("What is LongCache?")  # Returns np.ndarray shape (384,)
```

* `embed(text: str) -> np.ndarray`: Embeds and L2-normalizes the input string.
* `dimension`: Returns vector dimension (384 for BGE-small).

---

## Middleware & Framework Wrappers

### `LongCacheMiddleware`
```python
from longcache.middleware import LongCacheMiddleware

app.add_middleware(LongCacheMiddleware, cache=cache, cached_routes={"/v1/chat/completions"})
```

### `CachedLLM` & `CachedChain`
```python
from longcache.integrations.langchain import CachedLLM, CachedChain

cached_llm = CachedLLM(llm, cache=cache)
cached_chain = CachedChain(chain, cache=cache)
```

### `CachedQueryEngine`
```python
from longcache.integrations.llamaindex import CachedQueryEngine

cached_engine = CachedQueryEngine(query_engine, cache=cache)
```

### `CachedOpenAI`
```python
from longcache.integrations.openai import CachedOpenAI

cached_client = CachedOpenAI(openai_client, cache=cache)
```
