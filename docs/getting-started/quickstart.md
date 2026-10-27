# Quick Start

Get up and running with LongCache in under 60 seconds.

## 1. Using the `@cache.cached` Decorator

The easiest way to cache LLM calls is via the `@cache.cached` decorator. LongCache automatically handles function arguments, return values, and detects sync vs async functions.

```python
from longcache import LongCache

# Default configuration uses local SQLite persistence at .longcache/cache.db
cache = LongCache()

@cache.cached
def ask_assistant(question: str) -> str:
    print("--> Calling expensive LLM API...")
    # Your real LLM or LangChain chain call here
    return f"Response to: {question}"

# 1. First execution — Cache MISS
print(ask_assistant("How can I update my billing email address?"))
# Output:
# --> Calling expensive LLM API...
# Response to: How can I update my billing email address?

# 2. Second execution with semantically identical question — Cache HIT!
print(ask_assistant("Where do I change my invoice email?"))
# Output (instant response, no LLM call printed!):
# Response to: How can I update my billing email address?
```

## 2. Async Functions

Async functions are detected automatically:

```python
import asyncio
from longcache import LongCache

cache = LongCache()

@cache.cached(namespace="support", ttl=3600)
async def ask_async(question: str) -> str:
    await asyncio.sleep(1.0)  # Simulate network API latency
    return f"Processed: {question}"

async def main():
    res1 = await ask_async("What is the return policy?")
    res2 = await ask_async("Can I get a refund on an item?")  # Cache hit!
    print(res2)

asyncio.run(main())
```

## 3. Explicit `get()` and `set()` API

If you prefer full programmatic control over when values are saved and retrieved:

```python
from longcache import LongCache

cache = LongCache()

query = "What are your business hours?"

# Look up in cache
cached_result = cache.get(query)

if cached_result is not None:
    print(f"Cache HIT: {cached_result}")
else:
    print("Cache MISS — calling model")
    result = "We are open Monday to Friday from 9 AM to 6 PM EST."
    
    # Store in cache with optional TTL
    cache.set(query, result, ttl=7200)
```

## 4. Configuring Cache Behavior

You can customize the similarity threshold, backend, TTL, and namespace using `CacheConfig`:

```python
from longcache import LongCache, CacheConfig

config = CacheConfig(
    similarity_threshold=0.92,   # Cosine similarity cutoff (0.0 to 1.0)
    ttl_seconds=86400,            # 24 hour TTL
    max_entries=50000,           # Max entries before LRU eviction
    namespace="customer_support", # Isolated namespace
    backend="sqlite",            # "sqlite", "redis", or "memory"
    db_path="data/support_cache.db"
)

cache = LongCache(config)
```
