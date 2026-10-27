"""
Redis backend + LongCache single-process example.

Demonstrates Redis as shared response storage for one LongCache process.
Each process maintains its own FAISS index; writes by another process
are not discovered automatically in v0.1.1.

Prerequisites:
    1. Redis server running locally or accessible via URL
    2. Install longcache with Redis extra:
       uv pip install longcache[redis]

Usage:
    uv run python examples/redis_production.py
"""

from longcache import LongCache, CacheConfig

# Initialize with Redis backend
cache = LongCache(
    CacheConfig(
        similarity_threshold=0.92,
        ttl_seconds=7200,  # 2-hour TTL for production
        max_entries=100_000,  # Large cache for production
        backend="redis",
        redis_url="redis://localhost:6379/0",  # Adjust to your Redis URL
        namespace="production_faq",
    )
)


@cache.cached
def production_llm_call(query: str) -> str:
    """
    Production LLM call with Redis-backed caching.

    In production, this would be your actual RAG chain:
        - Embed the query
        - Retrieve documents from vector store
        - Generate answer with LLM
        - Return response

    With LongCache, semantically similar queries hit the cache
    and skip the entire pipeline.
    """
    import time

    # Simulate expensive RAG pipeline (retrieval + LLM)
    time.sleep(1.0)
    return f"[PRODUCTION] Answer for: {query}"


if __name__ == "__main__":
    print("=" * 60)
    print("LongCache + Redis Production Example")
    print("=" * 60)

    # Make sure Redis is accessible
    try:
        cache.backend._client.ping()
        print("Redis connection: OK")
    except Exception as e:
        print(f"Redis connection FAILED: {e}")
        print("Make sure Redis is running and accessible.")
        print("Start Redis locally with: docker run -p 6379:6379 redis:7")
        exit(1)

    print("\n1. First query (MISS — full pipeline runs):")
    result1 = production_llm_call("What is your refund policy?")
    print(f"   Result: {result1}")

    print("\n2. Same query again (HIT — instant from Redis cache):")
    result2 = production_llm_call("What is your refund policy?")
    print(f"   Result: {result2}")

    print("\n3. Semantically similar query (HIT — semantic cache!):")
    result3 = production_llm_call("How do I return a product?")
    print(f"   Result: {result3}")

    print("\n4. Cache stats:")
    stats = cache.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n5. Inspect nearest entry:")
    inspection = cache.inspect("refund")
    if inspection:
        for key, value in inspection.items():
            print(f"   {key}: {value}")
    else:
        print("   No matching entries found")

    print("\nDone! Cache entries persist in Redis across restarts.")
