"""
Basic LangChain + LongCache example.

Demonstrates how to add semantic caching to a LangChain LLM
with minimal code changes. Run this script to see cache hits
and misses in action.

Usage:
    uv run python examples/basic_langchain.py
"""

from longcache import LongCache, CacheConfig

# Initialize the cache with default settings (SQLite backend)
cache = LongCache(
    CacheConfig(
        similarity_threshold=0.90,  # 90% similarity = cache hit
        ttl_seconds=3600,  # Cache entries live for 1 hour
        max_entries=1000,  # Max 1000 entries per namespace
        namespace="langchain_demo",
    )
)


# ---------- Approach 1: Decorator ----------


@cache.cached
def expensive_llm_call(query: str) -> str:
    """
    Simulates an expensive LLM call.
    In production, replace this with your actual LangChain chain.
    """
    import time

    time.sleep(0.5)  # Simulate API latency
    return f"LLM response for: {query}"


# ---------- Approach 2: Manual get/set ----------


def ask_llm(query: str) -> str:
    """Manually check cache before calling the LLM."""
    cached = cache.get(query)
    if cached is not None:
        print(f"  [CACHE HIT] for: {query!r}")
        return cached

    print(f"  [CACHE MISS] for: {query!r}")
    import time

    time.sleep(0.5)
    result = f"LLM response for: {query}"
    cache.set(query, result)
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("LongCache + LangChain Demo")
    print("=" * 60)

    # --- Decorator approach ---
    print("\n1. Decorator approach:")
    print("   First call (MISS):")
    result1 = expensive_llm_call("What is your refund policy?")
    print(f"   Result: {result1}")

    print("\n   Second call with SAME query (HIT):")
    result2 = expensive_llm_call("What is your refund policy?")
    print(f"   Result: {result2}")

    print("\n   Third call with SIMILAR query (HIT if similarity > 90%):")
    result3 = expensive_llm_call("How do I get a refund?")
    print(f"   Result: {result3}")

    # --- Manual approach ---
    print("\n2. Manual get/set approach:")
    result4 = ask_llm("What are your business hours?")
    result5 = ask_llm("What are your business hours?")  # Should be HIT
    result6 = ask_llm("When are you open?")  # Might be HIT (similar)

    # --- Stats ---
    print("\n3. Cache statistics:")
    stats = cache.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\nDone!")
