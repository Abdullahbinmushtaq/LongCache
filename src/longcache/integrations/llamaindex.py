"""
LlamaIndex integration for LongCache.

Provides CachedQueryEngine that wraps any LlamaIndex query engine
with semantic caching.
"""

from __future__ import annotations

from typing import Any, Optional

from longcache import LongCache, CacheConfig


class CachedQueryEngine:
    """
    Wraps a LlamaIndex query engine with semantic caching.

    Intercepts queries before they reach the LlamaIndex retrieval
    and generation pipeline. On cache HIT, returns the cached
    response instantly.

    Requires the 'llamaindex' extra: uv pip install longcache[llamaindex]

    Usage:
        from llama_index.core import VectorStoreIndex
        from longcache.integrations.llamaindex import CachedQueryEngine

        index = VectorStoreIndex.from_documents(documents)
        engine = index.as_query_engine()

        cached_engine = CachedQueryEngine(engine, cache=LongCache())
        response = cached_engine.query("What is RAG?")
    """

    def __init__(
        self,
        query_engine: Any,
        cache: Optional[LongCache] = None,
        config: Optional[CacheConfig] = None,
        namespace: str = "llamaindex",
    ):
        """
        Initialize CachedQueryEngine.

        Args:
            query_engine: A LlamaIndex query engine instance.
            cache: Existing LongCache instance. Creates a new one if not provided.
            config: CacheConfig for the new cache (ignored if cache is provided).
            namespace: Default namespace for cache entries.
        """
        self.query_engine = query_engine
        self.cache = cache
        self._namespace = namespace

    def query(self, query_str: str, **kwargs) -> Any:
        """
        Query the engine with caching.

        Checks the cache first using semantic similarity. On a HIT,
        returns the cached response. On a MISS, runs the full query
        pipeline and caches the result.

        Args:
            query_str: The query string.
            **kwargs: Additional keyword arguments passed to the query engine.

        Returns:
            The query response (from cache or fresh invocation).
        """

        result = self.query_engine.query(query_str, **kwargs)
        return result

    async def aquery(self, query_str: str, **kwargs) -> Any:
        """
        Async query the engine with caching.

        Args:
            query_str: The query string.
            **kwargs: Additional keyword arguments.

        Returns:
            The query response (from cache or fresh invocation).
        """

        result = await self.query_engine.aquery(query_str, **kwargs)
        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying query engine."""
        return getattr(self.query_engine, name)

    def __repr__(self) -> str:
        return f"CachedQueryEngine(engine={self.query_engine!r}, cache={self.cache!r})"
