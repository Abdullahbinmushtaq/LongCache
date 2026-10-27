"""
LongCache — Main orchestrator class.

Provides the primary API for semantic similarity caching:
- get/set for manual cache control
- @cached decorator for transparent caching
- invalidate/clear for cache management
- stats for monitoring hit rates and estimated savings
"""

from __future__ import annotations

import time
import asyncio
import inspect
import logging
import hashlib
import json
import threading as _threading
from typing import Any, Callable, Optional, AsyncGenerator, Generator

_log = logging.getLogger(__name__)

from longcache.config import CacheConfig
from longcache.core.embedder import LocalEmbedder
from longcache.core.index import SimilarityIndex
from longcache.core.entry import CacheEntry
from longcache.backends.sqlite import SQLiteCacheBackend
from longcache.backends.memory import InMemoryCacheBackend


class LongCache:
    """
    Semantic similarity cache for LLM/RAG pipelines.

    Intercepts queries before they reach expensive LLM calls,
    embeds the query locally (sub-5ms with MiniLM), and checks
    cosine similarity against a rolling FAISS/SQLite cache.

    On HIT (similarity > threshold): returns cached response instantly.
    On MISS: runs the full chain, caches the result, returns to user.

    Usage:
        # Simple decorator usage
        cache = LongCache()

        @cache.cached
        def my_chain(query: str) -> str:
            return expensive_llm_call(query)

        # Manual get/set usage
        cached = cache.get("What is the refund policy?")
        if cached is None:
            result = expensive_llm_call("What is the refund policy?")
            cache.set("What is the refund policy?", result)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize LongCache with optional configuration.

        Args:
            config: CacheConfig instance. Uses sensible defaults if not provided.
        """
        self.config = config or CacheConfig()
        self.embedder = LocalEmbedder(self.config.embedder_model)
        self.backend = self._init_backend()
        self.index = SimilarityIndex(self.backend, self.config)
        self._stats = {
            "hits": 0,
            "misses": 0,
            "total_saved_ms": 0.0,
        }
        self._stats_lock = _threading.Lock()

        # Load persisted stats from backend if available
        if hasattr(self.backend, "load_stats"):
            persisted = self.backend.load_stats()
            with self._stats_lock:
                self._stats.update(persisted)

        # Rebuild FAISS index from existing backend entries
        self._rebuild_index_from_backend()

        self._sweep_interval = 300  # seconds between TTL sweeps (5 minutes)
        self._start_sweep_thread()

    def __del__(self):
        """Signal the sweeper thread to stop on garbage collection."""
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

    def _start_sweep_thread(self) -> None:
        """Start background thread that periodically deletes expired entries."""
        if self.config.ttl_seconds == 0:
            return  # No TTL configured, no need to sweep

        def sweeper():
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=self._sweep_interval)
                if self._stop_event.is_set():
                    break
                try:
                    if hasattr(self.backend, "sweep_expired"):
                        deleted = self.backend.sweep_expired()
                        if deleted > 0:
                            _log.debug("TTL sweep: deleted %d expired entries", deleted)
                            self._rebuild_index_from_backend()
                except Exception as e:
                    _log.warning("TTL sweep error: %s", e)

        self._stop_event = _threading.Event()
        t = _threading.Thread(target=sweeper, name="longcache-sweeper", daemon=True)
        t.start()

    def _init_backend(self):
        """
        Initialize the cache storage backend based on configuration.

        Returns:
            An instance of the appropriate AbstractCacheBackend subclass.
        """
        if self.config.backend == "sqlite":
            return SQLiteCacheBackend(
                db_path=self.config.db_path,
                max_entries=self.config.max_entries,
            )
        elif self.config.backend == "redis":
            try:
                from longcache.backends.redis import RedisCacheBackend
            except ImportError:
                raise ImportError(
                    "Redis backend requires the 'redis' extra. "
                    "Install with: uv pip install longcache[redis]"
                )
            return RedisCacheBackend(
                redis_url=self.config.redis_url,
                max_entries=self.config.max_entries,
            )
        elif self.config.backend == "async_redis":
            try:
                from longcache.backends.async_redis import AsyncRedisCacheBackend
            except ImportError:
                raise ImportError(
                    "Async Redis backend requires the 'redis' extra. "
                    "Install with: uv pip install longcache[redis]"
                )
            return AsyncRedisCacheBackend(
                redis_url=self.config.redis_url,
                max_entries=self.config.max_entries,
            )
        elif self.config.backend == "memory":
            return InMemoryCacheBackend(
                max_entries=self.config.max_entries,
            )
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def _rebuild_index_from_backend(self) -> None:
        """
        Rebuild the FAISS index from entries already stored in the backend.

        Called on startup, after sweeping expired entries, or after evictions
        so that cached entries are cleanly re-indexed without duplicate vector accumulation.
        """
        try:
            entries = self.backend.list_entries(self.config.namespace)
            self.index.rebuild(entries or [])
        except Exception as e:
            _log.warning("Index rebuild failed: %s — starting with empty index.", e)

    def _build_context_hash(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Build a deterministic hash from model, system_prompt, and temperature.
        Returns empty string if all context parameters are default/empty.
        """
        m = model if model is not None else self.config.model
        sp = system_prompt if system_prompt is not None else self.config.system_prompt
        t = temperature if temperature is not None else self.config.temperature

        if not m and not sp and t < 0.0:
            return ""

        context: dict[str, Any] = {
            "model": m,
            "system_prompt": sp,
        }
        if t >= 0.0:
            context["temperature"] = t

        serialized = json.dumps(context, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def _effective_namespace(
        self,
        namespace: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Returns the namespace scoped to tenant and execution context (model/prompt/temp).
        """
        base = namespace or self.config.namespace
        ctx = self._build_context_hash(model, system_prompt, temperature)
        if not ctx:
            return base
        return f"{base}:{ctx}"

    def get(
        self,
        query: str,
        namespace: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        track_miss: bool = True,
    ) -> Optional[Any]:
        """
        Check the cache for a semantically similar query.

        Embeds the query locally, searches the FAISS index, and returns
        the cached response if a sufficiently similar entry exists.

        Args:
            query: The input query string.
            namespace: Override the default namespace for this lookup.
            model: Optional model identifier override.
            system_prompt: Optional system prompt override.
            temperature: Optional temperature override.
            track_miss: Whether to record a miss if no match is found.

        Returns:
            The cached response if a HIT is found, None on MISS.
        """
        ns = self._effective_namespace(namespace, model, system_prompt, temperature)
        vector = self.embedder.embed(query)

        entry = self.index.find_similar(
            vector,
            threshold=self.config.similarity_threshold,
            namespace=ns,
        )

        if entry is not None and not entry.is_expired():
            if self.config.track_stats:
                with self._stats_lock:
                    self._stats["hits"] += 1
                if hasattr(self.backend, "increment_hits"):
                    self.backend.increment_hits()
            _log.debug("Cache HIT  ns=%s query=%r", ns, query[:60])
            return entry.response

        if track_miss and self.config.track_stats:
            with self._stats_lock:
                self._stats["misses"] += 1
            if hasattr(self.backend, "increment_misses"):
                self.backend.increment_misses(0.0)

        _log.debug("Cache MISS ns=%s query=%r", ns, query[:60])
        return None

    async def aget(
        self,
        query: str,
        namespace: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        track_miss: bool = True,
    ) -> Optional[Any]:
        """Async version of get(). Use in async contexts to avoid blocking."""
        loop = asyncio.get_event_loop()
        ns = self._effective_namespace(namespace, model, system_prompt, temperature)
        vector = await loop.run_in_executor(None, self.embedder.embed, query)
        entry = self.index.find_similar(
            vector, threshold=self.config.similarity_threshold, namespace=ns
        )
        if entry is not None and not entry.is_expired():
            if self.config.track_stats:
                with self._stats_lock:
                    self._stats["hits"] += 1
                if hasattr(self.backend, "increment_hits"):
                    if asyncio.iscoroutinefunction(getattr(self.backend, "increment_hits")):
                        await self.backend.increment_hits()
                    else:
                        self.backend.increment_hits()
            _log.debug("Cache HIT  ns=%s query=%r", ns, query[:60])
            return entry.response

        if track_miss and self.config.track_stats:
            with self._stats_lock:
                self._stats["misses"] += 1
            if hasattr(self.backend, "increment_misses"):
                if asyncio.iscoroutinefunction(getattr(self.backend, "increment_misses")):
                    await self.backend.increment_misses(0.0)
                else:
                    self.backend.increment_misses(0.0)

        _log.debug("Cache MISS ns=%s query=%r", ns, query[:60])
        return None

    def set(
        self,
        query: str,
        response: Any,
        namespace: Optional[str] = None,
        ttl: Optional[int] = None,
        tags: Optional[list[str]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Store a query-response pair in the cache.

        Embeds the query locally, creates a CacheEntry, stores it in
        the backend, and adds the vector to the FAISS index.

        Args:
            query: The query string to cache.
            response: The LLM response to cache. Must be serializable by the backend.
            namespace: Override the default namespace for this entry.
            ttl: Override the default TTL (in seconds) for this entry.
            tags: Optional list of tags to associate with this entry.
            model: Optional model identifier override.
            system_prompt: Optional system prompt override.
            temperature: Optional temperature override.
        """
        ns = self._effective_namespace(namespace, model, system_prompt, temperature)
        effective_ttl = ttl if ttl is not None else self.config.ttl_seconds

        vector = self.embedder.embed(query)

        now = time.time()
        expires_at = (now + effective_ttl) if effective_ttl > 0 else None

        entry = CacheEntry(
            namespace=ns,
            query=query,
            vector=vector,
            response=response,
            created_at=now,
            expires_at=expires_at,
            hit_count=0,
            model=model if model is not None else self.config.model,
            system_prompt=system_prompt if system_prompt is not None else self.config.system_prompt,
            temperature=temperature if temperature is not None else self.config.temperature,
            tags=list(tags or []),
        )

        # Store in backend (assigns entry.id)
        entry_id = self.backend.store(entry)
        entry.id = entry_id

        # Add to FAISS index
        self.index.add(vector, entry_id)

        # Enforce max_entries eviction
        self._evict_if_needed(ns)

        _log.debug("Cache SET  ns=%s ttl=%s query=%r", ns, effective_ttl, query[:60])

    async def aset(
        self,
        query: str,
        response: Any,
        namespace: Optional[str] = None,
        ttl: Optional[int] = None,
        tags: Optional[list[str]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """Async version of set(). Use in async contexts to avoid blocking."""
        loop = asyncio.get_event_loop()
        ns = self._effective_namespace(namespace, model, system_prompt, temperature)
        effective_ttl = ttl if ttl is not None else self.config.ttl_seconds
        vector = await loop.run_in_executor(None, self.embedder.embed, query)
        now = time.time()
        expires_at = (now + effective_ttl) if effective_ttl > 0 else None
        entry = CacheEntry(
            namespace=ns,
            query=query,
            vector=vector,
            response=response,
            created_at=now,
            expires_at=expires_at,
            hit_count=0,
            model=model if model is not None else self.config.model,
            system_prompt=system_prompt if system_prompt is not None else self.config.system_prompt,
            temperature=temperature if temperature is not None else self.config.temperature,
            tags=list(tags or []),
        )
        # Handle async_redis case where store is a coroutine
        if asyncio.iscoroutinefunction(self.backend.store):
            entry_id = await self.backend.store(entry)
        else:
            entry_id = self.backend.store(entry)

        entry.id = entry_id
        await self.index.async_add(vector, entry_id)
        self._evict_if_needed(ns)

        _log.debug("Cache ASET ns=%s ttl=%s query=%r", ns, effective_ttl, query[:60])

    def cached(
        self,
        func: Optional[Callable] = None,
        *,
        namespace: Optional[str] = None,
        ttl: Optional[int] = None,
        tags: Optional[list[str]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Callable:
        """
        Decorator for caching any function that takes a query string as first argument.

        Can be used with or without arguments:
            @cache.cached
            def my_chain(query): ...

            @cache.cached(namespace="faq", ttl=7200, model="gpt-4o", tags=["billing"])
            def my_chain(query): ...
        """

        def decorator(fn: Callable) -> Callable:
            if asyncio.iscoroutinefunction(fn):

                async def async_wrapper(query: str, *args, **kwargs):
                    cached_result = await self.aget(
                        query,
                        namespace=namespace,
                        model=model,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        track_miss=False,
                    )
                    if cached_result is not None:
                        return cached_result

                    t0 = time.time()
                    result = await fn(query, *args, **kwargs)
                    elapsed_ms = (time.time() - t0) * 1000

                    await self.aset(
                        query,
                        result,
                        namespace=namespace,
                        ttl=ttl,
                        tags=tags,
                        model=model,
                        system_prompt=system_prompt,
                        temperature=temperature,
                    )

                    if self.config.track_stats:
                        with self._stats_lock:
                            self._stats["misses"] += 1
                            self._stats["total_saved_ms"] += elapsed_ms
                        if hasattr(self.backend, "increment_misses"):
                            if asyncio.iscoroutinefunction(
                                getattr(self.backend, "increment_misses")
                            ):
                                await self.backend.increment_misses(elapsed_ms)
                            else:
                                self.backend.increment_misses(elapsed_ms)
                    return result

                async_wrapper.__name__ = fn.__name__
                async_wrapper.__doc__ = fn.__doc__
                async_wrapper.__wrapped__ = fn
                return async_wrapper
            else:

                def wrapper(query: str, *args, **kwargs):
                    cached_result = self.get(
                        query,
                        namespace=namespace,
                        model=model,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        track_miss=False,
                    )
                    if cached_result is not None:
                        return cached_result

                    t0 = time.time()
                    result = fn(query, *args, **kwargs)
                    elapsed_ms = (time.time() - t0) * 1000

                    self.set(
                        query,
                        result,
                        namespace=namespace,
                        ttl=ttl,
                        tags=tags,
                        model=model,
                        system_prompt=system_prompt,
                        temperature=temperature,
                    )

                    if self.config.track_stats:
                        with self._stats_lock:
                            self._stats["misses"] += 1
                            self._stats["total_saved_ms"] += elapsed_ms
                        if hasattr(self.backend, "increment_misses"):
                            self.backend.increment_misses(elapsed_ms)

                    return result

                wrapper.__name__ = fn.__name__
                wrapper.__doc__ = fn.__doc__
                wrapper.__wrapped__ = fn
                return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    def cached_stream(
        self,
        func: Optional[Callable] = None,
        *,
        namespace: Optional[str] = None,
        ttl: Optional[int] = None,
        tags: Optional[list[str]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Callable:
        """
        Decorator for caching async or sync streaming generator functions.

        On MISS: yields stream chunks live while collecting full text, then saves in cache.
        On HIT:  re-yields cached text in chunks to emulate the streaming interface.
        """

        def decorator(fn: Callable) -> Callable:
            if inspect.isasyncgenfunction(fn):

                async def async_stream_wrapper(query: str, *args, **kwargs):
                    cached_result = await self.aget(
                        query,
                        namespace=namespace,
                        model=model,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        track_miss=False,
                    )
                    if cached_result is not None:
                        if isinstance(cached_result, str):
                            chunk_size = max(1, len(cached_result) // 20 or 1)
                            for i in range(0, len(cached_result), chunk_size):
                                yield cached_result[i : i + chunk_size]
                        else:
                            yield cached_result
                        return

                    t0 = time.time()
                    collected_tokens = []
                    async for token in fn(query, *args, **kwargs):
                        collected_tokens.append(token)
                        yield token

                    elapsed_ms = (time.time() - t0) * 1000
                    full_response = (
                        "".join(collected_tokens)
                        if all(isinstance(c, str) for c in collected_tokens)
                        else collected_tokens
                    )
                    await self.aset(
                        query,
                        full_response,
                        namespace=namespace,
                        ttl=ttl,
                        tags=tags,
                        model=model,
                        system_prompt=system_prompt,
                        temperature=temperature,
                    )

                    if self.config.track_stats:
                        with self._stats_lock:
                            self._stats["misses"] += 1
                            self._stats["total_saved_ms"] += elapsed_ms
                        if hasattr(self.backend, "increment_misses"):
                            if asyncio.iscoroutinefunction(
                                getattr(self.backend, "increment_misses")
                            ):
                                await self.backend.increment_misses(elapsed_ms)
                            else:
                                self.backend.increment_misses(elapsed_ms)

                async_stream_wrapper.__name__ = fn.__name__
                async_stream_wrapper.__doc__ = fn.__doc__
                async_stream_wrapper.__wrapped__ = fn
                return async_stream_wrapper
            else:

                def sync_stream_wrapper(query: str, *args, **kwargs):
                    cached_result = self.get(
                        query,
                        namespace=namespace,
                        model=model,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        track_miss=False,
                    )
                    if cached_result is not None:
                        if isinstance(cached_result, str):
                            chunk_size = max(1, len(cached_result) // 20 or 1)
                            for i in range(0, len(cached_result), chunk_size):
                                yield cached_result[i : i + chunk_size]
                        else:
                            yield cached_result
                        return

                    t0 = time.time()
                    collected_tokens = []
                    for token in fn(query, *args, **kwargs):
                        collected_tokens.append(token)
                        yield token

                    elapsed_ms = (time.time() - t0) * 1000
                    full_response = (
                        "".join(collected_tokens)
                        if all(isinstance(c, str) for c in collected_tokens)
                        else collected_tokens
                    )
                    self.set(
                        query,
                        full_response,
                        namespace=namespace,
                        ttl=ttl,
                        tags=tags,
                        model=model,
                        system_prompt=system_prompt,
                        temperature=temperature,
                    )

                    if self.config.track_stats:
                        with self._stats_lock:
                            self._stats["misses"] += 1
                            self._stats["total_saved_ms"] += elapsed_ms
                        if hasattr(self.backend, "increment_misses"):
                            self.backend.increment_misses(elapsed_ms)

                sync_stream_wrapper.__name__ = fn.__name__
                sync_stream_wrapper.__doc__ = fn.__doc__
                sync_stream_wrapper.__wrapped__ = fn
                return sync_stream_wrapper

        if func is not None:
            return decorator(func)
        return decorator

    def invalidate(
        self,
        namespace: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> int:
        """
        Invalidate all entries in a namespace (optionally scoped to model/prompt/temperature).

        Removes all cache entries and their vectors from both the
        backend storage and the FAISS index.

        Args:
            namespace: The namespace to clear. Defaults to the configured namespace.
            model: Optional model scope.
            system_prompt: Optional system prompt scope.
            temperature: Optional temperature scope.

        Returns:
            The number of entries deleted.
        """
        ns = self._effective_namespace(namespace, model, system_prompt, temperature)
        count = self.backend.delete_namespace(ns)

        # Rebuild index after namespace deletion
        self._rebuild_index_from_backend()

        return count

    def invalidate_by_tag(
        self,
        tag: str,
        namespace: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> int:
        """
        Invalidate all cache entries carrying the given tag.

        Args:
            tag: The tag label to match.
            namespace: Scope the invalidation. Defaults to config namespace.
            model: Optional model context.
            system_prompt: Optional system prompt context.
            temperature: Optional temperature context.

        Returns:
            The number of entries deleted.
        """
        ns = self._effective_namespace(namespace, model, system_prompt, temperature)
        count = self.backend.delete_by_tag(tag, ns)
        if count > 0:
            self._rebuild_index_from_backend()
        return count

    def clear(self) -> int:
        """
        Clear the entire cache across all namespaces.

        Removes all entries from backend storage and resets the FAISS index.

        Returns:
            The total number of entries deleted.
        """
        count = self.backend.clear()

        # Reset FAISS index
        self.index.rebuild([])

        # Reset stats
        if self.config.track_stats:
            with self._stats_lock:
                self._stats = {"hits": 0, "misses": 0, "total_saved_ms": 0.0}

        return count

    def record_hit(self) -> None:
        """Record a cache hit in a thread-safe manner."""
        if self.config.track_stats:
            with self._stats_lock:
                self._stats["hits"] += 1
            if hasattr(self.backend, "increment_hits"):
                self.backend.increment_hits()

    def record_miss(self, elapsed_ms: float = 0.0) -> None:
        """Record a cache miss and elapsed generation time in a thread-safe manner."""
        if self.config.track_stats:
            with self._stats_lock:
                self._stats["misses"] += 1
                self._stats["total_saved_ms"] += elapsed_ms
            if hasattr(self.backend, "increment_misses"):
                self.backend.increment_misses(elapsed_ms)

    def stats(self) -> dict:
        """
        Get cache statistics including hit rate and estimated savings.

        Returns:
            Dictionary with:
                - hits: Number of cache hits
                - misses: Number of cache misses
                - hit_rate: Hit rate as a float between 0 and 1
                - total_saved_ms: Total milliseconds saved across all misses/hits
                - estimated_tokens_saved: Rough estimate of tokens saved by caching
                - estimated_cost_saved_usd: Estimated USD saved (at GPT-4 pricing)
                - total_entries: Total number of entries in the cache
        """
        with self._stats_lock:
            hits = self._stats.get("hits", 0)
            misses = self._stats.get("misses", 0)
            total_saved_ms = self._stats.get("total_saved_ms", 0.0)

        total = hits + misses
        avg_tokens_per_hit = 1500  # rough estimate for typical RAG responses
        gpt4_cost_per_1k_tokens = 0.03  # approximate blended rate

        estimated_tokens_saved = hits * avg_tokens_per_hit
        estimated_cost_saved = (estimated_tokens_saved / 1000) * gpt4_cost_per_1k_tokens

        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total > 0 else 0.0,
            "total_saved_ms": total_saved_ms,
            "estimated_tokens_saved": estimated_tokens_saved,
            "estimated_cost_saved_usd": round(estimated_cost_saved, 4),
            "total_entries": self.backend.count(self.config.namespace),
        }

    def inspect(
        self,
        query: str,
        namespace: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Inspect the nearest cached entry for a query without returning the cached response.

        Useful for debugging and understanding what would be cached for a given query.

        Args:
            query: The query to inspect.
            namespace: Override namespace for the lookup.
            model: Optional model context.
            system_prompt: Optional system prompt context.
            temperature: Optional temperature context.

        Returns:
            Dictionary with entry metadata and similarity info, or None if no match.
        """
        ns = self._effective_namespace(namespace, model, system_prompt, temperature)
        vector = self.embedder.embed(query)

        entry = self.index.find_similar(
            vector,
            threshold=0.0,  # Find the nearest regardless of threshold
            namespace=ns,
        )

        if entry is None:
            return None

        return {
            "id": entry.id,
            "query": entry.query,
            "namespace": entry.namespace,
            "hit_count": entry.hit_count,
            "created_at": entry.created_at,
            "expires_at": entry.expires_at,
            "is_expired": entry.is_expired(),
            "model": entry.model,
            "system_prompt": entry.system_prompt,
            "temperature": entry.temperature,
            "tags": list(entry.tags or []),
        }

    def _evict_if_needed(self, namespace: str) -> None:
        """
        Evict oldest entries if the namespace exceeds max_entries.

        Uses LRU-style eviction: entries with the oldest created_at
        timestamps are removed first.

        Args:
            namespace: The namespace to check for eviction.
        """
        count = self.backend.count(namespace)
        if count <= self.config.max_entries:
            return

        # Evict the excess entries (oldest first)
        excess = count - self.config.max_entries
        evicted = self.backend.evict_oldest(namespace, excess)

        # Rebuild index to keep it consistent
        if evicted > 0:
            self._rebuild_index_from_backend()

    def __repr__(self) -> str:
        return (
            f"LongCache(backend={self.config.backend}, "
            f"namespace={self.config.namespace!r}, "
            f"threshold={self.config.similarity_threshold})"
        )
