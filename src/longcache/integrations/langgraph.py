"""
LangGraph integration for LongCache.

Provides CachedState wrapper for semantic caching of agent/graph states.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from longcache import LongCache, CacheConfig


class CachedState:
    """
    Wraps LangGraph state evaluation with semantic caching.

    Can be used as a node in a LangGraph graph to check the semantic cache
    before executing expensive LLM nodes.

    Usage:
        from longcache.integrations.langgraph import CachedState
        from longcache import LongCache

        cache = LongCache()

        def cache_node(state: dict):
            # Check cache based on the latest message/input
            return CachedState.check(state, cache, query_key="messages")
    """

    @staticmethod
    def check(
        state: dict,
        cache: LongCache,
        query_key: str = "messages",
        namespace: str = "langgraph",
    ) -> dict:
        """
        Check the cache for a given state.

        Args:
            state: The LangGraph state dictionary.
            cache: LongCache instance.
            query_key: The key in the state to use as the cache query.
                If it's a list of messages, the last human message is used.
            namespace: The cache namespace.

        Returns:
            A dictionary with cache results. If a hit is found, it can be
            merged back into the state to skip LLM execution.
        """
        query_input = state.get(query_key)

        # Extract string query
        query = ""
        if isinstance(query_input, str):
            query = query_input
        elif isinstance(query_input, list):
            for msg in reversed(query_input):
                if hasattr(msg, "content") and getattr(msg, "type", "") == "human":
                    query = msg.content
                    break
                elif hasattr(msg, "content"):
                    query = msg.content
            if not query and query_input:
                query = str(query_input[-1])
        elif isinstance(query_input, dict):
            query = query_input.get("query", str(query_input))
        else:
            query = str(query_input)

        if not query:
            return {"__cache_hit": False}

        cached = cache.get(query, namespace=namespace)
        if cached is not None:
            return {"__cache_hit": True, "cached_response": cached}

        return {"__cache_hit": False, "cached_query": query}

    @staticmethod
    async def acheck(
        state: dict,
        cache: LongCache,
        query_key: str = "messages",
        namespace: str = "langgraph",
    ) -> dict:
        """Async version of check()."""
        query_input = state.get(query_key)

        # Extract string query
        query = ""
        if isinstance(query_input, str):
            query = query_input
        elif isinstance(query_input, list):
            for msg in reversed(query_input):
                if hasattr(msg, "content") and getattr(msg, "type", "") == "human":
                    query = msg.content
                    break
                elif hasattr(msg, "content"):
                    query = msg.content
            if not query and query_input:
                query = str(query_input[-1])
        elif isinstance(query_input, dict):
            query = query_input.get("query", str(query_input))
        else:
            query = str(query_input)

        if not query:
            return {"__cache_hit": False}

        cached = await cache.aget(query, namespace=namespace)
        if cached is not None:
            return {"__cache_hit": True, "cached_response": cached}

        return {"__cache_hit": False, "cached_query": query}

    @staticmethod
    def save(
        state: dict,
        response: Any,
        cache: LongCache,
        namespace: str = "langgraph",
    ) -> None:
        """
        Save a new response to the cache.

        Args:
            state: The state dictionary (must contain 'cached_query' from check).
            response: The response to cache.
            cache: LongCache instance.
            namespace: The cache namespace.
        """
        query = state.get("cached_query")
        if query:
            cache.set(query, response, namespace=namespace)

    @staticmethod
    async def asave(
        state: dict,
        response: Any,
        cache: LongCache,
        namespace: str = "langgraph",
    ) -> None:
        """Async version of save()."""
        query = state.get("cached_query")
        if query:
            await cache.aset(query, response, namespace=namespace)
