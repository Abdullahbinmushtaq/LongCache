"""
OpenAI integration for LongCache.

Provides CachedOpenAI that wraps the OpenAI Python client
with semantic caching, supporting both sync and async calls
as well as streaming.
"""

from __future__ import annotations

from typing import Any, Optional

from longcache import LongCache, CacheConfig


class CachedOpenAI:
    """
    Wraps the OpenAI Python client with semantic caching.

    Intercepts chat completion calls before they reach the OpenAI API.
    On cache HIT, returns the cached response instantly without an API call.

    Requires the 'openai' extra: uv pip install longcache[openai]

    Usage:
        from openai import OpenAI
        from longcache.integrations.openai import CachedOpenAI

        client = CachedOpenAI(OpenAI(api_key="..."), cache=LongCache())
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is RAG?"}]
        )
    """

    def __init__(
        self,
        client: Any,
        cache: Optional[LongCache] = None,
        config: Optional[CacheConfig] = None,
        namespace: str = "openai",
    ):
        """
        Initialize CachedOpenAI.

        Args:
            client: An OpenAI client instance (sync or async).
            cache: Existing LongCache instance. Creates a new one if not provided.
            config: CacheConfig for the new cache (ignored if cache is provided).
            namespace: Default namespace for cache entries.
        """
        self._client = client
        self.cache = cache or LongCache(config or CacheConfig(namespace=namespace))
        self._namespace = namespace

        # Wrap chat.completions with caching
        self.chat = CachedChatCompletions(self)

    def _extract_query(self, messages: list[dict]) -> str:
        """
        Extract the user query from a messages list.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            The content of the last user message.
        """
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", str(msg))
        # Fallback: stringify all messages
        return str(messages)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying client."""
        return getattr(self._client, name)

    def __repr__(self) -> str:
        return f"CachedOpenAI(client={self._client!r}, cache={self.cache!r})"


class CachedChatCompletions:
    """
    Cached wrapper for the chat.completions endpoint.

    Provides create() methods that check the cache before making API calls.
    """

    def __init__(self, parent: CachedOpenAI):
        """
        Initialize CachedChatCompletions.

        Args:
            parent: The parent CachedOpenAI instance.
        """
        self._parent = parent

    def create(self, **kwargs) -> Any:
        """
        Create a chat completion with caching.

        Extracts the user query from messages, checks the cache,
        and only calls the API on a cache MISS.

        Args:
            **kwargs: Arguments passed to openai.chat.completions.create().
                Must include 'messages' with at least one user message.

        Returns:
            The chat completion response (from cache or API).
        """
        messages = kwargs.get("messages", [])
        query = self._parent._extract_query(messages)

        cached = self._parent.cache.get(query, namespace=self._parent._namespace)
        if cached is not None:
            return cached

        result = self._parent._client.chat.completions.create(**kwargs)
        self._parent.cache.set(query, result, namespace=self._parent._namespace)
        return result

    async def acreate(self, **kwargs) -> Any:
        """
        Async create a chat completion with caching.

        Args:
            **kwargs: Arguments passed to the async OpenAI client.

        Returns:
            The chat completion response (from cache or API).
        """
        messages = kwargs.get("messages", [])
        query = self._parent._extract_query(messages)

        cached = self._parent.cache.get(query, namespace=self._parent._namespace)
        if cached is not None:
            return cached

        result = await self._parent._client.chat.completions.acreate(**kwargs)
        self._parent.cache.set(query, result, namespace=self._parent._namespace)
        return result

    def __repr__(self) -> str:
        return f"CachedChatCompletions(parent={self._parent!r})"
