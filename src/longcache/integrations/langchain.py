"""
LangChain integration for LongCache.

Provides CachedLLM and CachedChain wrappers that add semantic caching
to any LangChain LLM or LCEL chain with zero code changes.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional, Union

from longcache import LongCache, CacheConfig


class CachedLLM:
    """
    Wraps any LangChain LLM/ChatModel with semantic caching.

    Drop-in replacement that preserves the full LLM interface.
    Cache is checked before the LLM is called, and new responses
    are cached automatically.

    Usage:
        from langchain_openai import ChatOpenAI
        from longcache.integrations.langchain import CachedLLM

        llm = CachedLLM(ChatOpenAI(model="gpt-4o"), cache=LongCache())
        response = llm.invoke("What is the refund policy?")  # same API
        # Second call with similar query returns cached response instantly
        response2 = llm.invoke("How do I get a refund?")  # cache HIT
    """

    def __init__(
        self,
        llm: Any,
        cache: Optional[LongCache] = None,
        config: Optional[CacheConfig] = None,
        namespace: str = "langchain",
    ):
        """
        Initialize CachedLLM.

        Args:
            llm: Any LangChain BaseLLM or BaseChatModel instance.
            cache: Existing LongCache instance. Creates a new one if not provided.
            config: CacheConfig for the new cache (ignored if cache is provided).
            namespace: Default namespace for cache entries.
        """
        self.llm = llm
        self.cache = cache or LongCache(config or CacheConfig(namespace=namespace))
        self._namespace = namespace

    def _extract_query(self, input: Any) -> str:
        """
        Extract a query string from various LangChain input formats.

        Supports:
            - str: used directly
            - list of BaseMessage: extracts last HumanMessage content
            - dict with 'query', 'question', 'input', or 'messages' key

        Args:
            input: LangChain input in any supported format.

        Returns:
            Extracted query string.
        """
        if isinstance(input, str):
            return input

        if isinstance(input, list):
            # Try to extract from message list
            for msg in reversed(input):
                if hasattr(msg, "content") and hasattr(msg, "type"):
                    if msg.type == "human":
                        return msg.content
                # Fallback: use last message with content
                if hasattr(msg, "content"):
                    return msg.content
            # Last resort: stringify
            return str(input[-1]) if input else ""

        if isinstance(input, dict):
            for key in ("query", "question", "input", "messages", "prompt"):
                if key in input:
                    value = input[key]
                    if isinstance(value, str):
                        return value
                    if isinstance(value, list):
                        return self._extract_query(value)
                    return str(value)

        return str(input)

    def invoke(self, input: Any, *args, **kwargs) -> Any:
        """
        Invoke the LLM with caching.

        Checks the cache first. On a HIT, returns the cached response.
        On a MISS, calls the LLM and caches the result.

        Args:
            input: The LLM input (string, messages list, or dict).
            *args: Additional positional arguments passed to the LLM.
            **kwargs: Additional keyword arguments passed to the LLM.

        Returns:
            The LLM response (from cache or fresh invocation).
        """
        query_key = self._extract_query(input)
        cached = self.cache.get(query_key, namespace=self._namespace)
        if cached is not None:
            return cached

        result = self.llm.invoke(input, *args, **kwargs)
        self.cache.set(query_key, result, namespace=self._namespace)
        return result

    async def ainvoke(self, input: Any, *args, **kwargs) -> Any:
        """
        Async invoke the LLM with caching.

        Same as invoke() but uses the async LLM interface.

        Args:
            input: The LLM input.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The LLM response (from cache or fresh invocation).
        """
        query_key = self._extract_query(input)
        cached = await self.cache.aget(query_key, namespace=self._namespace)
        if cached is not None:
            return cached

        result = await self.llm.ainvoke(input, *args, **kwargs)
        await self.cache.aset(query_key, result, namespace=self._namespace)
        return result

    def stream(self, input: Any, *args, **kwargs):
        """
        Stream the LLM response with caching.

        On a cache HIT, replays the cached response token-by-token.
        On a MISS, streams from the LLM, collects the response, and caches it.

        Args:
            input: The LLM input.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Yields:
            Response chunks (from cache replay or live LLM stream).
        """
        query_key = self._extract_query(input)
        cached = self.cache.get(query_key, namespace=self._namespace)

        if cached is not None:
            # Replay cached response preserving original whitespace
            cached_str = str(cached)
            tokens = re.findall(r"\S+\s*", cached_str)
            for token in tokens:
                yield token
            return

        # Stream from LLM and collect
        collected_chunks = []
        for chunk in self.llm.stream(input, *args, **kwargs):
            collected_chunks.append(chunk)
            yield chunk

        # Cache the assembled response
        self.cache.set(query_key, collected_chunks, namespace=self._namespace)

    async def astream(self, input: Any, *args, **kwargs):
        """
        Async stream the LLM response with caching.

        Args:
            input: The LLM input.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Yields:
            Response chunks.
        """
        query_key = self._extract_query(input)
        cached = await self.cache.aget(query_key, namespace=self._namespace)

        if cached is not None:
            cached_str = str(cached)
            tokens = re.findall(r"\S+\s*", cached_str)
            for token in tokens:
                yield token
            return

        collected_chunks = []
        async for chunk in self.llm.astream(input, *args, **kwargs):
            collected_chunks.append(chunk)
            yield chunk

        await self.cache.aset(query_key, collected_chunks, namespace=self._namespace)

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying LLM."""
        return getattr(self.llm, name)

    def __repr__(self) -> str:
        return f"CachedLLM(llm={self.llm!r}, cache={self.cache!r})"


class CachedChain:
    """
    Wraps any LCEL Runnable chain with semantic caching.

    Cache key is the final user query string extracted from the chain input.
    Works with any LCEL chain: prompt | llm | parser, RetrievalQA, etc.

    Usage:
        from langchain_openai import ChatOpenAI
        from longcache.integrations.langchain import CachedChain

        chain = prompt | ChatOpenAI() | StrOutputParser()
        cached_chain = CachedChain(chain, cache=LongCache())
        result = cached_chain.invoke({"question": "What is RAG?"})
    """

    def __init__(
        self,
        chain: Any,
        cache: Optional[LongCache] = None,
        config: Optional[CacheConfig] = None,
        query_key: str = "question",
        namespace: str = "langchain_chain",
    ):
        """
        Initialize CachedChain.

        Args:
            chain: Any LCEL Runnable chain.
            cache: Existing LongCache instance. Creates a new one if not provided.
            config: CacheConfig for the new cache (ignored if cache is provided).
            query_key: Key to extract from input dict as the cache query.
            namespace: Default namespace for cache entries.
        """
        self.chain = chain
        self.cache = cache or LongCache(config or CacheConfig(namespace=namespace))
        self.query_key = query_key
        self._namespace = namespace

    def _extract_query(self, input: Any) -> str:
        """
        Extract the query string from chain input.

        Args:
            input: Chain input (dict or string).

        Returns:
            The extracted query string.
        """
        if isinstance(input, dict):
            return input.get(self.query_key, str(input))
        return str(input)

    def invoke(self, input: Any, *args, **kwargs) -> Any:
        """
        Invoke the chain with caching.

        Args:
            input: Chain input (typically a dict with the query key).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The chain output (from cache or fresh invocation).
        """
        query = self._extract_query(input)
        cached = self.cache.get(query, namespace=self._namespace)
        if cached is not None:
            return cached

        result = self.chain.invoke(input, *args, **kwargs)
        self.cache.set(query, result, namespace=self._namespace)
        return result

    async def ainvoke(self, input: Any, *args, **kwargs) -> Any:
        """
        Async invoke the chain with caching.

        Args:
            input: Chain input.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The chain output (from cache or fresh invocation).
        """
        query = self._extract_query(input)
        cached = await self.cache.aget(query, namespace=self._namespace)
        if cached is not None:
            return cached

        result = await self.chain.ainvoke(input, *args, **kwargs)
        await self.cache.aset(query, result, namespace=self._namespace)
        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying chain."""
        return getattr(self.chain, name)

    def __repr__(self) -> str:
        return f"CachedChain(chain={self.chain!r}, cache={self.cache!r})"
