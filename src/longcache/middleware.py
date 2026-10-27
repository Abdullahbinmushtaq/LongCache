"""
FastAPI/Starlette ASGI middleware for LongCache.

Intercepts HTTP requests to chat completion endpoints, checks the
semantic cache before forwarding to the upstream LLM API, and
auto-caches responses. Drop-in middleware for any Starlette app.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional, Set

from longcache import LongCache, CacheConfig


class LongCacheMiddleware:
    """
    ASGI middleware that adds semantic caching to LLM API endpoints.

    Intercepts POST requests to configured routes (default: /v1/chat/completions),
    extracts the user query, checks the semantic cache, and either returns
    the cached response or forwards the request and caches the result.

    Usage:
        from fastapi import FastAPI
        from longcache.middleware import LongCacheMiddleware

        app = FastAPI()
        app.add_middleware(LongCacheMiddleware, cache=LongCache())

    Or with Starlette:
        from starlette.applications import Starlette
        from longcache.middleware import LongCacheMiddleware

        app = Starlette(middleware=[
            (LongCacheMiddleware, {"cache": LongCache()})
        ])
    """

    def __init__(
        self,
        app: Callable,
        cache: Optional[LongCache] = None,
        config: Optional[CacheConfig] = None,
        namespace: str = "http",
        cached_routes: Optional[Set[str]] = None,
        exclude_namespaces: Optional[Set[str]] = None,
    ):
        """
        Initialize the LongCache ASGI middleware.

        Args:
            app: The ASGI application to wrap.
            cache: Existing LongCache instance. Creates a new one if not provided.
            config: CacheConfig for the new cache (ignored if cache is provided).
            namespace: Default namespace for cache entries.
            cached_routes: Set of URL paths that should be cached.
                Defaults to {"/v1/chat/completions", "/chat/completions"}.
            exclude_namespaces: Set of namespaces to exclude from caching.
        """
        self.app = app
        self.cache = cache or LongCache(config or CacheConfig(namespace=namespace))
        self._namespace = namespace
        self.cached_routes = cached_routes or {
            "/v1/chat/completions",
            "/chat/completions",
        }
        self.exclude_namespaces = exclude_namespaces or set()

    async def __call__(self, scope: dict, receive: Callable, send: Callable):
        """
        ASGI middleware entry point.

        Intercepts request/response flow for configured routes.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")

        # Only intercept POST requests to cached routes
        if method != "POST" or path not in self.cached_routes:
            await self.app(scope, receive, send)
            return

        # Read the request body
        body = await self._read_body(receive)

        try:
            request_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            # Not JSON, pass through without caching
            await self._send_passthrough(scope, body, send)
            return

        # Extract the user query from messages
        query = self._extract_query(request_data)
        if not query:
            await self._send_passthrough(scope, body, send)
            return

        # Check cache
        cached = self.cache.get(query, namespace=self._namespace)
        if cached is not None:
            await self._send_cached_response(cached, send)
            return

        # Cache MISS — forward request and capture response
        response_data = await self._forward_and_capture(scope, body, send)

        # Cache the response
        if response_data is not None:
            self.cache.set(query, response_data, namespace=self._namespace)

    def _extract_query(self, request_data: dict) -> str:
        """
        Extract the user query from a chat completion request.

        Args:
            request_data: Parsed JSON request body.

        Returns:
            The extracted query string, or empty string if not found.
        """
        messages = request_data.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    async def _read_body(self, receive: Callable) -> bytes:
        """
        Read the full request body from the ASGI receive callable.

        Args:
            receive: The ASGI receive callable.

        Returns:
            The complete request body as bytes.
        """
        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break
        return body

    async def _send_passthrough(self, scope: dict, body: bytes, send: Callable):
        """
        Forward the original request without caching.

        Reconstructs the ASGI receive callable with the captured body
        and forwards to the upstream app.
        """

        async def receive_with_body():
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }

        await self.app(scope, receive_with_body, send)

    async def _send_cached_response(self, cached_response: Any, send: Callable):
        """
        Send a cached response as an HTTP response.

        Args:
            cached_response: The cached response data.
            send: The ASGI send callable.
        """
        response_body = json.dumps(cached_response, default=str).encode("utf-8")

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"x-longcache", b"HIT"],
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": response_body,
            }
        )

    async def _forward_and_capture(self, scope: dict, body: bytes, send: Callable) -> Optional[Any]:
        """
        Forward the request to the upstream app and capture the response.

        Args:
            scope: The ASGI scope.
            body: The request body.
            send: The original ASGI send callable.

        Returns:
            The parsed response data if successful, None otherwise.
        """
        response_parts = []
        response_started = False
        response_status = 200

        async def capturing_send(message: dict):
            nonlocal response_started, response_status

            if message["type"] == "http.response.start":
                response_started = True
                response_status = message.get("status", 200)
                # Only capture headers, pass through to real send
                await send(message)
            elif message["type"] == "http.response.body":
                response_parts.append(message.get("body", b""))
                await send(message)

        async def receive_with_body():
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }

        try:
            await self.app(scope, receive_with_body, capturing_send)
        except Exception:
            return None

        if response_status != 200:
            return None

        # Parse the captured response
        full_response = b"".join(response_parts)
        try:
            return json.loads(full_response)
        except (json.JSONDecodeError, TypeError):
            return None

    def __repr__(self) -> str:
        return f"LongCacheMiddleware(routes={self.cached_routes}, cache={self.cache!r})"


class StreamingCacheAdapter:
    """
    Adapter for caching streaming LLM responses.

    Collects streaming chunks into a complete response for caching,
    and replays cached responses token-by-token to maintain the
    streaming UX for end users.

    Usage:
        from longcache.middleware import StreamingCacheAdapter

        def my_stream_func(query):
            for chunk in llm.stream(query):
                yield chunk

        adapter = StreamingCacheAdapter(cache=LongCache(), stream_func=my_stream_func)
        for token in adapter.stream("What is RAG?"):
            print(token, end="")
    """

    def __init__(
        self,
        cache: LongCache,
        stream_func: Callable,
        namespace: str = "streaming",
        replay_delay: float = 0.01,
    ):
        """
        Initialize the streaming cache adapter.

        Args:
            cache: LongCache instance for storage.
            stream_func: Callable that yields response chunks for a query.
            namespace: Cache namespace for streaming entries.
            replay_delay: Delay in seconds between replayed tokens
                to simulate streaming pacing.
        """
        self.cache = cache
        self.stream_func = stream_func
        self._namespace = namespace
        self._replay_delay = replay_delay

    def stream(self, query: str):
        """
        Stream a response with caching.

        On cache HIT, replays the cached response token-by-token.
        On cache MISS, streams from the source, collects chunks,
        and caches the assembled response.

        Args:
            query: The query string.

        Yields:
            Response tokens/chunks.
        """
        cached = self.cache.get(query, namespace=self._namespace)
        if cached is not None:
            # Replay cached response with realistic pacing
            import time as _time
            import re

            cached_str = str(cached)
            tokens = re.findall(r"\S+\s*", cached_str)
            for token in tokens:
                yield token
                if self._replay_delay > 0:
                    _time.sleep(self._replay_delay)
            return

        # Stream from source and collect
        collected = []
        for chunk in self.stream_func(query):
            collected.append(str(chunk))
            yield chunk

        # Cache the assembled response
        assembled = "".join(collected)
        self.cache.set(query, assembled, namespace=self._namespace)

    async def astream(self, query: str):
        """
        Async stream a response with caching.

        Args:
            query: The query string.

        Yields:
            Response tokens/chunks.
        """
        cached = self.cache.get(query, namespace=self._namespace)
        if cached is not None:
            import asyncio
            import re

            cached_str = str(cached)
            tokens = re.findall(r"\S+\s*", cached_str)
            for token in tokens:
                yield token
                if self._replay_delay > 0:
                    await asyncio.sleep(self._replay_delay)
            return

        collected = []
        async for chunk in self.stream_func(query):
            collected.append(str(chunk))
            yield chunk

        assembled = "".join(collected)
        self.cache.set(query, assembled, namespace=self._namespace)

    def __repr__(self) -> str:
        return f"StreamingCacheAdapter(cache={self.cache!r})"
