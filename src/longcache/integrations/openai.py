"""OpenAI chat-completion integration for supported single-turn text requests."""

from __future__ import annotations

import inspect
import json
from typing import Any, Optional

from longcache import CacheConfig, LongCache


def _request_context(kwargs: dict) -> Optional[tuple[str, str, str, float]]:
    """Return a complete cache identity for the supported request subset."""
    if set(kwargs) - {"messages", "model", "temperature", "stream"}:
        return None
    if kwargs.get("stream", False) is not False:
        return None
    model = kwargs.get("model")
    messages = kwargs.get("messages")
    if not isinstance(model, str) or not model or not isinstance(messages, list):
        return None
    if not messages or len(messages) > 3:
        return None
    instructions = []
    for message in messages[:-1]:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            return None
        if message["role"] not in {"system", "developer"}:
            return None
        if not isinstance(message["content"], str):
            return None
        instructions.append(message)
    last = messages[-1]
    if not isinstance(last, dict) or set(last) != {"role", "content"}:
        return None
    if last["role"] != "user" or not isinstance(last["content"], str) or not last["content"]:
        return None
    temperature = kwargs.get("temperature", -1.0)
    if temperature is None:
        temperature = -1.0
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        return None
    return last["content"], model, json.dumps(instructions, sort_keys=True), float(temperature)


def _serializable_response(result: Any) -> Optional[dict]:
    """Keep cached response types stable across persistent backends."""
    if isinstance(result, dict):
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        for choice in choices:
            if isinstance(choice, dict):
                if choice.get("finish_reason") not in (None, "stop"):
                    return None
                message = choice.get("message", {})
                if isinstance(message, dict) and (
                    message.get("refusal") or message.get("tool_calls")
                ):
                    return None
        payload = {"kind": "dict", "data": result}
    elif type(result).__name__ == "ChatCompletion" and type(result).__module__.startswith("openai"):
        if not result.choices or any(
            choice.finish_reason != "stop" or choice.message.refusal or choice.message.tool_calls
            for choice in result.choices
        ):
            return None
        payload = {"kind": "ChatCompletion", "data": result.model_dump(mode="json")}
    else:
        return None
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError):
        return None
    return payload


def _restore_response(cached: Any) -> Any:
    if not isinstance(cached, dict) or "kind" not in cached:
        return None
    if cached["kind"] == "dict":
        return cached.get("data")
    if cached["kind"] == "ChatCompletion":
        try:
            from openai.types.chat import ChatCompletion

            return ChatCompletion.model_validate(cached["data"])
        except (ImportError, TypeError, ValueError):
            return None
    return None


class CachedOpenAI:
    """Cache supported single-turn text calls on sync or async OpenAI clients."""

    def __init__(
        self,
        client: Any,
        cache: Optional[LongCache] = None,
        config: Optional[CacheConfig] = None,
        namespace: str = "openai",
    ):
        self._client = client
        self.cache = cache or LongCache(config or CacheConfig(namespace=namespace))
        # Do not reuse entries produced by the older, under-scoped wrapper.
        self._namespace = f"{namespace}:single-turn-v1"
        self.chat = _ChatProxy(self)

    def _extract_query(self, messages: list[dict]) -> str:
        """Compatibility helper; cache eligibility is checked separately."""
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                return message.get("content", str(message))
        return str(messages)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def __repr__(self) -> str:
        return f"CachedOpenAI(client={self._client!r}, cache={self.cache!r})"


class _ChatProxy:
    """Expose the documented `chat.completions.create` and legacy aliases."""

    def __init__(self, parent: CachedOpenAI):
        self._parent = parent
        create = parent._client.chat.completions.create
        wrapper = (
            AsyncCachedChatCompletions
            if inspect.iscoroutinefunction(create)
            else CachedChatCompletions
        )
        self.completions = wrapper(parent)

    def create(self, **kwargs) -> Any:
        return self.completions.create(**kwargs)

    async def acreate(self, **kwargs) -> Any:
        return await self.completions.acreate(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._parent._client.chat, name)


class CachedChatCompletions:
    """Cached wrapper for synchronous chat completions."""

    def __init__(self, parent: CachedOpenAI):
        self._parent = parent

    def create(self, **kwargs) -> Any:
        context = _request_context(kwargs)
        client = self._parent._client.chat.completions
        if context is None:
            return client.create(**kwargs)
        query, model, instructions, temperature = context
        cached = self._parent.cache.get(
            query,
            namespace=self._parent._namespace,
            model=model,
            system_prompt=instructions,
            temperature=temperature,
        )
        if cached is not None:
            restored = _restore_response(cached)
            if restored is not None:
                return restored
        result = client.create(**kwargs)
        payload = _serializable_response(result)
        if payload is not None:
            self._parent.cache.set(
                query,
                payload,
                namespace=self._parent._namespace,
                model=model,
                system_prompt=instructions,
                temperature=temperature,
            )
        return result

    async def acreate(self, **kwargs) -> Any:
        context = _request_context(kwargs)
        client = self._parent._client.chat.completions
        call = getattr(client, "acreate", None)
        if call is None or not inspect.iscoroutinefunction(call):
            if inspect.iscoroutinefunction(client.create):
                call = client.create
            else:
                raise TypeError("An async OpenAI client is required for acreate")
        if context is None:
            return await call(**kwargs)
        query, model, instructions, temperature = context
        cached = await self._parent.cache.aget(
            query,
            namespace=self._parent._namespace,
            model=model,
            system_prompt=instructions,
            temperature=temperature,
        )
        if cached is not None:
            restored = _restore_response(cached)
            if restored is not None:
                return restored
        result = await call(**kwargs)
        payload = _serializable_response(result)
        if payload is not None:
            await self._parent.cache.aset(
                query,
                payload,
                namespace=self._parent._namespace,
                model=model,
                system_prompt=instructions,
                temperature=temperature,
            )
        return result

    def __repr__(self) -> str:
        return f"CachedChatCompletions(parent={self._parent!r})"


class AsyncCachedChatCompletions(CachedChatCompletions):
    """Expose the SDK's awaited `create` call for AsyncOpenAI."""

    async def create(self, **kwargs) -> Any:
        return await self.acreate(**kwargs)
