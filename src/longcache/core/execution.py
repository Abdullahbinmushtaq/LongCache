"""One scoped execution protocol for manual use and provider/decorator adapters."""

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import ContextVar
import inspect
import json
import time

from .errors import (
    CacheOperationError,
    SingleflightLeaderError,
    SingleflightReentrancyError,
    SingleflightTimeoutError,
    SingleflightInvalidatedError,
)
from .identity import digest, canonical
from .policy import admit, decode_response, verify
from .singleflight import Completion, Singleflight
from .errors import CacheClosedError


class Execution:
    def __init__(self, cache):
        self.cache = cache
        self.registry = Singleflight(cache.config)
        self._owned = ContextVar(f"longcache_flight_owner_{id(self)}", default=())

    def shutdown(self):
        self.registry.abort(shutdown=True)

    @staticmethod
    def options(ttl, tags):
        if ttl is not None and (type(ttl) is not int or ttl < 0):
            raise ValueError("TTL must be a nonnegative integer")
        if tags is not None and (
            not isinstance(tags, (tuple, list)) or not all(type(t) is str for t in tags)
        ):
            raise ValueError("Tags must be strings")
        return tuple(sorted(set(tags or ())))

    async def afinish(self, *args):
        return await self.cache._lifecycle.call(self.finish, *args)

    async def aclaim(self, *args):
        future, token = self.cache._lifecycle.submit(self.claim, *args)
        bridge = asyncio.wrap_future(future)
        bridge.add_done_callback(lambda result: None if result.cancelled() else result.exception())
        try:
            return await asyncio.shield(bridge)
        except asyncio.CancelledError:
            token.set()

            def abandon(result):
                if not result.cancelled() and result.exception() is None:
                    flight, leader, _ = result.result()
                    if leader:
                        self.registry.complete(flight, Completion("error"))
                    else:
                        self.registry.detach(flight)

            future.add_done_callback(abandon)
            raise

    def prepare(self, query, context, namespace, *, _cancel=None):
        return self.cache._prepare(query, context, namespace, {})

    def claim(self, query, prepared, namespace, ttl, tags, *, _cancel=None):
        cache = self.cache
        ctx = prepared[0]
        exact = ctx.exact_key(
            query,
            namespace if namespace is not None else cache.config.namespace,
            cache.config.scope_policy,
            cache._fingerprint,
            cache.config.policy_version,
        )
        write_policy = digest(
            {
                "ttl": cache.config.ttl_seconds if ttl is None else ttl,
                "tags": sorted(set(tags or ())),
            }
        )
        with cache._mutation_lock:
            epoch = cache._epoch
            key = exact, write_policy, epoch
            if key in self._owned.get():
                raise SingleflightReentrancyError("Execution cannot wait on its own key")
            flight, leader = self.registry.claim(key)
        cache._observability.count("singleflight", role="leader" if leader else "follower")
        return flight, leader, epoch

    def lookup(self, query, context, namespace, *, _cancel=None):
        return self.cache.lookup(
            query, namespace, context=context, _record=False, _internal=True, _cancel=_cancel
        )

    def finish(
        self, prepared, decision, response, source, mode, role, entry_point, epoch, *, _cancel=None
    ):
        cache = self.cache
        with cache._mutation_lock:
            cache._ensure_open()
            if _cancel is not None and _cancel.is_set():
                raise CacheClosedError("Operation was cancelled before delivery")
            response_epoch = decision._epoch if source == "cache" else epoch
            if source in ("cache", "coalesced") and response_epoch != cache._epoch:
                raise SingleflightInvalidatedError(
                    "Shared response was invalidated before delivery"
                )
            if prepared is not None:
                if source == "cache":
                    try:
                        cache._dependency(
                            "accounting", cache.backend.update_hit_count, decision.entry_id
                        )
                    except CacheOperationError:
                        pass
                cache._account(prepared[1], "hits" if source == "cache" else "misses")
        outcome = "hit" if source == "cache" else "miss" if prepared else "bypass"
        cache._observability.count("requests", operation="execution", outcome=outcome, mode=mode)
        cache._observability.count(
            "decisions", operation="execution", outcome=outcome, source=source
        )
        cache._observability.event(
            operation="execution",
            outcome=outcome,
            source=source,
            mode=mode,
            role=role,
            entry_point=entry_point,
        )
        return response

    def envelope(self, result, context, expires_at):
        payload, admission = admit(result, context, self.cache._admit)
        if admission.action != "accept":
            return Completion("non_shareable")
        encoded = canonical(payload).encode("utf-8")
        if len(encoded) > self.cache.config.max_capture_bytes:
            return Completion("non_shareable")
        return Completion("value", encoded, expires_at=expires_at)

    def replay(self, completion, query, context, *, _cancel=None):
        if completion.kind == "invalidated":
            raise type(completion.error)(str(completion.error))
        if completion.kind == "error":
            raise SingleflightLeaderError(
                "Shared execution failed or its leader was cancelled"
            ) from completion.error
        if completion.kind != "value":
            return None, False
        if completion.expires_at is not None and time.time() >= completion.expires_at:
            return None, False
        response = decode_response(json.loads(completion.payload), context.codec)
        payload, admission = admit(response, context, self.cache._admit)
        if admission.action != "accept":
            return None, False
        verification = verify(query, query, payload, 1.0, context, self.cache._verifier)
        self.cache._observability.count(
            "verification",
            outcome="hit" if verification.action == "accept" else "rejected",
            reason=verification.reason,
        )
        return response, verification.action == "accept"

    def publish(self, flight, result, context, epoch, expires_at=None, *, _cancel=None):
        completion = self.envelope(result, context, expires_at)
        with self.cache._mutation_lock:
            if epoch != self.cache._epoch:
                completion = Completion(
                    "invalidated",
                    error=SingleflightInvalidatedError("Shared execution was invalidated"),
                )
            self.registry.complete(flight, completion)

    def origin(self, compute):
        self.cache._observability.count("origin_executions")
        started = time.monotonic()
        try:
            result = compute()
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("Use aget_or_compute with an asynchronous callback")
            return result
        finally:
            self.cache._observability.duration("origin", time.monotonic() - started)

    async def aorigin(self, compute):
        self.cache._observability.count("origin_executions")
        started = time.monotonic()
        try:
            pending = compute()
            if not inspect.isawaitable(pending):
                raise TypeError("aget_or_compute requires an awaitable-producing callback")
            return await pending
        finally:
            self.cache._observability.duration("origin", time.monotonic() - started, "async")

    def run(
        self,
        query,
        compute,
        *,
        context=None,
        namespace=None,
        ttl=None,
        tags=None,
        _entry_point="manual",
    ):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Use aget_or_compute from an event loop")
        if not callable(compute) or inspect.iscoroutinefunction(compute):
            raise TypeError("get_or_compute requires a synchronous callable")
        tags = self.options(ttl, tags)
        return self._run(
            query,
            compute,
            context=context,
            namespace=namespace,
            ttl=ttl,
            tags=tags,
            _entry_point=_entry_point,
        )

    def _run(self, query, compute, *, context, namespace, ttl, tags, _entry_point):
        cache = self.cache
        with cache._lifecycle.operation():
            try:
                prepared, _ = self.prepare(query, context, namespace)
            except CacheOperationError:
                if cache.config.cache_error_policy != "bypass":
                    raise
                prepared = None
            if prepared is None:
                result = self.origin(compute)
                return self.finish(
                    None, None, result, "origin", "sync", "none", _entry_point, cache._epoch
                )
            ctx = prepared[0]
            decision = self.lookup(query, ctx, namespace)
            if decision.outcome == "HIT":
                return self.finish(
                    prepared,
                    decision,
                    decision.response,
                    "cache",
                    "sync",
                    "none",
                    _entry_point,
                    cache._epoch,
                )
            if ttl is not None and (type(ttl) is not int or ttl < 0):
                raise ValueError("TTL must be a nonnegative integer")
            if tags is not None and (
                not isinstance(tags, (tuple, list)) or not all(type(t) is str for t in tags)
            ):
                raise ValueError("Tags must be strings")
            flight, leader, epoch = (
                self.claim(query, prepared, namespace, ttl, tags)
                if cache.config.singleflight_enabled
                else (None, True, cache._epoch)
            )
            if flight is not None and not leader:
                try:
                    decision = self.lookup(query, ctx, namespace)
                    if decision.outcome == "HIT":
                        return self.finish(
                            prepared,
                            decision,
                            decision.response,
                            "cache",
                            "sync",
                            "follower",
                            _entry_point,
                            epoch,
                        )
                    started = time.monotonic()
                    try:
                        completion = flight.future.result(
                            timeout=cache.config.singleflight_wait_timeout_seconds
                        )
                    except FutureTimeout:
                        raise SingleflightTimeoutError(
                            "Timed out waiting for shared execution"
                        ) from None
                    finally:
                        cache._observability.duration("wait", time.monotonic() - started)
                    if completion.kind in ("error", "invalidated"):
                        self.replay(completion, query, ctx)
                    decision = (
                        self.lookup(query, ctx, namespace) if completion.kind == "value" else None
                    )
                    if decision is not None and decision.outcome == "HIT":
                        return self.finish(
                            prepared,
                            decision,
                            decision.response,
                            "cache",
                            "sync",
                            "follower",
                            _entry_point,
                            epoch,
                        )
                    response, accepted = self.replay(completion, query, ctx)
                    if accepted:
                        return self.finish(
                            prepared,
                            None,
                            response,
                            "coalesced",
                            "sync",
                            "follower",
                            _entry_point,
                            epoch,
                        )
                finally:
                    self.registry.detach(flight)
                result = self.origin(compute)
                cache.set(
                    query,
                    result,
                    namespace,
                    context=ctx,
                    ttl=ttl,
                    tags=tags,
                    _expected_epoch=epoch,
                    _internal=True,
                )
                return self.finish(
                    prepared, None, result, "origin", "sync", "follower", _entry_point, epoch
                )
            owner = self._owned.set((*self._owned.get(), flight.key)) if flight else None
            try:
                decision = self.lookup(query, ctx, namespace)
                if decision.outcome == "HIT":
                    if flight:
                        self.publish(flight, decision.response, ctx, epoch, decision._expires_at)
                    return self.finish(
                        prepared,
                        decision,
                        decision.response,
                        "cache",
                        "sync",
                        "leader",
                        _entry_point,
                        epoch,
                    )
                result = self.origin(compute)
                effective_ttl = cache.config.ttl_seconds if ttl is None else ttl
                expiry = time.time() + effective_ttl if effective_ttl else None
                stored = cache.set(
                    query,
                    result,
                    namespace,
                    context=ctx,
                    ttl=ttl,
                    tags=tags,
                    _expected_epoch=epoch,
                    _internal=True,
                )
                if flight:
                    if stored.outcome == "REJECTED" and stored.reason not in (
                        "cache_unavailable",
                        "invalidated",
                    ):
                        self.registry.complete(flight, Completion("non_shareable"))
                    else:
                        self.publish(flight, result, ctx, epoch, expiry)
                return self.finish(
                    prepared, None, result, "origin", "sync", "leader", _entry_point, epoch
                )
            except BaseException as exc:
                if flight:
                    self.registry.complete(flight, Completion("error", error=exc))
                raise
            finally:
                if flight:
                    self._owned.reset(owner)

    async def arun(
        self,
        query,
        compute,
        *,
        context=None,
        namespace=None,
        ttl=None,
        tags=None,
        _entry_point="manual",
    ):
        if not callable(compute):
            raise TypeError("aget_or_compute requires an awaitable-producing callable")
        tags = self.options(ttl, tags)
        cache = self.cache
        with cache._lifecycle.operation():
            try:
                prepared, _ = await cache._lifecycle.call(self.prepare, query, context, namespace)
            except CacheOperationError:
                if cache.config.cache_error_policy != "bypass":
                    raise
                prepared = None
            if prepared is None:
                result = await self.aorigin(compute)
                return await self.afinish(
                    None, None, result, "origin", "async", "none", _entry_point, cache._epoch
                )
            ctx = prepared[0]
            decision = await cache._lifecycle.call(self.lookup, query, ctx, namespace)
            if decision.outcome == "HIT":
                return await self.afinish(
                    prepared,
                    decision,
                    decision.response,
                    "cache",
                    "async",
                    "none",
                    _entry_point,
                    cache._epoch,
                )
            if ttl is not None and (type(ttl) is not int or ttl < 0):
                raise ValueError("TTL must be a nonnegative integer")
            if tags is not None and (
                not isinstance(tags, (tuple, list)) or not all(type(t) is str for t in tags)
            ):
                raise ValueError("Tags must be strings")
            flight, leader, epoch = (
                await self.aclaim(query, prepared, namespace, ttl, tags)
                if cache.config.singleflight_enabled
                else (None, True, cache._epoch)
            )
            if flight is not None and not leader:
                try:
                    decision = await cache._lifecycle.call(self.lookup, query, ctx, namespace)
                    if decision.outcome == "HIT":
                        return await self.afinish(
                            prepared,
                            decision,
                            decision.response,
                            "cache",
                            "async",
                            "follower",
                            _entry_point,
                            epoch,
                        )
                    started = time.monotonic()
                    try:
                        bridge = asyncio.wrap_future(flight.future)
                        completion = await asyncio.wait_for(
                            asyncio.shield(bridge), cache.config.singleflight_wait_timeout_seconds
                        )
                    except asyncio.TimeoutError:
                        raise SingleflightTimeoutError(
                            "Timed out waiting for shared execution"
                        ) from None
                    finally:
                        cache._observability.duration("wait", time.monotonic() - started, "async")
                    if completion.kind in ("error", "invalidated"):
                        self.replay(completion, query, ctx)
                    decision = (
                        await cache._lifecycle.call(self.lookup, query, ctx, namespace)
                        if completion.kind == "value"
                        else None
                    )
                    if decision is not None and decision.outcome == "HIT":
                        return await self.afinish(
                            prepared,
                            decision,
                            decision.response,
                            "cache",
                            "async",
                            "follower",
                            _entry_point,
                            epoch,
                        )
                    response, accepted = await cache._lifecycle.call(
                        self.replay, completion, query, ctx
                    )
                    if accepted:
                        return await self.afinish(
                            prepared,
                            None,
                            response,
                            "coalesced",
                            "async",
                            "follower",
                            _entry_point,
                            epoch,
                        )
                finally:
                    self.registry.detach(flight)
                result = await self.aorigin(compute)
                await cache._async(
                    cache.set,
                    query,
                    result,
                    namespace,
                    context=ctx,
                    ttl=ttl,
                    tags=tags,
                    _expected_epoch=epoch,
                    _internal=True,
                )
                return await self.afinish(
                    prepared, None, result, "origin", "async", "follower", _entry_point, epoch
                )
            owner = self._owned.set((*self._owned.get(), flight.key)) if flight else None
            try:
                decision = await cache._lifecycle.call(self.lookup, query, ctx, namespace)
                if decision.outcome == "HIT":
                    if flight:
                        await cache._lifecycle.call(
                            self.publish,
                            flight,
                            decision.response,
                            ctx,
                            epoch,
                            decision._expires_at,
                        )
                    return await self.afinish(
                        prepared,
                        decision,
                        decision.response,
                        "cache",
                        "async",
                        "leader",
                        _entry_point,
                        epoch,
                    )
                result = await self.aorigin(compute)
                effective_ttl = cache.config.ttl_seconds if ttl is None else ttl
                expiry = time.time() + effective_ttl if effective_ttl else None
                stored = await cache._async(
                    cache.set,
                    query,
                    result,
                    namespace,
                    context=ctx,
                    ttl=ttl,
                    tags=tags,
                    _expected_epoch=epoch,
                    _internal=True,
                )
                if flight:
                    if stored.outcome == "REJECTED" and stored.reason not in (
                        "cache_unavailable",
                        "invalidated",
                    ):
                        self.registry.complete(flight, Completion("non_shareable"))
                    else:
                        await cache._lifecycle.call(
                            self.publish, flight, result, ctx, epoch, expiry
                        )
                return await self.afinish(
                    prepared, None, result, "origin", "async", "leader", _entry_point, epoch
                )
            except BaseException as exc:
                if flight:
                    self.registry.complete(flight, Completion("error", error=exc))
                raise
            finally:
                if flight:
                    self._owned.reset(owner)
