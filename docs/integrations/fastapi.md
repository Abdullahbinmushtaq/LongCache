# ASGI / FastAPI

`LongCacheMiddleware` supports bounded non-streaming chat-completion JSON requests. Supply `context` with provider/deployment/operation/version configuration and a synchronous `scope_resolver(scope)` reading an authenticated principal from trusted ASGI state.

Authentication middleware must run before scope resolution. Raw tenant/user headers, Authorization values, cookies, and API keys are not identity sources. Authenticated requests bypass without a trusted resolver. Resolver failure bypasses without shared-tenant fallback.

The capture limit defaults to 1 MiB. Oversized, malformed/duplicate-key, unsupported-shape, and streaming requests pass through without cache storage. Complete status-200 JSON responses are admitted only when safe response semantics permit reuse; cookie-bearing, `Vary`, private/no-store, incomplete, or oversized responses are not stored.

`X-LongCache`, `X-LongCache-Reason`, and an applicable same-scope similarity header explain lookup. Headers sent at response start do not claim later admission results. Upstream diagnostic copies are replaced; unsupported responses keep their body/status/framing.
