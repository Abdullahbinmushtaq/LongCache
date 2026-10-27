# FastAPI & Starlette Middleware

LongCache includes `LongCacheMiddleware` for ASGI frameworks (FastAPI, Starlette, Litestar). It intercepts HTTP requests to LLM endpoints and serves cached responses with custom HTTP cache headers.

## Installation

```bash
uv pip install "longcache[server]"
# or
pip install "longcache[server]"
```

---

## Usage in FastAPI

```python
from fastapi import FastAPI, Request
from longcache import LongCache, CacheConfig
from longcache.middleware import LongCacheMiddleware

app = FastAPI(title="LLM Gateway API")

# Initialize cache (SQLite or Redis)
cache = LongCache(CacheConfig(
    similarity_threshold=0.92,
    namespace="api_gateway"
))

# Add middleware
app.add_middleware(
    LongCacheMiddleware,
    cache=cache,
    cached_routes={"/v1/chat/completions", "/api/chat"},  # Target endpoints
)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    # Your upstream LLM call here (e.g. LiteLLM, vLLM, OpenAI, Ollama)
    return {
        "id": "chatcmpl-123",
        "choices": [
            {"message": {"role": "assistant", "content": "Hello from upstream model!"}}
        ]
    }
```

---

## Response Headers

When `LongCacheMiddleware` serves a cached response, it injects observability headers into the HTTP response:

| Header | Value | Meaning |
|---|---|---|
| `X-LongCache` | `HIT` | Present when a cached response is served; absent on a provider miss or bypass |

## Supported requests in v0.1.1

Eligible requests contain a model and a single text user message, with optional system/developer instructions and temperature. Multi-turn, streaming, tool, multimodal, and additional request options bypass caching. Requests with Authorization, Cookie, or X-API-Key headers also bypass caching because automatic user/tenant scoping is not implemented.

The upstream response is cached only for successful JSON chat completions containing text and no refusal or tool calls. The supported context fields are included in the cache identity, and entries from the older middleware namespace are not reused.
