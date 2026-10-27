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
    routes={"/v1/chat/completions", "/api/chat"},  # Target endpoints
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
| `X-LongCache-Status` | `HIT` or `MISS` | Whether the response was served from cache |
| `X-LongCache-Similarity` | `0.942` | The cosine similarity of the matched query |
| `X-LongCache-Latency-Saved-Ms` | `1850` | Estimated milliseconds saved vs upstream call |
