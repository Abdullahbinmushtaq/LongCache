"""
FastAPI middleware + LongCache example.

Demonstrates how to add semantic caching as ASGI middleware
to a FastAPI application. Requests to /v1/chat/completions
are automatically cached by semantic similarity.

Usage:
    uv pip install longcache[server]
    uv run python examples/fastapi_middleware.py
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from longcache import LongCache, CacheConfig
from longcache.middleware import LongCacheMiddleware

# Initialize the cache
cache = LongCache(
    CacheConfig(
        similarity_threshold=0.90,
        ttl_seconds=3600,
        backend="sqlite",
        db_path=".longcache/fastapi_cache.db",
        namespace="http_api",
    )
)

# Create the FastAPI app
app = FastAPI(title="LongCache Demo API")

# Add the LongCache middleware
app.add_middleware(
    LongCacheMiddleware,
    cache=cache,
    namespace="http_api",
    cached_routes={
        "/v1/chat/completions",
        "/chat/completions",
    },
)


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """
    Simulated chat completions endpoint.

    In production, this would call the OpenAI API or your LLM provider.
    Here we simulate it with a simple response.
    """
    import asyncio

    await asyncio.sleep(0.3)  # Simulate API latency

    messages = request.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    return JSONResponse(
        {
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"This is a simulated response to: {user_message}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    )


@app.get("/cache/stats")
async def cache_stats():
    """View cache statistics."""
    return cache.stats()


@app.delete("/cache/clear")
async def cache_clear():
    """Clear the entire cache."""
    count = cache.clear()
    return {"deleted": count}


if __name__ == "__main__":
    import uvicorn

    print("Starting LongCache + FastAPI demo server...")
    print("Try: curl -X POST http://localhost:8000/v1/chat/completions")
    print("       -H 'Content-Type: application/json'")
    print(
        '       -d \'{"model": "demo", "messages": [{"role": "user", "content": "What is RAG?"}]}\''
    )
    print()
    print("View stats: curl http://localhost:8000/cache/stats")
    print("Clear cache: curl -X DELETE http://localhost:8000/cache/clear")
    uvicorn.run(app, host="0.0.0.0", port=8000)
