# Caching Strategies

Depending on your application architecture, several caching patterns can be applied.

---

## 1. Gateway Caching (HTTP / REST Middleware)

Intercept incoming user requests at the HTTP routing layer before any application logic or agent orchestration executes.

- **Pros**: Zero backend compute, fastest response times (<5ms), cuts all token usage.
- **Cons**: Coarse-grained; caches the entire HTTP response payload.
- **Tool**: Use `longcache.middleware.FastAPICacheMiddleware`.

```python
from fastapi import FastAPI
from longcache.middleware import FastAPICacheMiddleware

app = FastAPI()
app.add_middleware(FastAPICacheMiddleware, similarity_threshold=0.92)
```

---

## 2. Agent Node Caching (LangGraph / State Machine)

In multi-agent systems, agents frequently execute auxiliary sub-tasks (e.g. summarization, query classification, intent extraction).

- **Pros**: Sub-agent tasks are heavily repetitive and highly cacheable.
- **Cons**: Requires decorating internal agent node methods.
- **Tool**: Use `@cache.cached(namespace="intent_classifier")`.

---

## 3. RAG Retrieval Caching

Cache the synthesis step of RAG pipelines based on query similarity.

```python
@cache.cached(namespace="rag_synthesis", ttl=14400)
def generate_grounded_answer(query: str, retrieved_docs_hash: str) -> str:
    return llm.invoke(f"Query: {query}\nContext: {docs}")
```

---

## 4. Cache Pre-Warming

To ensure high hit rates from day 1, you can pre-seed your cache with canonical questions and golden answers:

```python
from longcache import LongCache

cache = LongCache()

faq_knowledge_base = [
    (
        "What payment methods do you accept?",
        "We accept Visa, Mastercard, AMEX, PayPal, and Apple Pay.",
    ),
    ("How long does shipping take?", "Standard shipping takes 3-5 business days."),
]

for question, answer in faq_knowledge_base:
    cache.set(question, answer, ttl=86400 * 30)
```
