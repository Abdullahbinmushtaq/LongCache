# LongCache ⚡

<p align="center">
  <img src="assets/logo.png" alt="LongCache Logo" width="280"/>
</p>

<p align="center"><strong>Production-grade semantic similarity cache for LLM and RAG pipelines.</strong></p>

---

## What is LongCache?

> *"What is your refund policy?" costs $0.003 the first time.*  
> *It should cost $0.0000 every time after that.*

**LongCache** intercepts incoming queries before they reach expensive LLM APIs or vector databases. Unlike traditional string-matching key-value caches, LongCache understands that:

* *"What is your return process?"*
* *"How can I get a refund?"*
* *"Do you accept returns on opened items?"*

all share the exact same underlying intent. By matching queries in high-dimensional semantic space with local embedding models and FAISS vector indices, LongCache delivers **sub-5ms cache hits** while slashing API token costs by **40% or more**.

---

## Key Features

- ⚡ **Zero-API Local Embeddings**: Uses `BAAI/bge-small-en-v1.5` locally on CPU. No OpenAI embedding API costs or network roundtrips required.
- 🚀 **Blazing-Fast Vector Search**: Powered by FAISS (Flat & HNSW) for sub-millisecond similarity lookups over tens of thousands of entries.
- 🛡️ **Composite Key Isolation**: Prevents cross-model and prompt poisoning by scoping entries by model name, system prompt, temperature, and namespace.
- 🔌 **Pluggable Storage Backends**: SQLite for zero-config persistence, Redis / Async-Redis for distributed clusters, and Memory for testing.
- 🧩 **First-Class Framework Adapters**: Drop-in decorators and wrappers for **LangChain**, **LangGraph**, **LlamaIndex**, **OpenAI**, and **FastAPI**.
- 📊 **CLI & Diagnostics**: Built-in CLI commands to check hit rates, inspect semantic similarities, evict entries, and estimate dollar savings.

---

## Quick Example

```python
from longcache import LongCache

cache = LongCache()

@cache.cached
def my_rag_pipeline(query: str) -> str:
    # Expensive retrieval + frontier model invocation
    return expensive_rag_call(query)

# Call 1: Cache MISS (takes ~2,000ms, calls LLM)
ans1 = my_rag_pipeline("What is your refund policy?")

# Call 2 (exact query): Cache HIT (takes ~5ms, 0 tokens)
ans2 = my_rag_pipeline("What is your refund policy?")

# Call 3 (paraphrase): Cache HIT (takes ~5ms, 0 tokens)
ans3 = my_rag_pipeline("How do I return a product?")
```

---

## Performance at a Glance

| Metric | Without LongCache | With LongCache (Hit) | Improvement |
|---|---|---|---|
| **Latency** | 1,500 – 3,500 ms | **3 – 8 ms** | **~400x faster** |
| **API Token Cost** | $0.002 – $0.015 / req | **$0.0000** | **100% saved on hit** |
| **Throughput** | Limited by API rate limits | **10,000+ QPS** (Redis) | **Uncapped** |
| **Repeat Cost Savings** | 0% | **40% – 65%** in production | **Dramatic ROI** |

---

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [How It Works](how-it-works/architecture.md)
- [Framework Integrations](integrations/langchain.md)
- [Storage Backends](backends/overview.md)
