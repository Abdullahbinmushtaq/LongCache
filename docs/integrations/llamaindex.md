# LlamaIndex Integration

LongCache integrates directly with LlamaIndex query engines via `CachedQueryEngine`.

## Installation

```bash
uv pip install "longcache[llamaindex]"
# or
pip install "longcache[llamaindex]"
```

---

## Caching Query Engines (`CachedQueryEngine`)

Wrap any standard `BaseQueryEngine` to intercept repeat or paraphrased queries before index retrieval:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from longcache import LongCache
from longcache.integrations.llamaindex import CachedQueryEngine

# 1. Load documents and create standard LlamaIndex engine
documents = SimpleDirectoryReader("data/docs").load_data()
index = VectorStoreIndex.from_documents(documents)
raw_engine = index.as_query_engine()

# 2. Wrap with LongCache
cached_engine = CachedQueryEngine(query_engine=raw_engine, cache=LongCache(), namespace="kb_docs")

# Call 1: Misses cache, searches index, synthesizes response with LLM
res1 = cached_engine.query("What is the warranty period for hardware?")
print(res1)

# Call 2: Paraphrased query — hits cache instantly!
res2 = cached_engine.query("How long is the hardware guaranteed?")
print(res2)
```

Async query engines are supported via `aquery()`:

```python
response = await cached_engine.aquery("How long is the warranty?")
```
