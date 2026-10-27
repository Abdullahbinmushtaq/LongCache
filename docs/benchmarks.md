# Performance and cost measurement

Latency and savings depend on the workload, embedding model, hardware, storage backend, threshold, and request shapes eligible for caching. The project does not promise a fixed response latency, hit rate, throughput, or percentage saving.

## Reproduce the namespace lookup benchmark

From the repository root:

```bash
uv sync --locked --extra all --extra dev
uv run --no-sync python tests/benchmarks/namespace_lookup.py
```

The benchmark uses 384-dimensional vectors and a single FAISS thread. It places the target entry behind 1,000 and 10,000 closer entries from other namespaces, runs each lookup 20 times, and asserts that the correct entry is returned. This exercises the progressive candidate search added in v0.1.1. It excludes embedding time, Redis/SQLite access, network calls, and concurrent traffic; its timings are a synthetic baseline.

## Evaluate an application

Record the dataset, model, threshold, cache size, namespace distribution, backend, interpreter, package versions, thread count, and hardware. Measure cold and warm requests separately. Include p50/p95/p99 end-to-end latency, memory use, hit rate, bypass rate, and incorrect-hit rate on a held-out set.

Compare actual provider usage before and after caching, including embedding and storage costs. A cache hit avoids a provider call, but a high hit rate is useful only if cached answers remain correct for the request. Dollar projections must state their assumed traffic, token usage, provider prices, and hit rate; they are not observed savings.
