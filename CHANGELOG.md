# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-27

### Added

- **Core Semantic Cache Engine:**
  - Fast local embeddings using `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors) running fully offline.
  - FAISS index integration with support for exact flat cosine similarity and scalable HNSW indexing.
  - Transparent sync and async decorator `@cache.cached` with parameter hashing, TTL expiry, and namespace scoping.
  - Composite cache keys to prevent cross-model poisoning (scoped by model, system prompt, temperature, and query).
- **Pluggable Storage Backends:**
  - `SQLiteBackend`: Zero-configuration local disk persistence with transactional safety and WAL mode.
  - `RedisBackend` & `AsyncRedisBackend`: High-throughput distributed caching for multi-container and cluster deployments.
  - `MemoryBackend`: High-speed in-memory storage for unit tests, ephemeral workers, and local experiments.
- **Framework Integrations:**
  - `langchain`: Semantic caching adapter for LangChain LLMs and runnables.
  - `langgraph`: Contextual state cache node for multi-agent workflows.
  - `llamaindex`: Semantic cache wrapper for LlamaIndex query engines.
  - `openai`: Drop-in wrapper for `openai.OpenAI` and `openai.AsyncOpenAI` chat completions.
  - `fastapi`: ASGI HTTP middleware for automatic semantic caching of REST / SSE endpoints.
- **CLI & Diagnostics:**
  - `longcache stats`: Real-time cache hit rate, token savings, and cost metrics.
  - `longcache clear`: Scoped namespace and total cache eviction.
  - `longcache inspect`: Query lookup, similarity breakdown, and payload inspection.
  - `longcache export`: JSON and CSV dataset exporting for cache analytics.
- **Developer Tools & CI/CD:**
  - CI workflow with Python 3.10, 3.11, and 3.12 test matrix.
  - Automated GitHub Pages documentation deployment with Material for MkDocs.
  - Auto-tagging and GitHub release generation on main branch merge.
  - Automated PyPI publishing pipeline with Trusted Publisher authentication.
