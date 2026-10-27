# LongCache Roadmap

This document outlines the strategic priorities, upcoming features, and long-term vision for **LongCache**.

---

## 🎯 Vision

LongCache aims to be the fastest, most reliable, and easiest-to-adopt semantic caching layer for AI agents, RAG architectures, and enterprise LLM applications.

---

## 📍 Current Release: v0.1.0

- [x] Zero-dependency core local embedder (`BAAI/bge-small-en-v1.5`)
- [x] FAISS CPU vector index with Flat and HNSW support
- [x] Memory, SQLite, and Redis (Sync & Async) backends
- [x] Integrations: LangChain, LangGraph, LlamaIndex, OpenAI, and FastAPI
- [x] CLI for inspection, stats, eviction, and data export
- [x] CI/CD workflows, automated packaging, and MkDocs site

---

## 🚀 Near-Term: v0.2.0

### Vector Store & Backend Extensions
- [ ] **PgVector Backend**: Direct PostgreSQL vector caching via `asyncpg` / `SQLAlchemy`.
- [ ] **Qdrant & Chroma Backends**: Native storage adapters for external vector databases.
- [ ] **Dynamic Threshold Tuning**: Auto-adjust similarity threshold per namespace based on negative feedback or domain complexity.

### Multi-Modal & Structured Caching
- [ ] **Vision / Multi-Modal Cache**: Semantic caching for image input queries (CLIP / SigLIP embeddings).
- [ ] **Structured Output Caching**: JSON schema validation and structured Pydantic model response preservation.

---

## ⚡ Mid-Term: v0.3.0

### Production Observability & Intelligence
- [ ] **OpenTelemetry Integration**: Traces for cache lookup duration, cosine similarity distance, and hit/miss span tags.
- [ ] **Prometheus Metrics Exporter**: Native `/metrics` endpoint for Grafana dashboards.
- [ ] **Semantic Clustering & Invalidation**: Invalidate entire topic clusters when underlying knowledge base / documentation changes.
- [ ] **Cache Pre-Warming**: Pre-populate cache from golden FAQ datasets, documentation sites, or past evaluation logs.

---

## 🌐 Long-Term Vision: v1.0.0+

- [ ] **Distributed Raft Cluster**: High-availability embedded clustering without requiring external Redis clusters.
- [ ] **Semantic Routing**: Route query variations to optimal specialized models (e.g. SLMs vs large frontier LLMs) based on semantic proximity to cached competencies.
- [ ] **Streaming First-Token Acceleration**: Instant streaming replay of cached tokens with sub-millisecond time-to-first-token (TTFT).

---

## 💡 Suggesting Roadmap Items

Have an idea or need an integration for your production stack? Open an [Integration Request](https://github.com/ENDEVSOLS/LongCache/issues/new?template=integration_request.yml) or start a discussion!
