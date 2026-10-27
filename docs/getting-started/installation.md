# Installation

LongCache requires **Python 3.10** or newer.

## Installing via uv (Recommended)

[uv](https://github.com/astral-sh/uv) provides instant, reproducible installations:

```bash
# Core package (SQLite backend + FAISS + BGE-small local embedder)
uv pip install longcache

# With LangChain & LangGraph
uv pip install "longcache[langchain]"

# With Redis for multi-worker production
uv pip install "longcache[redis]"

# With LlamaIndex
uv pip install "longcache[llamaindex]"

# With OpenAI official client wrapper
uv pip install "longcache[openai]"

# With FastAPI / Starlette middleware
uv pip install "longcache[server]"

# Full installation (all integrations)
uv pip install "longcache[all]"
```

## Installing via pip

```bash
# Standard installation
pip install longcache

# Full bundle
pip install "longcache[all]"
```

!!! tip "CPU PyTorch Optimization"
    LongCache uses PyTorch purely for local embedding inference with `sentence-transformers`. In production environments without GPUs, you can install the lightweight CPU-only wheel of PyTorch (~180MB instead of ~3GB):
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install longcache
    ```

## Offline Usage & First Run

On the very first run, LongCache automatically downloads the compact **BAAI/bge-small-en-v1.5** model weights (~130MB) into your local Hugging Face cache (`~/.cache/huggingface/hub`).

Once downloaded, LongCache works **100% offline** without any outgoing network connections required for embedding queries.

To pre-warm the model in container builds (e.g. Dockerfile):

```bash
python -c "from longcache.core.embedder import LocalEmbedder; LocalEmbedder().embed('warmup')"
```
