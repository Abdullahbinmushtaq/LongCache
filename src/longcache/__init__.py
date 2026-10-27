"""
LongCache — Semantic similarity cache for LLM/RAG pipelines.

Reuse responses for semantically similar queries when their context matches.
"""

from longcache.core.cache import LongCache
from longcache.config import CacheConfig

__version__ = "0.1.1"
__all__ = ["LongCache", "CacheConfig"]
