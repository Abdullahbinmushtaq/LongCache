"""
LongCache — Semantic similarity cache for LLM/RAG pipelines.

Cut repeat query costs by 40%+ by caching semantically similar queries.
"""

from longcache.core.cache import LongCache
from longcache.config import CacheConfig

__version__ = "0.1.0"
__all__ = ["LongCache", "CacheConfig"]
