"""
Configuration for LongCache.

Provides CacheConfig dataclass with sensible defaults,
validation, and factory methods for construction from dicts or env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CacheConfig:
    """
    Configuration for LongCache with sensible defaults.

    Attributes:
        similarity_threshold: Cosine similarity threshold for cache hits (0.0-1.0).
            Higher = stricter matching. 0.92 is a good default for most RAG use cases.
        ttl_seconds: Time-to-live for cache entries in seconds. 0 = no expiry.
        max_entries: Maximum number of entries per namespace. Oldest entries
            are evicted first when the limit is reached.
        namespace: Default namespace for cache entries. Namespaces isolate
            different applications or use cases sharing the same backend.
        embedder_model: Sentence-transformers model name for local embedding.
            Must be compatible with sentence-transformers library.
        backend: Storage backend type. One of: "sqlite", "redis", "memory".
        db_path: Path to the SQLite database file (only used when backend="sqlite").
        redis_url: Redis connection URL (only used when backend="redis").
        track_stats: Whether to track hit/miss statistics internally.
    """

    similarity_threshold: float = 0.92
    ttl_seconds: int = 3600
    max_entries: int = 10_000
    namespace: str = "default"
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    backend: str = "sqlite"
    db_path: str = ".longcache/cache.db"
    redis_url: str = "redis://localhost:6379/0"
    track_stats: bool = True
    faiss_index_type: str = "flat"
    hnsw_m: int = 32
    hnsw_ef_search: int = 64
    model: str = ""
    system_prompt: str = ""
    temperature: float = -1.0

    def __post_init__(self):
        """Validate configuration values after initialization."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )
        if self.ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be >= 0, got {self.ttl_seconds}")
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.backend not in ("sqlite", "redis", "async_redis", "memory"):
            raise ValueError(
                f"backend must be one of 'sqlite', 'redis', 'async_redis', 'memory', "
                f"got '{self.backend}'"
            )

    @classmethod
    def from_dict(cls, data: dict) -> CacheConfig:
        """
        Create a CacheConfig from a dictionary.

        Only keys that match CacheConfig fields are used; others are ignored.

        Args:
            data: Dictionary with configuration keys.

        Returns:
            CacheConfig instance with values from the dict.
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_env(cls, prefix: str = "LONGCACHE_") -> CacheConfig:
        """
        Create a CacheConfig from environment variables.

        Environment variables are mapped by uppercasing the field name and
        prepending the prefix. For example, similarity_threshold becomes
        LONGCACHE_SIMILARITY_THRESHOLD.

        Args:
            prefix: Environment variable prefix. Defaults to "LONGCACHE_".

        Returns:
            CacheConfig instance with values from environment variables.
        """
        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        env_values = {}

        for field_name in cls.__dataclass_fields__:
            env_key = f"{prefix}{field_name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                # Cast to appropriate type
                field_type = field_types[field_name]
                if field_type == "float":
                    env_values[field_name] = float(env_val)
                elif field_type == "int":
                    env_values[field_name] = int(env_val)
                elif field_type == "bool":
                    env_values[field_name] = env_val.lower() in ("true", "1", "yes")
                else:
                    env_values[field_name] = env_val

        return cls(**env_values)

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)
