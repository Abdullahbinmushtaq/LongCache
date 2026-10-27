"""
Shared test fixtures for LongCache tests.

Provides mock embedder, temporary database, and common test data
to avoid repeating setup across test files.
"""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from longcache.config import CacheConfig
from longcache.core.entry import CacheEntry
from longcache.core.embedder import LocalEmbedder


# ---------- Fixtures ----------


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_db_path(temp_dir):
    """Provide a temporary SQLite database path."""
    return os.path.join(temp_dir, "test_cache.db")


@pytest.fixture
def memory_config():
    """CacheConfig using the in-memory backend."""
    return CacheConfig(backend="memory", track_stats=True)


@pytest.fixture
def sqlite_config(temp_db_path):
    """CacheConfig using the SQLite backend with a temporary database."""
    return CacheConfig(backend="sqlite", db_path=temp_db_path, track_stats=True)


@pytest.fixture
def mock_embedder():
    """
    Mock embedder that returns deterministic vectors.

    Uses a simple hash-based approach to generate consistent
    embedding vectors for testing, avoiding the need to download
    the actual MiniLM model.
    """
    embedder = MagicMock(spec=LocalEmbedder)
    embedder._model = MagicMock()  # Mark as loaded
    embedder._dimension = 384

    def deterministic_embed(text: str) -> np.ndarray:
        """Generate a deterministic vector from text using hashing."""
        np.random.seed(hash(text) % (2**31))
        vector = np.random.randn(384).astype(np.float32)
        vector /= np.linalg.norm(vector)  # L2 normalize
        return vector

    embedder.embed = deterministic_embed
    embedder.dimension = 384
    return embedder


@pytest.fixture
def sample_vectors():
    """Provide sample embedding vectors for testing similarity search."""
    # Create two similar vectors and one dissimilar vector
    base = np.random.randn(384).astype(np.float32)
    base /= np.linalg.norm(base)

    similar = base + np.random.randn(384).astype(np.float32) * 0.05
    similar /= np.linalg.norm(similar)

    dissimilar = np.random.randn(384).astype(np.float32)
    dissimilar /= np.linalg.norm(dissimilar)

    return {
        "base": base,
        "similar": similar,
        "dissimilar": dissimilar,
    }


@pytest.fixture
def sample_entries(sample_vectors):
    """Provide sample CacheEntry objects for testing."""
    return [
        CacheEntry(
            id=1,
            namespace="default",
            query="What is the refund policy?",
            vector=sample_vectors["base"],
            response="Our refund policy allows returns within 30 days.",
            created_at=1000000.0,
            expires_at=None,
            hit_count=0,
        ),
        CacheEntry(
            id=2,
            namespace="default",
            query="How do I get a refund?",
            vector=sample_vectors["similar"],
            response="You can request a refund within 30 days of purchase.",
            created_at=1000001.0,
            expires_at=None,
            hit_count=0,
        ),
        CacheEntry(
            id=3,
            namespace="faq",
            query="What are your business hours?",
            vector=sample_vectors["dissimilar"],
            response="We are open Mon-Fri, 9am-5pm.",
            created_at=1000002.0,
            expires_at=None,
            hit_count=0,
        ),
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM that returns canned responses."""
    call_count = 0

    def mock_invoke(query: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Response to: {query} (call #{call_count})"

    return mock_invoke
