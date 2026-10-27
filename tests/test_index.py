"""Tests for the SimilarityIndex component."""

import numpy as np
import pytest

from longcache.config import CacheConfig
from longcache.core.entry import CacheEntry
from longcache.core.index import SimilarityIndex
from longcache.backends.memory import InMemoryCacheBackend


@pytest.fixture
def backend():
    """Provide a fresh in-memory backend."""
    return InMemoryCacheBackend()


@pytest.fixture
def config():
    """Provide a default CacheConfig."""
    return CacheConfig(backend="memory", similarity_threshold=0.92)


@pytest.fixture
def index(backend, config):
    """Provide a fresh SimilarityIndex."""
    return SimilarityIndex(backend, config)


class TestSimilarityIndex:
    def test_add_and_find(self, index, backend):
        """Test adding a vector and finding it via similarity search."""
        vector = np.random.randn(384).astype(np.float32)
        vector /= np.linalg.norm(vector)

        # Create and store an entry in the backend first
        entry = CacheEntry(
            namespace="default",
            query="test query",
            vector=vector,
            response="test response",
        )
        entry_id = backend.store(entry)
        entry.id = entry_id

        # Add to index
        index.add(vector, entry_id)

        # Search with the same vector
        result = index.find_similar(vector, threshold=0.9, namespace="default")
        assert result is not None
        assert result.query == "test query"

    def test_find_similar_below_threshold(self, index, backend):
        """Test that dissimilar vectors don't match above threshold."""
        vector1 = np.random.randn(384).astype(np.float32)
        vector1 /= np.linalg.norm(vector1)

        vector2 = np.random.randn(384).astype(np.float32)
        vector2 /= np.linalg.norm(vector2)

        entry = CacheEntry(
            namespace="default",
            query="test query",
            vector=vector1,
            response="test response",
        )
        entry_id = backend.store(entry)
        entry.id = entry_id

        index.add(vector1, entry_id)

        # Search with a very different vector — should not match
        result = index.find_similar(vector2, threshold=0.99, namespace="default")
        assert result is None

    def test_find_similar_wrong_namespace(self, index, backend):
        """Test that namespace filtering works."""
        vector = np.random.randn(384).astype(np.float32)
        vector /= np.linalg.norm(vector)

        entry = CacheEntry(
            namespace="namespace_A",
            query="test query",
            vector=vector,
            response="test response",
        )
        entry_id = backend.store(entry)
        entry.id = entry_id

        index.add(vector, entry_id)

        # Search in a different namespace — should not find it
        result = index.find_similar(vector, threshold=0.5, namespace="namespace_B")
        assert result is None

    def test_find_similar_expired_entry(self, index, backend):
        """Test that expired entries are not returned."""
        vector = np.random.randn(384).astype(np.float32)
        vector /= np.linalg.norm(vector)

        import time

        entry = CacheEntry(
            namespace="default",
            query="test query",
            vector=vector,
            response="test response",
            created_at=time.time() - 100,
            expires_at=time.time() - 1,  # expired 1 second ago
        )
        entry_id = backend.store(entry)
        entry.id = entry_id

        index.add(vector, entry_id)

        result = index.find_similar(vector, threshold=0.5, namespace="default")
        assert result is None

    def test_empty_index_returns_none(self, index):
        """Test that searching an empty index returns None."""
        vector = np.random.randn(384).astype(np.float32)
        result = index.find_similar(vector, threshold=0.5, namespace="default")
        assert result is None

    def test_size_property(self, index, backend):
        """Test the size property of the index."""
        assert index.size == 0

        vector = np.random.randn(384).astype(np.float32)
        vector /= np.linalg.norm(vector)

        entry = CacheEntry(
            namespace="default",
            query="test",
            vector=vector,
            response="response",
        )
        entry_id = backend.store(entry)
        index.add(vector, entry_id)

        assert index.size == 1

    def test_rebuild_from_entries(self, index, backend):
        """Test rebuilding the index from a list of entries."""
        vectors = []
        entries = []

        for i in range(5):
            v = np.random.randn(384).astype(np.float32)
            v /= np.linalg.norm(v)
            vectors.append(v)

            entry = CacheEntry(
                namespace="default",
                query=f"query {i}",
                vector=v,
                response=f"response {i}",
            )
            # Store in backend first so retrieve() works
            entry_id = backend.store(entry)
            entry.id = entry_id
            entries.append(entry)

        # Rebuild
        index.rebuild(entries)
        assert index.size == 5

        # Should be able to find entries — search with the exact same vector
        result = index.find_similar(vectors[0], threshold=0.9, namespace="default")
        assert result is not None

    def test_hit_count_increment(self, index, backend):
        """Test that finding a similar entry increments its hit count."""
        vector = np.random.randn(384).astype(np.float32)
        vector /= np.linalg.norm(vector)

        entry = CacheEntry(
            namespace="default",
            query="test query",
            vector=vector,
            response="test response",
            hit_count=0,
        )
        entry_id = backend.store(entry)
        entry.id = entry_id

        index.add(vector, entry_id)

        # Find it twice
        index.find_similar(vector, threshold=0.5, namespace="default")
        index.find_similar(vector, threshold=0.5, namespace="default")

        # Check hit count in backend
        stored_entry = backend.retrieve(entry_id)
        assert stored_entry.hit_count >= 2

    def test_remove_from_id_map(self, index, backend):
        """Test removing an entry from the ID map."""
        vector = np.random.randn(384).astype(np.float32)
        vector /= np.linalg.norm(vector)

        entry = CacheEntry(
            namespace="default",
            query="test query",
            vector=vector,
            response="test response",
        )
        entry_id = backend.store(entry)
        index.add(vector, entry_id)

        assert index.size == 1

        # Remove via FAISS ID 0 (first added)
        index.remove(0)

        # Search should now return None
        result = index.find_similar(vector, threshold=0.5, namespace="default")
        assert result is None
