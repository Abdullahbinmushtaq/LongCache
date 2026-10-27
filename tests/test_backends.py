"""Tests for cache storage backends."""

import os
import time
import tempfile

import numpy as np
import pytest

from longcache.core.entry import CacheEntry
from longcache.backends.memory import InMemoryCacheBackend
from longcache.backends.sqlite import SQLiteCacheBackend


# ---------- Helpers ----------


def make_entry(
    namespace: str = "default",
    query: str = "test query",
    response: str = "test response",
    expires_at: float | None = None,
) -> CacheEntry:
    """Create a test CacheEntry with a random normalized vector."""
    vector = np.random.randn(384).astype(np.float32)
    vector /= np.linalg.norm(vector)
    return CacheEntry(
        namespace=namespace,
        query=query,
        vector=vector,
        response=response,
        created_at=time.time(),
        expires_at=expires_at,
    )


# ---------- InMemory Backend ----------


class TestInMemoryBackend:
    def test_store_and_retrieve(self):
        """Test basic store and retrieve."""
        backend = InMemoryCacheBackend()
        entry = make_entry()
        entry_id = backend.store(entry)
        assert entry_id > 0

        result = backend.retrieve(entry_id)
        assert result is not None
        assert result.query == "test query"
        assert result.response == "test response"

    def test_retrieve_nonexistent(self):
        """Test retrieving a non-existent entry."""
        backend = InMemoryCacheBackend()
        result = backend.retrieve(9999)
        assert result is None

    def test_delete(self):
        """Test deleting an entry."""
        backend = InMemoryCacheBackend()
        entry = make_entry()
        entry_id = backend.store(entry)

        assert backend.delete(entry_id) is True
        assert backend.retrieve(entry_id) is None

    def test_delete_nonexistent(self):
        """Test deleting a non-existent entry."""
        backend = InMemoryCacheBackend()
        assert backend.delete(9999) is False

    def test_delete_namespace(self):
        """Test deleting all entries in a namespace."""
        backend = InMemoryCacheBackend()
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_B"))

        count = backend.delete_namespace("ns_A")
        assert count == 2
        assert backend.count("ns_A") == 0
        assert backend.count("ns_B") == 1

    def test_delete_by_tag(self):
        """Test deleting entries by tag."""
        backend = InMemoryCacheBackend()
        e1 = make_entry(namespace="ns_A", query="q1")
        e1.tags = ["tag1", "tag2"]
        e2 = make_entry(namespace="ns_A", query="q2")
        e2.tags = ["tag2"]
        e3 = make_entry(namespace="ns_A", query="q3")
        e3.tags = ["tag3"]
        backend.store(e1)
        backend.store(e2)
        backend.store(e3)

        deleted = backend.delete_by_tag("tag1", "ns_A")
        assert deleted == 1
        assert backend.count("ns_A") == 2

    def test_clear(self):
        """Test clearing all entries."""
        backend = InMemoryCacheBackend()
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_B"))

        count = backend.clear()
        assert count == 2
        assert backend.count("ns_A") == 0

    def test_count(self):
        """Test counting entries per namespace."""
        backend = InMemoryCacheBackend()
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_B"))

        assert backend.count("ns_A") == 2
        assert backend.count("ns_B") == 1
        assert backend.count("ns_C") == 0

    def test_list_entries(self):
        """Test listing entries in a namespace."""
        backend = InMemoryCacheBackend()
        backend.store(make_entry(query="q1", namespace="ns_A"))
        backend.store(make_entry(query="q2", namespace="ns_A"))
        backend.store(make_entry(query="q3", namespace="ns_B"))

        entries = backend.list_entries("ns_A")
        assert len(entries) == 2
        queries = {e.query for e in entries}
        assert queries == {"q1", "q2"}

    def test_update_hit_count(self):
        """Test incrementing hit count."""
        backend = InMemoryCacheBackend()
        entry_id = backend.store(make_entry())

        backend.update_hit_count(entry_id)
        result = backend.retrieve(entry_id)
        assert result.hit_count == 1

        backend.update_hit_count(entry_id)
        result = backend.retrieve(entry_id)
        assert result.hit_count == 2

    def test_evict_oldest(self):
        """Test LRU eviction of oldest entries."""
        backend = InMemoryCacheBackend()

        # Add entries with increasing created_at
        for i in range(5):
            entry = make_entry(query=f"q{i}")
            entry.created_at = 1000.0 + i
            backend.store(entry)

        # Evict 2 oldest
        evicted = backend.evict_oldest("default", 2)
        assert evicted == 2

        # Should have 3 entries remaining
        assert backend.count("default") == 3

    def test_expired_entry_not_returned(self):
        """Test that expired entries are not returned by retrieve()."""
        backend = InMemoryCacheBackend()
        entry = make_entry(expires_at=time.time() - 1)
        entry_id = backend.store(entry)

        result = backend.retrieve(entry_id)
        assert result is None

    def test_get_expired(self):
        """Test getting IDs of expired entries."""
        backend = InMemoryCacheBackend()
        expired_id = backend.store(make_entry(expires_at=time.time() - 1))
        valid_id = backend.store(make_entry(expires_at=time.time() + 3600))

        expired = backend.get_expired("default")
        assert expired_id in expired
        assert valid_id not in expired


# ---------- SQLite Backend ----------


class TestSQLiteBackend:
    @pytest.fixture
    def backend(self, temp_dir):
        """Provide a fresh SQLite backend with a temporary database."""
        db_path = os.path.join(temp_dir, "test_cache.db")
        return SQLiteCacheBackend(db_path=db_path)

    def test_store_and_retrieve(self, backend):
        """Test basic store and retrieve."""
        entry = make_entry()
        entry_id = backend.store(entry)
        assert entry_id > 0

        result = backend.retrieve(entry_id)
        assert result is not None
        assert result.query == "test query"
        assert result.response == "test response"

    def test_vector_round_trip(self, backend):
        """Test that vectors survive serialization to/from SQLite."""
        original_entry = make_entry()
        entry_id = backend.store(original_entry)

        result = backend.retrieve(entry_id)
        assert result.vector is not None
        np.testing.assert_array_almost_equal(original_entry.vector, result.vector, decimal=6)

    def test_retrieve_nonexistent(self, backend):
        """Test retrieving a non-existent entry."""
        result = backend.retrieve(9999)
        assert result is None

    def test_delete(self, backend):
        """Test deleting an entry."""
        entry = make_entry()
        entry_id = backend.store(entry)

        assert backend.delete(entry_id) is True
        assert backend.retrieve(entry_id) is None

    def test_delete_namespace(self, backend):
        """Test deleting all entries in a namespace."""
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_B"))

        count = backend.delete_namespace("ns_A")
        assert count == 2
        assert backend.count("ns_A") == 0
        assert backend.count("ns_B") == 1

    def test_clear(self, backend):
        """Test clearing all entries."""
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_B"))

        count = backend.clear()
        assert count == 2

    def test_count(self, backend):
        """Test counting entries."""
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_B"))

        assert backend.count("ns_A") == 2
        assert backend.count("ns_B") == 1

    def test_list_entries(self, backend):
        """Test listing entries."""
        backend.store(make_entry(query="q1", namespace="ns_A"))
        backend.store(make_entry(query="q2", namespace="ns_A"))

        entries = backend.list_entries("ns_A")
        assert len(entries) == 2

    def test_update_hit_count(self, backend):
        """Test incrementing hit count."""
        entry_id = backend.store(make_entry())

        backend.update_hit_count(entry_id)
        result = backend.retrieve(entry_id)
        assert result.hit_count == 1

    def test_evict_oldest(self, backend):
        """Test LRU eviction."""
        for i in range(5):
            entry = make_entry(query=f"q{i}")
            entry.created_at = 1000.0 + i
            backend.store(entry)

        evicted = backend.evict_oldest("default", 2)
        assert evicted == 2

    def test_expired_entry_auto_deleted(self, backend):
        """Test that expired entries are auto-deleted on retrieval."""
        entry = make_entry(expires_at=time.time() - 1)
        entry_id = backend.store(entry)

        result = backend.retrieve(entry_id)
        assert result is None

    def test_get_expired(self, backend):
        """Test getting expired entry IDs."""
        backend.store(make_entry(expires_at=time.time() - 1))
        backend.store(make_entry(expires_at=time.time() + 3600))

        expired = backend.get_expired("default")
        assert len(expired) >= 1

    def test_export_all(self, backend):
        """Test exporting all entries."""
        backend.store(make_entry(query="q1"))
        backend.store(make_entry(query="q2"))

        entries = backend.export_all()
        assert len(entries) >= 2
        assert all("query" in e for e in entries)

    def test_schema_creation(self, temp_dir):
        """Test that the database and schema are created automatically."""
        db_path = os.path.join(temp_dir, "new_cache.db")
        assert not os.path.exists(db_path)

        _ = SQLiteCacheBackend(db_path=db_path)
        assert os.path.exists(db_path)

    def test_directory_auto_created(self, temp_dir):
        """Test that parent directories are created automatically."""
        db_path = os.path.join(temp_dir, "subdir", "deep", "cache.db")
        backend = SQLiteCacheBackend(db_path=db_path)
        # Should not raise an error
        backend.store(make_entry())

    def test_json_response_round_trip(self, backend):
        """Test that JSON-serializable responses survive storage."""
        entry = make_entry(response={"key": "value", "nested": [1, 2, 3]})
        entry_id = backend.store(entry)

        result = backend.retrieve(entry_id)
        assert isinstance(result.response, dict)
        assert result.response["key"] == "value"
        assert result.response["nested"] == [1, 2, 3]

    def test_delete_by_tag(self, backend):
        """Test deleting entries by tag in SQLite."""
        e1 = make_entry(namespace="ns_A", query="q1")
        e1.tags = ["tag1", "tag2"]
        e2 = make_entry(namespace="ns_A", query="q2")
        e2.tags = ["tag2"]
        e3 = make_entry(namespace="ns_A", query="q3")
        e3.tags = ["tag3"]
        id1 = backend.store(e1)
        id2 = backend.store(e2)
        id3 = backend.store(e3)

        deleted = backend.delete_by_tag("tag2", "ns_A")
        assert deleted == 2
        assert backend.retrieve(id1) is None
        assert backend.retrieve(id2) is None
        assert backend.retrieve(id3) is not None
