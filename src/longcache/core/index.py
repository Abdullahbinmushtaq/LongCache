"""
Similarity index for LongCache using FAISS.

Provides fast approximate nearest neighbor search over embedding vectors.
Uses IndexFlatIP (inner product) for exact search, which works well
for caches with up to ~100K entries. For larger caches, consider
switching to an IVF index.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

import numpy as np

from longcache.core.entry import CacheEntry


class SimilarityIndex:
    """
    FAISS-based similarity index for cache lookups.

    Maintains an in-memory FAISS index alongside a mapping from
    internal FAISS IDs to CacheEntry objects. All operations are
    thread-safe via a reentrant lock.

    Usage:
        index = SimilarityIndex(backend, config)
        index.add(vector, entry_id)
        result = index.find_similar(query_vector, threshold=0.92)
    """

    def __init__(self, backend, config):
        """
        Initialize the similarity index.

        Args:
            backend: The cache backend (used for retrieving entries by ID).
            config: CacheConfig instance with similarity_threshold and other settings.
        """
        self._backend = backend
        self._config = config
        self._lock = threading.RLock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._faiss_index = None
        self._id_map: dict[int, int] = {}  # faiss_internal_id -> entry_id
        self._next_faiss_id = 0
        self._dimension = None
        self._orphan_count = 0  # vectors in FAISS with no id_map entry
        self._REBUILD_THRESHOLD = 0.20  # rebuild when 20% of index is orphaned
        self._removed_entry_ids: set[int] = set()  # entry ids explicitly removed via remove()

    def _get_async_lock(self) -> asyncio.Lock:
        """Lazily create asyncio.Lock on first async use."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _ensure_index(self, dimension: int):
        """
        Lazily create the FAISS index with the given dimension.

        Called automatically on first add() with a vector of the
        appropriate dimension.

        Args:
            dimension: The dimensionality of the embedding vectors.
        """
        if self._faiss_index is not None:
            return

        import faiss

        index_type = getattr(self._config, "faiss_index_type", "flat")
        if index_type == "hnsw":
            m = getattr(self._config, "hnsw_m", 32)
            ef = getattr(self._config, "hnsw_ef_search", 64)
            # HNSW with inner product metric (cosine for normalized vectors)
            self._faiss_index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
            self._faiss_index.hnsw.efSearch = ef
        else:
            # Default: exact flat index
            self._faiss_index = faiss.IndexFlatIP(dimension)

        self._dimension = dimension

    def add(self, vector: np.ndarray, entry_id: int) -> None:
        """
        Add an embedding vector to the index.

        Creates the FAISS index on first call, inferring dimension
        from the input vector.

        Args:
            vector: L2-normalized embedding vector of shape (dimension,).
            entry_id: The database ID of the corresponding CacheEntry.
        """
        with self._lock:
            if self._faiss_index is None:
                self._ensure_index(vector.shape[0])

            # FAISS expects 2D arrays
            vector_2d = vector.reshape(1, -1).astype(np.float32)
            faiss_id = self._next_faiss_id
            # Assign id_map BEFORE calling FAISS to keep mapping consistent
            self._id_map[faiss_id] = entry_id
            self._faiss_index.add(vector_2d)
            self._next_faiss_id += 1

    async def async_add(self, vector: np.ndarray, entry_id: int) -> None:
        """Async-safe version of add() for use in async contexts."""
        async with self._get_async_lock():
            self.add(vector, entry_id)

    def find_similar(
        self,
        vector: np.ndarray,
        threshold: float = 0.92,
        namespace: str = "default",
    ) -> Optional[CacheEntry]:
        """
        Find the most similar cached entry for the given query vector.

        Searches the FAISS index for the nearest neighbor, then verifies
        the entry exists in the backend, belongs to the correct namespace,
        and meets the similarity threshold.

        Args:
            vector: L2-normalized query embedding vector.
            threshold: Minimum cosine similarity for a cache hit.
            namespace: Only return entries from this namespace.

        Returns:
            CacheEntry if a similar entry is found above the threshold,
            None otherwise.
        """
        with self._lock:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return None

            # Search for top-k (k=1) nearest neighbors
            vector_2d = vector.reshape(1, -1).astype(np.float32)
            similarities, indices = self._faiss_index.search(vector_2d, 1)

            if len(indices) == 0 or indices[0][0] < 0:
                return None

            best_similarity = float(similarities[0][0])
            best_faiss_id = int(indices[0][0])

            if best_similarity < threshold:
                return None

            # Look up the entry ID from our mapping
            entry_id = self._id_map.get(best_faiss_id)
            if entry_id is None:
                return None

            # Retrieve the full entry from the backend
            entry = self._backend.retrieve(entry_id)
            if entry is None:
                return None

            # Verify namespace
            if entry.namespace != namespace:
                return None

            # Verify not expired
            if entry.is_expired():
                return None

            # Update hit count
            entry.increment_hit()
            self._backend.update_hit_count(entry_id)

            return entry

    def remove(self, faiss_id: int) -> None:
        """
        Remove an entry from the ID map by its FAISS internal ID.

        Note: FAISS IndexFlatIP does not support efficient removal.
        The entry is removed from the ID map, so it won't be returned
        in future searches even though the vector remains in the index.
        For production use with high churn, consider rebuilding the index
        periodically.

        Args:
            faiss_id: The internal FAISS index ID to remove.
        """
        with self._lock:
            if faiss_id in self._id_map:
                entry_id = self._id_map.pop(faiss_id)
                self._removed_entry_ids.add(entry_id)
                self._orphan_count += 1
                self._auto_rebuild_if_needed()

    def _auto_rebuild_if_needed(self) -> None:
        """Rebuild FAISS index if orphan ratio exceeds threshold."""
        total = self._faiss_index.ntotal if self._faiss_index else 0
        if total > 0 and self._orphan_count / total >= self._REBUILD_THRESHOLD:
            import logging

            logging.getLogger(__name__).info(
                "Auto-rebuilding FAISS index: %d orphaned vectors (%.0f%%)",
                self._orphan_count,
                (self._orphan_count / total) * 100,
            )
            entries = self._backend.list_entries(self._config.namespace)
            entries = [e for e in entries if e.id not in self._removed_entry_ids]
            self.rebuild(entries)
            self._orphan_count = 0

    def rebuild(self, entries: list[CacheEntry]) -> None:
        """
        Rebuild the FAISS index from a list of cache entries.

        Useful for compacting the index after many removals, or for
        loading the index from persistent storage on startup.

        Args:
            entries: List of CacheEntry objects with their vectors populated.
        """
        with self._lock:
            import faiss

            valid_entries = [
                e for e in (entries or []) if e.vector is not None and e.id is not None
            ]

            if not valid_entries:
                self._faiss_index = None
                self._id_map = {}
                self._next_faiss_id = 0
                self._orphan_count = 0
                self._removed_entry_ids.clear()
                return

            dimension = valid_entries[0].vector.shape[0]
            self._ensure_index(dimension)

            # Reset index
            self._faiss_index = faiss.IndexFlatIP(dimension)
            self._id_map = {}
            self._next_faiss_id = 0

            # Add all valid entries
            vectors = []
            for entry in valid_entries:
                vectors.append(entry.vector)
                self._id_map[self._next_faiss_id] = entry.id
                self._next_faiss_id += 1

            if vectors:
                vectors_array = np.stack(vectors).astype(np.float32)
                self._faiss_index.add(vectors_array)

            self._orphan_count = 0
            self._removed_entry_ids.clear()

    @property
    def size(self) -> int:
        """Return the number of entries currently in the index."""
        with self._lock:
            return self._faiss_index.ntotal if self._faiss_index is not None else 0

    def __repr__(self) -> str:
        size = self.size
        dim = self._dimension or "uninitialized"
        return f"SimilarityIndex(size={size}, dimension={dim})"
