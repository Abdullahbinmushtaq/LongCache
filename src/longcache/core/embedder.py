"""
Local embedding engine for LongCache.

Uses sentence-transformers to generate embeddings locally with no API calls.
The default model (all-MiniLM-L6-v2) produces 384-dimensional vectors
in under 5ms on modern hardware.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from longcache.core.base_embedder import BaseEmbedder

_log = logging.getLogger(__name__)


class LocalEmbedder(BaseEmbedder):
    """
    Local embedding engine using sentence-transformers.

    Loads the model lazily on first use to avoid import penalties at module level.
    Thread-safe — model loading is protected by a lock.

    Usage:
        embedder = LocalEmbedder()
        vector = embedder.embed("What is your refund policy?")
        # vector.shape = (384,) for MiniLM-L6-v2
    """

    # Class-level cache for model instances (singleton per model name)
    _instances: dict[str, LocalEmbedder] = {}
    _lock = threading.Lock()

    def __new__(cls, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Singleton pattern: one embedder instance per model name.

        This avoids loading the same model multiple times in memory.

        Args:
            model_name: The sentence-transformers model identifier.

        Returns:
            The singleton LocalEmbedder instance for the given model.
        """
        with cls._lock:
            if model_name not in cls._instances:
                instance = super().__new__(cls)
                instance._model_name = model_name
                instance._model = None
                instance._dimension = None
                instance._model_lock = threading.Lock()
                cls._instances[model_name] = instance
            return cls._instances[model_name]

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedder. Model is loaded lazily on first embed() call.

        Args:
            model_name: The sentence-transformers model identifier.
                Defaults to all-MiniLM-L6-v2 (384-dim, sub-5ms).
        """
        # __new__ handles actual initialization; __init__ just ensures
        # the model name is set (idempotent for singleton)
        self._model_name = model_name

    def _load_model(self):
        """
        Lazily load the sentence-transformers model.

        This is called on the first embed() invocation, not at import time,
        so importing longcache does not trigger a heavy model download.
        """
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for LocalEmbedder. "
                    "Install with: uv pip install longcache"
                )

            _log.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_embedding_dimension()
            _log.info("Embedding model loaded: dim=%d", self._dimension)

    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension for this model.

        Loads the model if not already loaded.

        Returns:
            The dimensionality of the embedding vectors.
        """
        self._load_model()
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for the input text.

        The model is loaded lazily on the first call. Subsequent calls
        use the cached model instance.

        Args:
            text: The input text to embed.

        Returns:
            L2-normalized numpy float32 array of shape (dimension,).
            Normalization ensures cosine similarity equals inner product
            when using FAISS IndexFlatIP.
        """
        self._load_model()
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate embedding vectors for multiple texts in a single batch.

        More efficient than calling embed() in a loop for multiple texts
        because the model can parallelize computation across the batch.

        Args:
            texts: List of input texts to embed.

        Returns:
            numpy float32 array of shape (len(texts), dimension).
            Each row is L2-normalized.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        self._load_model()
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vectors.astype(np.float32)

    def __repr__(self) -> str:
        model_status = "loaded" if self._model is not None else "not loaded"
        return f"LocalEmbedder(model={self._model_name!r}, status={model_status})"
