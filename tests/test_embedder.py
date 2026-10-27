"""Tests for the LocalEmbedder component."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from longcache.core.embedder import LocalEmbedder


def _model_available():
    """Check if sentence-transformers is available and model can be loaded."""
    try:
        from sentence_transformers import SentenceTransformer

        return True
    except ImportError:
        return False


class TestLocalEmbedderUnit:
    """Unit tests for LocalEmbedder that don't require model download."""

    def test_singleton_pattern(self):
        """Test that same model name returns the same instance."""
        original_instances = LocalEmbedder._instances.copy()
        LocalEmbedder._instances = {}

        try:
            embedder1 = LocalEmbedder("test-model-1")
            embedder2 = LocalEmbedder("test-model-1")
            assert embedder1 is embedder2
        finally:
            LocalEmbedder._instances = original_instances

    def test_different_models_different_instances(self):
        """Test that different model names return different instances."""
        original_instances = LocalEmbedder._instances.copy()
        LocalEmbedder._instances = {}

        try:
            embedder1 = LocalEmbedder("test-model-A")
            embedder2 = LocalEmbedder("test-model-B")
            assert embedder1 is not embedder2
        finally:
            LocalEmbedder._instances = original_instances

    def test_repr_not_loaded(self):
        """Test string representation before model is loaded."""
        original_instances = LocalEmbedder._instances.copy()
        LocalEmbedder._instances = {}

        try:
            embedder = LocalEmbedder("test-model-repr")
            repr_str = repr(embedder)
            assert "test-model-repr" in repr_str
            assert "not loaded" in repr_str
        finally:
            LocalEmbedder._instances = original_instances

    def test_dimension_property_requires_loading(self):
        """Test that accessing dimension triggers model loading."""
        embedder = LocalEmbedder.__new__(LocalEmbedder)
        embedder._model_name = "test-model"
        embedder._model = None
        embedder._dimension = None
        embedder._model_lock = __import__("threading").RLock()

        with patch.object(embedder, "_load_model", side_effect=ImportError("test")):
            with pytest.raises(ImportError, match="test"):
                _ = embedder.dimension


class TestLocalEmbedderIntegration:
    """Integration tests that require the sentence-transformers model."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton cache for test isolation."""
        original = LocalEmbedder._instances.copy()
        LocalEmbedder._instances = {}
        yield
        LocalEmbedder._instances = original

    @pytest.mark.skipif(
        not _model_available(), reason="sentence-transformers not available or model not downloaded"
    )
    def test_embed_returns_correct_shape(self):
        """Test that embed() returns a vector of the expected dimension."""
        embedder = LocalEmbedder()
        vector = embedder.embed("Hello, world!")
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert vector.shape == (384,)

    @pytest.mark.skipif(
        not _model_available(), reason="sentence-transformers not available or model not downloaded"
    )
    def test_embed_is_normalized(self):
        """Test that the embedding vector is L2-normalized."""
        embedder = LocalEmbedder()
        vector = embedder.embed("Test normalization")
        norm = np.linalg.norm(vector)
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.skipif(
        not _model_available(), reason="sentence-transformers not available or model not downloaded"
    )
    def test_similar_queries_have_high_similarity(self):
        """Test that semantically similar queries produce similar vectors."""
        embedder = LocalEmbedder()
        v1 = embedder.embed("What is your refund policy?")
        v2 = embedder.embed("How do I get a refund?")
        similarity = np.dot(v1, v2)
        assert similarity > 0.7, f"Similar queries should have high similarity, got {similarity}"

    @pytest.mark.skipif(
        not _model_available(), reason="sentence-transformers not available or model not downloaded"
    )
    def test_dissimilar_queries_have_low_similarity(self):
        """Test that semantically different queries produce dissimilar vectors."""
        embedder = LocalEmbedder()
        v1 = embedder.embed("What is your refund policy?")
        v2 = embedder.embed("How to bake a chocolate cake?")
        similarity = np.dot(v1, v2)
        assert similarity < 0.5, f"Dissimilar queries should have low similarity, got {similarity}"
