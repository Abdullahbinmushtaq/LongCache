"""Tests for the OpenAI direct integration (CachedOpenAI)."""

from unittest.mock import MagicMock, AsyncMock, patch

import numpy as np
import pytest

from longcache.config import CacheConfig


def _make_mock_embedder():
    embedder = MagicMock()
    embedder._model = MagicMock()
    embedder._dimension = 384

    def deterministic_embed(text: str):
        np.random.seed(hash(text) % (2**31))
        vector = np.random.randn(384).astype(np.float32)
        vector /= np.linalg.norm(vector)
        return vector

    embedder.embed = deterministic_embed
    embedder.dimension = 384
    return embedder


@patch("longcache.core.cache.LocalEmbedder")
class TestCachedOpenAI:
    def _cached_client(self, MockEmbedder, mock_openai_client):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        from longcache.integrations.openai import CachedOpenAI

        client = CachedOpenAI(
            mock_openai_client,
            config=CacheConfig(backend="memory", similarity_threshold=0.9),
        )
        client.cache.embedder = mock_embedder
        return client

    def test_create_caches_result(self, MockEmbedder):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = {"choices": ["fake completion"]}

        client = self._cached_client(MockEmbedder, mock_client)

        messages = [{"role": "user", "content": "What is RAG?"}]

        # CachedOpenAI.chat is a CachedChatCompletions wrapper — its own
        # .create() proxies through to the underlying client's
        # chat.completions.create(), so the cached call site is client.chat.create().
        result1 = client.chat.create(model="gpt-4o", messages=messages)
        assert result1 == {"choices": ["fake completion"]}
        assert mock_client.chat.completions.create.call_count == 1

        result2 = client.chat.create(model="gpt-4o", messages=messages)
        assert result2 == {"choices": ["fake completion"]}
        # Second call is a cache HIT — no additional API call
        assert mock_client.chat.completions.create.call_count == 1

    def test_extract_query_from_messages(self, MockEmbedder):
        client = self._cached_client(MockEmbedder, MagicMock())
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is the refund policy?"},
        ]
        assert client._extract_query(messages) == "What is the refund policy?"

    def test_extract_query_fallback_when_no_user_message(self, MockEmbedder):
        client = self._cached_client(MockEmbedder, MagicMock())
        messages = [{"role": "system", "content": "You are helpful."}]
        # Falls back to stringifying all messages
        result = client._extract_query(messages)
        assert "system" in result

    def test_proxy_attributes(self, MockEmbedder):
        mock_client = MagicMock()
        mock_client.some_attribute = "proxied_value"
        client = self._cached_client(MockEmbedder, mock_client)
        assert client.some_attribute == "proxied_value"

    async def test_acreate_caches_result(self, MockEmbedder):
        mock_client = MagicMock()
        mock_client.chat.completions.acreate = AsyncMock(
            return_value={"choices": ["fake async completion"]}
        )

        client = self._cached_client(MockEmbedder, mock_client)
        messages = [{"role": "user", "content": "What is async RAG?"}]

        result1 = await client.chat.acreate(model="gpt-4o", messages=messages)
        assert result1 == {"choices": ["fake async completion"]}
        assert mock_client.chat.completions.acreate.call_count == 1

        result2 = await client.chat.acreate(model="gpt-4o", messages=messages)
        assert result2 == {"choices": ["fake async completion"]}
        assert mock_client.chat.completions.acreate.call_count == 1
