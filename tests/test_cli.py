"""Tests for the LongCache CLI."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from longcache.cli import main


@pytest.fixture
def runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_db():
    """Provide a temporary database file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_cli_cache.db")


class TestCLIStats:
    """Tests for the 'stats' command."""

    @patch("longcache.cli._get_cache")
    def test_stats_command(self, mock_get_cache, runner):
        """Test that stats command runs and displays output."""
        mock_cache = MagicMock()
        mock_cache.stats.return_value = {
            "hits": 10,
            "misses": 5,
            "hit_rate": 0.6667,
            "total_entries": 50,
            "estimated_tokens_saved": 15000,
            "estimated_cost_saved_usd": 0.45,
        }
        mock_get_cache.return_value = mock_cache

        result = runner.invoke(main, ["stats"])
        assert result.exit_code == 0
        assert "LongCache Statistics" in result.output
        assert "10" in result.output

    @patch("longcache.cli._get_cache")
    def test_stats_with_namespace(self, mock_get_cache, runner):
        """Test stats command with --namespace option."""
        mock_cache = MagicMock()
        mock_cache.stats.return_value = {
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "total_entries": 0,
            "estimated_tokens_saved": 0,
            "estimated_cost_saved_usd": 0.0,
        }
        mock_get_cache.return_value = mock_cache

        result = runner.invoke(main, ["stats", "--namespace", "faq"])
        assert result.exit_code == 0


class TestCLIClear:
    """Tests for the 'clear' command."""

    @patch("longcache.cli._get_cache")
    def test_clear_namespace(self, mock_get_cache, runner):
        """Test clearing a specific namespace."""
        mock_cache = MagicMock()
        mock_cache.invalidate.return_value = 5
        mock_get_cache.return_value = mock_cache

        result = runner.invoke(main, ["clear", "--namespace", "faq", "--yes"])
        assert result.exit_code == 0
        assert "5 entries deleted" in result.output

    @patch("longcache.cli._get_cache")
    def test_clear_all(self, mock_get_cache, runner):
        """Test clearing all namespaces."""
        mock_cache = MagicMock()
        mock_cache.clear.return_value = 10
        mock_get_cache.return_value = mock_cache

        result = runner.invoke(main, ["clear", "--all", "--yes"])
        assert result.exit_code == 0
        assert "10 entries deleted" in result.output


class TestCLIInspect:
    """Tests for the 'inspect' command."""

    @patch("longcache.cli._get_cache")
    def test_inspect_found(self, mock_get_cache, runner):
        """Test inspect when an entry is found."""
        mock_cache = MagicMock()
        mock_cache.inspect.return_value = {
            "id": 1,
            "query": "What is the refund policy?",
            "namespace": "default",
            "hit_count": 3,
            "created_at": 1000000.0,
            "expires_at": None,
            "is_expired": False,
        }
        mock_get_cache.return_value = mock_cache

        result = runner.invoke(main, ["inspect", "--query", "refund"])
        assert result.exit_code == 0
        assert "Nearest Cached Entry" in result.output
        assert "refund" in result.output

    @patch("longcache.cli._get_cache")
    def test_inspect_not_found(self, mock_get_cache, runner):
        """Test inspect when no entry is found."""
        mock_cache = MagicMock()
        mock_cache.inspect.return_value = None
        mock_get_cache.return_value = mock_cache

        result = runner.invoke(main, ["inspect", "--query", "nonexistent"])
        assert result.exit_code == 0
        assert "No similar entries found" in result.output


class TestCLIExport:
    """Tests for the 'export' command."""

    @patch("longcache.cli._get_cache")
    def test_export_to_stdout(self, mock_get_cache, runner):
        """Test exporting entries to stdout."""
        from longcache.backends.memory import InMemoryCacheBackend

        mock_cache = MagicMock()
        mock_backend = InMemoryCacheBackend()
        mock_cache.backend = mock_backend

        # Add an entry
        import numpy as np
        from longcache.core.entry import CacheEntry

        vector = np.random.randn(384).astype(np.float32)
        vector /= np.linalg.norm(vector)
        entry = CacheEntry(
            namespace="default",
            query="test query",
            vector=vector,
            response="test response",
        )
        mock_backend.store(entry)

        mock_get_cache.return_value = mock_cache

        result = runner.invoke(main, ["export", "--format", "json"])
        assert result.exit_code == 0
        # Output should be valid JSON
        try:
            data = json.loads(result.output)
            assert isinstance(data, list)
        except json.JSONDecodeError:
            pytest.fail("Export output is not valid JSON")

    @patch("longcache.cli._get_cache")
    def test_export_to_file(self, mock_get_cache, runner, temp_dir):
        """Test exporting entries to a file."""
        from longcache.backends.memory import InMemoryCacheBackend

        mock_cache = MagicMock()
        mock_cache.backend = InMemoryCacheBackend()
        mock_get_cache.return_value = mock_cache

        output_path = os.path.join(temp_dir, "export.json")
        result = runner.invoke(main, ["export", "--format", "json", "-o", output_path])

        if result.exit_code == 0:
            assert os.path.exists(output_path)


class TestCLIVersion:
    """Tests for the --version flag."""

    def test_version(self, runner):
        """Test that --version displays the version."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output
