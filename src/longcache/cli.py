"""
LongCache CLI — Command-line interface for cache management.

Provides commands for viewing stats, clearing cache, inspecting
entries, and exporting cache data.

Usage:
    longcache stats
    longcache clear --namespace default
    longcache clear --all
    longcache inspect --query "refund"
    longcache export --format json
"""

from __future__ import annotations

import json
from typing import Optional

import click

from longcache.config import CacheConfig
from longcache import __version__


def _get_cache(config: Optional[CacheConfig] = None):
    """Get a LongCache instance, lazily initialized."""
    from longcache import LongCache

    return LongCache(config or CacheConfig())


@click.group()
@click.version_option(version=__version__, prog_name="longcache")
def main():
    """
    LongCache — Semantic similarity cache for LLM/RAG pipelines.

    View stats, manage entries, and configure your cache from the command line.
    """
    pass


@main.command()
@click.option("--namespace", default=None, help="Namespace to show stats for.")
@click.option("--db-path", default=".longcache/cache.db", help="Path to cache database.")
def stats(namespace: Optional[str], db_path: str):
    """
    Show cache hit rate, estimated savings, and entry count.

    Displays key metrics about your cache performance including
    hit rate, estimated tokens saved, and estimated cost savings.

    Examples:

        longcache stats

        longcache stats --namespace faq
    """
    config = CacheConfig(db_path=db_path, namespace=namespace or "default")
    cache = _get_cache(config)

    s = cache.stats()

    click.echo("=" * 50)
    click.echo("  LongCache Statistics")
    click.echo("=" * 50)
    click.echo(f"  Namespace:          {config.namespace}")
    click.echo(f"  Cache Hits:         {s['hits']}")
    click.echo(f"  Cache Misses:       {s['misses']}")
    click.echo(f"  Hit Rate:           {s['hit_rate']:.1%}")
    click.echo(f"  Total Entries:      {s['total_entries']}")
    click.echo(f"  Est. Tokens Saved:  {s['estimated_tokens_saved']:,}")
    click.echo(f"  Est. Cost Saved:    ${s['estimated_cost_saved_usd']:.4f}")
    click.echo("=" * 50)


@main.command()
@click.option("--namespace", default=None, help="Namespace to clear.")
@click.option("--all", "clear_all", is_flag=True, help="Clear ALL namespaces.")
@click.option("--db-path", default=".longcache/cache.db", help="Path to cache database.")
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
def clear(namespace: Optional[str], clear_all: bool, db_path: str):
    """
    Clear cache entries by namespace or entirely.

    Use --namespace to clear a specific namespace, or --all
    to clear the entire cache across all namespaces.

    Examples:

        longcache clear --namespace faq

        longcache clear --all
    """
    config = CacheConfig(db_path=db_path)
    cache = _get_cache(config)

    if clear_all:
        count = cache.clear()
        click.echo(f"Cleared entire cache: {count} entries deleted.")
    elif namespace:
        count = cache.invalidate(namespace)
        click.echo(f"Cleared namespace '{namespace}': {count} entries deleted.")
    else:
        count = cache.invalidate()
        click.echo(f"Cleared namespace '{config.namespace}': {count} entries deleted.")


@main.command()
@click.option("--query", required=True, help="Query text to search for similar cached entries.")
@click.option("--namespace", default=None, help="Namespace to search in.")
@click.option("--db-path", default=".longcache/cache.db", help="Path to cache database.")
def inspect(query: str, namespace: Optional[str], db_path: str):
    """
    Inspect the nearest cached entry for a query.

    Shows the most similar cached entry regardless of the
    similarity threshold. Useful for debugging and understanding
    what would be cached for a given query.

    Examples:

        longcache inspect --query "What is the refund policy?"

        longcache inspect --query "refund" --namespace faq
    """
    config = CacheConfig(db_path=db_path, namespace=namespace or "default")
    cache = _get_cache(config)

    result = cache.inspect(query, namespace=namespace)

    if result is None:
        click.echo(f"No similar entries found for query: {query!r}")
        return

    click.echo("=" * 50)
    click.echo("  Nearest Cached Entry")
    click.echo("=" * 50)
    click.echo(f"  ID:            {result['id']}")
    click.echo(f"  Cached Query:  {result['query']!r}")
    click.echo(f"  Namespace:     {result['namespace']}")
    click.echo(f"  Hit Count:     {result['hit_count']}")
    click.echo(f"  Created At:    {result['created_at']}")
    click.echo(f"  Expires At:    {result['expires_at'] or 'Never'}")
    click.echo(f"  Is Expired:    {result['is_expired']}")
    click.echo("=" * 50)


@main.command()
@click.option(
    "--format",
    "export_format",
    default="json",
    type=click.Choice(["json"]),
    help="Export format (currently only JSON supported).",
)
@click.option("--namespace", default=None, help="Only export entries from this namespace.")
@click.option("--db-path", default=".longcache/cache.db", help="Path to cache database.")
@click.option("--output", "-o", default=None, help="Output file path. Defaults to stdout.")
def export(export_format: str, namespace: Optional[str], db_path: str, output: Optional[str]):
    """
    Export cache entries as JSON.

    Dumps all cache entries (or entries from a specific namespace)
    in JSON format. Useful for backup, migration, or analysis.

    Examples:

        longcache export --format json

        longcache export --namespace faq -o faq_cache.json
    """
    config = CacheConfig(db_path=db_path, namespace=namespace or "default")
    cache = _get_cache(config)

    # Access the backend directly for export
    from longcache.backends.sqlite import SQLiteCacheBackend

    backend = cache.backend

    if isinstance(backend, SQLiteCacheBackend):
        entries = backend.export_all()
    else:
        # Generic fallback
        ns = namespace or config.namespace
        entries_raw = backend.list_entries(ns)
        entries = [
            {
                "id": e.id,
                "namespace": e.namespace,
                "query": e.query,
                "response": str(e.response),
                "created_at": e.created_at,
                "expires_at": e.expires_at,
                "hit_count": e.hit_count,
            }
            for e in entries_raw
        ]

    # Filter by namespace if specified
    if namespace:
        entries = [e for e in entries if e.get("namespace") == namespace]

    json_output = json.dumps(entries, indent=2, default=str)

    if output:
        with open(output, "w") as f:
            f.write(json_output)
        click.echo(f"Exported {len(entries)} entries to {output}")
    else:
        click.echo(json_output)


if __name__ == "__main__":
    main()
