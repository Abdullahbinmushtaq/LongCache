# SQLite Backend

The SQLite backend is the default storage engine for LongCache. It requires **zero external services** and stores entries in a single, robust disk database.

## Features

- **Zero Setup**: The database and necessary schema tables are created automatically on first run.
- **WAL Mode**: Uses SQLite's Write-Ahead Logging (`PRAGMA journal_mode=WAL`) for concurrent reading without blocking writes.
- **Thread-Safe**: Safely usable across multithreaded Python applications.

## Configuration

```python
from longcache import LongCache, CacheConfig

config = CacheConfig(
    backend="sqlite",
    db_path="data/my_rag_cache.db",
    max_entries=20000,
    ttl_seconds=86400,
)

cache = LongCache(config)
```

## Database Schema

LongCache automatically manages the following table structure in SQLite:

- `entries`: Primary table containing entry IDs, namespace, raw query string, response payload, creation timestamp, and composite scope hash.
- `metadata`: Key-value store tracking total hits, misses, and runtime metrics.
