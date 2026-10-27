# Memory Backend

The `memory` backend stores all vectors, entries, and metadata purely in Python process memory (RAM).

## Features

- **Maximum Performance**: Zero I/O latency. Lookups execute in microsecond order.
- **Zero Artifacts**: No files created on disk. Perfect for unit test suites and ephemeral testing containers.

## Configuration

```python
from longcache import LongCache, CacheConfig

config = CacheConfig(
    backend="memory",
    similarity_threshold=0.90,
    track_stats=True,
)

cache = LongCache(config)
```

## Recommended Use Cases

- Pytest fixtures and unit test isolation.
- Short-lived CLI commands or batch data processing scripts where persistence across runs is not required.
- Memory-constrained edge devices with read-only root filesystems.
