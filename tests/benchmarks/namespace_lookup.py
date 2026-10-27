"""Measure the worst-case namespace lookup path without downloading a model.

Run from the package directory with ``python tests/benchmarks/namespace_lookup.py``.
The target is less similar than every entry in another namespace, so lookup
must inspect the full ranked set before returning the correct target.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from longcache.config import CacheConfig  # noqa: E402
from longcache.core.entry import CacheEntry  # noqa: E402
from longcache.core.index import SimilarityIndex  # noqa: E402


class Backend:
    def __init__(self, entries: list[CacheEntry]) -> None:
        self.entries = {entry.id: entry for entry in entries}

    def retrieve(self, entry_id: int) -> CacheEntry | None:
        return self.entries.get(entry_id)


def measure(count: int, dimensions: int, runs: int) -> tuple[float, float]:
    query = np.zeros(dimensions, dtype=np.float32)
    query[0] = 1.0
    target = query.copy()
    target[0] = 0.9
    target[1] = np.sqrt(1.0 - 0.9**2)
    entries = [
        CacheEntry(
            id=i + 1,
            namespace=f"other-{i % 100}",
            query="other",
            response="other",
            vector=query.copy(),
        )
        for i in range(count)
    ]
    entries.append(
        CacheEntry(
            id=count + 1,
            namespace="target",
            query="target",
            response="target",
            vector=target,
        )
    )
    index = SimilarityIndex(Backend(entries), CacheConfig(backend="memory"))
    index.rebuild(entries)

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        result = index.find_similar(query, threshold=0.8, namespace="target", record_hit=False)
        samples.append((time.perf_counter() - start) * 1000)
        assert result is not None and result.id == count + 1
    return statistics.median(samples), max(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entries", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    if args.dimensions < 2 or args.runs < 1 or any(n < 1 for n in args.entries):
        parser.error("dimensions must be at least 2; runs and entries must be positive")

    faiss.omp_set_num_threads(1)
    for count in args.entries:
        median_ms, max_ms = measure(count, args.dimensions, args.runs)
        print(
            f"{count} distractors, {args.dimensions} dimensions, {args.runs} runs: "
            f"median {median_ms:.2f} ms, max {max_ms:.2f} ms"
        )


if __name__ == "__main__":
    main()
