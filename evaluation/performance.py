"""Matched warm-hit benchmarks at 1K/10K entries, including real embedding inference."""

from __future__ import annotations
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import longcache
from longcache.core.embedder import LocalEmbedder

QUERY = "How do I reset my password?"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def measure(legacy=False, samples=50):
    import torch
    import faiss

    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    model = LocalEmbedder() if legacy else LocalEmbedder(revision=REVISION)
    model.embed(QUERY)

    class Hybrid(getattr(longcache, "BaseEmbedder", object)):
        dimension = 384
        if not legacy:
            descriptor = model.descriptor

        def embed(self, text):
            if text == QUERY:
                return model.embed(text)
            rng = np.random.default_rng(int.from_bytes(sha256(text.encode()).digest()[:8], "big"))
            value = rng.normal(size=384).astype(np.float32)
            return value / np.linalg.norm(value)

    embedder = Hybrid()
    result = []
    for kind in ("flat", "hnsw"):
        for size in (1000, 10000):
            cfg = longcache.CacheConfig(
                backend="memory", ttl_seconds=0, max_entries=20000, faiss_index_type=kind
            )
            if legacy:
                cache = longcache.LongCache(cfg)
                cache.embedder = embedder
                options = {}
            else:
                ctx = longcache.RequestContext(
                    scope=longcache.RequestScope("benchmark", "user"),
                    operation="benchmark.hit",
                    provider="fixture",
                    provider_scope="benchmark-v1",
                    model="fixture",
                    prompt_version="v1",
                )
                cache = longcache.LongCache(cfg, embedder=embedder)
                options = {"context": ctx}
            try:
                cache.set(QUERY, "answer", **options)
                for i in range(size - 1):
                    cache.set(f"noise {i}", "distractor", **options)
                for _ in range(5):
                    assert cache.get(QUERY, **options) == "answer"
                durations = []
                for _ in range(samples):
                    start = time.perf_counter()
                    assert cache.get(QUERY, **options) == "answer"
                    durations.append((time.perf_counter() - start) * 1000)
                result.append(
                    {
                        "index": kind,
                        "entries": size,
                        "samples": samples,
                        "p50_ms": float(np.percentile(durations, 50)),
                        "p95_ms": float(np.percentile(durations, 95)),
                        "p99_ms": float(np.percentile(durations, 99)),
                    }
                )
            finally:
                cache.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/warm_lookup.json"))
    args = parser.parse_args()
    result = measure(legacy=args.legacy)
    if args.legacy:
        print(json.dumps(result))
        return 0
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_threads": 1,
            "faiss_threads": 1,
        },
        "model_revision": REVISION,
        "v02": result,
        "limits": "Synthetic random distractor vectors measure index size overhead; warm queries include real pinned BGE inference. No miss/concurrent-maintenance throughput claim.",
        "comparison_passed": None,
    }
    if args.baseline_root:
        env = os.environ | {"PYTHONPATH": str(args.baseline_root.resolve() / "src")}
        legacy = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--legacy"],
            cwd=args.baseline_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        old = json.loads(legacy.stdout)
        report["v011"] = old
        comparisons = []
        for current, previous in zip(result, old):
            budget = 1.2 * previous["p95_ms"] + 5
            comparisons.append(
                {
                    "entries": current["entries"],
                    "index": current["index"],
                    "budget_p95_ms": budget,
                    "actual_p95_ms": current["p95_ms"],
                    "passed": current["p95_ms"] <= budget,
                }
            )
        report["comparisons"] = comparisons
        report["comparison_passed"] = all(row["passed"] for row in comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["comparison_passed"] is not True)


if __name__ == "__main__":
    raise SystemExit(main())
