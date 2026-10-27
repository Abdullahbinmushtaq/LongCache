# Evaluation and benchmarks

## Synthetic quality protocol

```bash
uv run --no-sync python -m evaluation.run --config evaluation/configs/v0_2_release.json
```

The pinned-model 4,800-case synthetic suite checks safe paraphrases, adversarial near misses, scope/context isolation, unsupported requests, and freshness. It reports accepted/incorrect hits, coverage, bypass reasons, confidence bounds, and scenario latency. The dataset and predeclared acceptance gates are versioned under `evaluation/`.

The initial run accepted 2,000 safe hits with no incorrect hits or isolation errors and bypassed all declared unsupported cases. These are correlated template-expanded fixtures, not a representative production error estimate. The JSON report includes family counts, model/dependency versions, dataset/config/source hashes, and limitations. Scenario timing excludes model inference through precomputed real-model vectors.

## Warm-hit end-to-end comparison

```bash
uv run --no-sync python -m evaluation.performance --baseline-root /path/to/v011/archive
```

The matched 1K/10K-entry flat/HNSW benchmark includes actual cached-model inference, uses one Torch/FAISS thread, and records 50 warm samples per scenario. The budget is `v0.2 p95 <= 1.2 × v0.1.1 p95 + 5 ms`. The initial local run passed all four comparisons, with v0.2 p95 approximately 28–34 ms. This is observed on the recorded hardware, not a general latency guarantee.

Synthetic random distractors isolate index-size overhead. Miss latency, concurrent maintenance, distributed operation, and cost savings are not established by this benchmark. See the [execution record](plans/v0.2.0-execution.md) and repository `evaluation/reports/` for evidence and limits.
