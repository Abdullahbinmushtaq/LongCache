from evaluation.dataset import cases
from evaluation.run import check_gates, metrics, upper_bound


def test_versioned_dataset_counts_and_disjoint_families():
    source, rows, fingerprint = cases("evaluation/data/seeds.json")
    assert len(rows) == 4800
    assert len({r.id for r in rows}) == 4800
    assert len(fingerprint) == 64
    assert not set(source["development_actions"]) & {r.family for r in rows}
    assert sum(r.group == "safe" for r in rows) == 2000
    assert sum(r.group == "near" for r in rows) == 2000


def test_zero_hits_cannot_pass_false_hit_gate():
    result = metrics([])
    assert result["false_hit_rate"] is None
    assert check_gates(result, {"false_hit_rate_max": 0.005, "accepted_hits_min": 1000})
    assert upper_bound(0, 1000) < 0.01
    assert upper_bound(1, 1) == 1
