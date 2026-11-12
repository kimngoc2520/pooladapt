"""Targeted tests for the frozen handoff's downstream analysis rules."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "improvement" / "improvement_1" / "scripts"))
import run_all as analysis  # noqa: E402


def test_random_seed_aggregation_keeps_300_query_population() -> None:
    rows = analysis.read_rows(analysis.INPUT)
    aggregated = analysis.aggregate(rows)
    random_b10 = [r for r in aggregated if r["method"] == "Random" and r["operating_point"] == "B10"]
    assert len(random_b10) == 300
    assert all(r["seed_count"] == 5 for r in random_b10)
    assert all(r["seed_policy"] == "mean within query" for r in random_b10)


def test_phase0_validation_covers_qrels_and_all_strata() -> None:
    result = analysis.validate(analysis.read_rows(analysis.INPUT))
    assert result["errors"] == []
    assert (result["rows"], result["queries"], result["strata"]) == (11100, 300, 37)
    assert result["positive_qrels_queries"] == 300
    assert result["positive_qrels_judgments"] == 339


def test_paired_statistics_are_query_paired_and_holm_adjusted() -> None:
    aggregated = analysis.aggregate(analysis.read_rows(analysis.INPUT))
    detailed, summary = analysis.paired_stats(aggregated)
    assert len(detailed) == len(summary) == 9
    assert all(row["n_queries"] == 300 for row in detailed)
    assert all("permutation_p_holm_adjusted" in row for row in detailed)
    assert all(0 <= row["permutation_p_holm_adjusted"] <= 1 for row in detailed)


def test_pareto_keeps_exact_ties_and_includes_full_rerank() -> None:
    points = [
        {"method": "Full Rerank", "operating_point": "B100", "average_actual_pairs": 100.0,
         "mean_reranking_latency_ms": 250.0, "ndcg_at_10": 0.8},
        {"method": "PoolAdapt", "operating_point": "B10", "average_actual_pairs": 10.0,
         "mean_reranking_latency_ms": 60.0, "ndcg_at_10": 0.7},
        {"method": "Fixed Prefix", "operating_point": "B10", "average_actual_pairs": 10.0,
         "mean_reranking_latency_ms": 60.0, "ndcg_at_10": 0.7},
    ]
    result = analysis.pareto(points)
    pair_front = [r for r in result if r["pareto_objective"] == "quality_vs_actual_pairs"]
    assert len(pair_front) == 3
    assert all(r["pareto_efficient"] for r in pair_front)
    assert {r["method"] for r in pair_front} == {"Full Rerank", "PoolAdapt", "Fixed Prefix"}


def test_primary_budget_rules_exclude_b30() -> None:
    rows = analysis.read_rows(analysis.INPUT)
    aggregate = analysis.aggregate(rows)
    summary = analysis.summarize(aggregate)
    pairs = [r for r in summary if r["method"] in ("PoolAdapt", "Fixed Prefix")]
    for budget in (10, 20, 50):
        pa = next(r for r in pairs if r["method"] == "PoolAdapt" and r["operating_point"] == f"B{budget}")
        fx = next(r for r in pairs if r["method"] == "Fixed Prefix" and r["operating_point"] == f"B{budget}")
        assert abs(100 * (pa["average_actual_pairs"] - fx["average_actual_pairs"]) / fx["average_actual_pairs"]) <= 5
    pa30 = next(r for r in pairs if r["method"] == "PoolAdapt" and r["operating_point"] == "B30")
    fx30 = next(r for r in pairs if r["method"] == "Fixed Prefix" and r["operating_point"] == "B30")
    assert abs(100 * (pa30["average_actual_pairs"] - fx30["average_actual_pairs"]) / fx30["average_actual_pairs"]) > 5
