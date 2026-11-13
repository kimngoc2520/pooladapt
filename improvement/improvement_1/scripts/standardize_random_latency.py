#!/usr/bin/env python3
"""Standardize Random latency percentiles from the frozen query-level handoff.

This script only reads the Improvement 1 handoff and the historical Phase 2
summary. It writes latency summaries beneath improvement/improvement_1 and
does not execute retrieval, reranking, or model inference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "improvement" / "improvement_1"
INPUT = OUT / "matched_budget" / "query_level_evaluation.csv"
SOURCE_SUMMARY = ROOT / "results" / "phase2" / "evaluation" / "final_gpu" / "phase2_summary.json"
OPS = ("B10", "B20", "B30", "B50")
SEEDS = ("0", "1", "2", "3", "4")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], p: float) -> float:
    """NumPy-compatible linear percentile, without requiring NumPy."""
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def main() -> None:
    rows = read_csv(INPUT)
    if len(rows) != 11_100:
        raise ValueError(f"Expected 11,100 handoff records; found {len(rows)}")

    query_ids = {row["query_id"] for row in rows}
    if len(query_ids) != 300:
        raise ValueError(f"Expected 300 unique TEST IDs; found {len(query_ids)}")
    random_rows = [row for row in rows if row["method"] == "Random"]
    if len(random_rows) != 6_000:
        raise ValueError(f"Expected 6,000 Random observations; found {len(random_rows)}")

    scope_values = {row["latency_measurement_scope"] for row in rows}
    if len(scope_values) != 1:
        raise ValueError(f"Latency scope varies across rows: {sorted(scope_values)}")
    scope = next(iter(scope_values))

    indexed: dict[tuple[str, str, str], float] = {}
    seed_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    query_seed_values: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in random_rows:
        op, query_id, seed = row["operating_point"], row["query_id"], row["seed"]
        key = (op, query_id, seed)
        if key in indexed:
            raise ValueError(f"Duplicate Random latency observation: {key}")
        if seed not in SEEDS:
            raise ValueError(f"Unexpected Random seed {seed!r}")
        value = float(row["reranking_latency_ms"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid Random latency for {key}: {value}")
        indexed[key] = value
        seed_values[(op, seed)].append(value)
        query_seed_values[(op, query_id)][seed] = value

    for op in OPS:
        expected = {(op, query_id, seed) for query_id in query_ids for seed in SEEDS}
        observed = {key for key in indexed if key[0] == op}
        if observed != expected:
            raise ValueError(
                f"{op}: expected complete 300 x 5 query-seed coverage; "
                f"missing={len(expected-observed)}, unexpected={len(observed-expected)}"
            )
        if len(seed_values[(op, SEEDS[0])]) != 300:
            raise ValueError(f"{op}: seed coverage is not 300 queries per seed")

    source = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))["final_test_results"]
    legacy = {
        (row["method"], f"B{int(row['target_budget'])}"): row
        for row in source
        if row["method"] == "Random"
    }
    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for op in OPS:
        per_seed = []
        for seed in SEEDS:
            values = seed_values[(op, seed)]
            result = {
                "method": "Random",
                "operating_point": op,
                "seed": int(seed),
                "n_queries": len(values),
                "p50_ms": percentile(values, 50),
                "p95_ms": percentile(values, 95),
                "scope": scope,
                "unit": "ms",
            }
            seed_rows.append(result)
            per_seed.append(result)

        query_means = [
            mean([query_seed_values[(op, query_id)][seed] for seed in SEEDS])
            for query_id in sorted(query_ids, key=int)
        ]
        pooled = [indexed[(op, query_id, seed)] for query_id in query_ids for seed in SEEDS]
        seed_p50s = [float(result["p50_ms"]) for result in per_seed]
        seed_p95s = [float(result["p95_ms"]) for result in per_seed]
        source_row = legacy[("Random", op)]
        old_p50 = float(source_row["p50_reranking_latency"]) * 1000.0
        old_p95 = float(source_row["p95_reranking_latency"]) * 1000.0
        summary_rows.append(
            {
                "method": "Random",
                "operating_point": op,
                "query_count": len(query_means),
                "expected_seeds": len(SEEDS),
                "observations_per_seed": 300,
                "pooled_query_seed_observations": len(pooled),
                "missing_query_seed_observations": 0,
                "primary_estimand": "P50/P95 across 300 query means; each mean averages seeds 0-4",
                "primary_p50_ms": percentile(query_means, 50),
                "primary_p95_ms": percentile(query_means, 95),
                "seed_p50_median_ms": percentile(seed_p50s, 50),
                "seed_p50_min_ms": min(seed_p50s),
                "seed_p50_max_ms": max(seed_p50s),
                "seed_p95_median_ms": percentile(seed_p95s, 50),
                "seed_p95_min_ms": min(seed_p95s),
                "seed_p95_max_ms": max(seed_p95s),
                "seed_p50_values_ms": ";".join(f"{value:.9f}" for value in seed_p50s),
                "seed_p95_values_ms": ";".join(f"{value:.9f}" for value in seed_p95s),
                "pooled_query_seed_p50_ms": percentile(pooled, 50),
                "pooled_query_seed_p95_ms": percentile(pooled, 95),
                "historical_mean_seed_p50_ms": old_p50,
                "historical_mean_seed_p95_ms": old_p95,
                "historical_minus_primary_p50_ms": old_p50 - percentile(query_means, 50),
                "historical_minus_primary_p95_ms": old_p95 - percentile(query_means, 95),
                "independent_query_population": 300,
                "scope": scope,
                "unit": "ms",
            }
        )

    write_csv(OUT / "latency_cost" / "random_latency_percentile_sensitivity.csv", summary_rows)
    write_csv(OUT / "latency_cost" / "random_latency_seed_percentiles.csv", seed_rows)

    # Keep the main latency table on the explicitly designated primary estimand.
    latency_path = OUT / "latency_cost" / "latency_cost_summary.csv"
    latency_rows = read_csv(latency_path)
    latency_by_op = {row["operating_point"]: row for row in summary_rows}
    for row in latency_rows:
        if row["method"] == "Random":
            primary = latency_by_op[row["operating_point"]]
            row["p50_ms"] = f"{primary['primary_p50_ms']:.12f}"
            row["p95_ms"] = f"{primary['primary_p95_ms']:.12f}"
    write_csv(latency_path, latency_rows)

    results_path = OUT / "tables" / "main_results_summary.csv"
    result_rows = read_csv(results_path)
    for row in result_rows:
        if row["method"] == "Random":
            primary = latency_by_op[row["operating_point"]]
            row["p50_reranking_latency_ms"] = f"{primary['primary_p50_ms']:.12f}"
            row["p95_reranking_latency_ms"] = f"{primary['primary_p95_ms']:.12f}"
    write_csv(results_path, result_rows)

    print(json.dumps({
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256(INPUT),
        "rows": len(rows),
        "unique_queries": len(query_ids),
        "random_rows": len(random_rows),
        "seeds": list(map(int, SEEDS)),
        "scope": scope,
        "summary": summary_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
