"""Final Phase 2 calibration and reporting, independent of model training.

Calibration accepts a validation-only logit matrix. It never accepts quality
scores: operating points are chosen solely by distance to the target workload.
"""
from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from src.pooladapt.selector import BUDGET_GRID, budget_bias

TARGETS = (10, 20, 30, 50)
SEEDS = (0, 1, 2, 3, 4)
METRICS = ("nDCG@10", "Recall@10", "MRR@10")
CALIBRATION_FIELDS = (
    "method", "target_budget", "actual_average_budget", "relative_budget_error",
    "calibration_status", "temperature", "budget_bias_lambda",
    "other_calibration_parameter", "validation_nDCG@10", "locked",
)
RESULT_FIELDS = (
    "method", "target_budget", "seed", "budget_type", "actual_average_budget",
    *METRICS, "avg_reranked_pairs", "total_reranked_pairs", "compression_ratio",
    "mean_reranking_latency", "p50_reranking_latency", "p95_reranking_latency",
    "latency_reduction", "calibration_status", "matched_budget",
    "validation_average_budget", *(f"{m}_seed_std" for m in METRICS),
)


def sha256(path: Path) -> str:
    """Hash a file for immutable input and Phase 1 provenance checks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    """Write a declared schema, including headers for empty result sets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_population(ids: Sequence[str], expected: Sequence[str], forbidden: Sequence[str] = ()) -> None:
    """Reject missing, duplicate, extra, or forbidden inference/calibration IDs."""
    if not ids or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("query population must exactly match the locked split")
    if set(ids) & set(forbidden):
        raise ValueError("calibration population contains forbidden TRAIN/TEST IDs")


def predict_budgets(logits: np.ndarray, temperature: float, bias_lambda: float) -> np.ndarray:
    """Apply the existing hard argmax with smaller-budget tie breaking."""
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(BUDGET_GRID) or not len(values) or not np.isfinite(values).all():
        raise ValueError("expected finite query-by-five logits")
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(bias_lambda):
        raise ValueError("invalid calibration parameters")
    return np.asarray(BUDGET_GRID)[np.argmax(values / temperature + bias_lambda * budget_bias(), axis=1)]


def lambda_sweep(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Enumerate all hard-policy regions on the entire real lambda line.

    Pairwise logit-line crossings partition the line into constant predictions.
    Include boundaries and interior representatives, plus both exterior regions.
    A T x lambda fallback cannot add a policy: positive T makes the argmax
    depend only on T*lambda. Thus this exhaustive 1-D sweep covers it already.
    """
    values = np.asarray(logits) / temperature
    bias = budget_bias()
    crossings = np.unique(np.concatenate([
        (values[:, j] - values[:, i]) / (bias[i] - bias[j])
        for i in range(5) for j in range(i + 1, 5)
    ]))
    interior = crossings[:-1] + np.diff(crossings) / 2
    return np.unique(np.concatenate((crossings, interior, [0., crossings[0] - 1., crossings[-1] + 1.])))


def calibrate(method: str, ids: Sequence[str], validation_ids: Sequence[str],
              forbidden_ids: Sequence[str], logits: np.ndarray,
              temperature: float = 1.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select closest achievable average K using validation workload alone."""
    verify_population(ids, validation_ids, forbidden_ids)
    if len(logits) != len(ids):
        raise ValueError("logit rows must align with validation IDs")
    sanity = {str(lam): float(predict_budgets(logits, temperature, lam).mean()) for lam in (-5., 0., 5.)}
    responsive = len(set(sanity.values())) > 1
    sweep = lambda_sweep(logits, temperature) if responsive else np.asarray([0.])
    averages = np.asarray([predict_budgets(logits, temperature, lam).mean() for lam in sweep])
    rows = []
    for target in TARGETS:
        # Predeclared workload-only tie break: closest to default lambda, then lambda.
        best = min(range(len(sweep)), key=lambda i: (abs(averages[i] - target), abs(sweep[i]), sweep[i]))
        average, lam = float(averages[best]), float(sweep[best])
        error = abs(average - target) / target
        rows.append(dict(method=method, target_budget=target, actual_average_budget=average,
                         relative_budget_error=error, calibration_status="SUCCESS" if responsive and error <= .05 else "FAILED",
                         temperature=temperature, budget_bias_lambda=lam, other_calibration_parameter=None,
                         **{"validation_nDCG@10": None}, locked=True))
    return rows, {"lambda_sanity_average_K": sanity, "responsive": responsive,
                  "search": "exhaustive 1-D hard-policy regions at frozen T",
                  "two_dimensional_search": False,
                  "temperature_note": "argmax depends only on T*lambda; a 2-D search adds no policy",
                  "candidate_configurations": len(sweep)}


def heuristic_gates(config: Mapping[str, Any]) -> dict[str, str]:
    """Preserve the frozen branch stop; never construct a post-hoc heuristic."""
    if config.get("g2_status") == "FAILED" and config.get("g3_status") == "NOT_EVALUATED":
        return {"G2": "FAILED", "G3a": "NOT_EVALUATED", "G3b": "NOT_EVALUATED",
                "explanation": "The single-feature heuristic branch was stopped by the predefined G2 gate before performance evaluation."}
    raise ValueError("Frozen heuristic state differs from expected stopped branch; its existing policy needs explicit protocol inspection")


def aggregate(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], full_latency: float) -> dict[str, Any]:
    """Aggregate same-population query metrics and measured CE latency in seconds."""
    ids = [str(row["query_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("aggregation requires unique queries")
    latency = np.asarray([row["reranking_latency"] for row in rows], dtype=float)
    if not np.isfinite(latency).all() or (latency < 0).any():
        raise ValueError("latency must be finite and nonnegative")
    budgets = np.asarray([row["reranked_pairs"] for row in rows], dtype=float)
    average = float(budgets.mean())
    target = config["target_budget"]
    return {**{key: config.get(key) for key in RESULT_FIELDS},
            **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
            "actual_average_budget": average, "avg_reranked_pairs": average,
            "total_reranked_pairs": int(budgets.sum()), "compression_ratio": 1 - average / 100,
            "mean_reranking_latency": float(latency.mean()),
            "p50_reranking_latency": float(np.percentile(latency, 50)),
            "p95_reranking_latency": float(np.percentile(latency, 95)),
            "latency_reduction": 1 - float(latency.mean()) / full_latency if full_latency > 0 else None,
            "matched_budget": abs(average - target) / target <= .05,
            "validation_average_budget": config["actual_average_budget"]}


def random_aggregate(seed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report means and sample SD across five seeds, never query-level SD."""
    if {row["seed"] for row in seed_rows} != set(SEEDS) or len(seed_rows) != 5:
        raise ValueError("Random requires exactly seeds 0 through 4")
    result = dict(seed_rows[0], seed=None)
    for key in (*METRICS, "total_reranked_pairs", "mean_reranking_latency", "p50_reranking_latency", "p95_reranking_latency", "latency_reduction"):
        result[key] = float(np.mean([row[key] for row in seed_rows]))
    for metric in METRICS:
        result[f"{metric}_seed_std"] = float(np.std([row[metric] for row in seed_rows], ddof=1))
    return result


def tradeoff_findings(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe measured quality/workload/latency, without superiority inference."""
    index = {(r['method'], r['target_budget']): r for r in results if r.get('nDCG@10') is not None}
    full = index[('Full Rerank', 100)]
    points = []
    for key, row in index.items():
        points.append({'method': key[0], 'target_budget': key[1],
                       'actual_average_budget': row['actual_average_budget'],
                       'matched_budget': row['matched_budget'],
                       'quality_delta_vs_full': {m: row[m] - full[m] for m in METRICS},
                       'latency_reduction_vs_full': row['latency_reduction']})
    comparisons = []
    for target in TARGETS:
        pairs = [('PACE-EF', 'Fixed Prefix'), ('PoolAdapt', 'Fixed Prefix'),
                 ('SAGE-SLO', 'Fixed Prefix'), ('PoolAdapt', 'SAGE-SLO'), ('PoolAdapt', 'PACE-EF')]
        for left, right in pairs:
            a, b = index.get((left, target)), index.get((right, target))
            if a and b and a['matched_budget'] and b['matched_budget']:
                comparisons.append({'method_a': left, 'method_b': right, 'target_budget': target,
                                    'quality_deltas_a_minus_b': {m: a[m] - b[m] for m in METRICS},
                                    'average_pairs_delta_a_minus_b': a['avg_reranked_pairs'] - b['avg_reranked_pairs'],
                                    'latency_delta_a_minus_b': a['mean_reranking_latency'] - b['mean_reranking_latency']})
        pooladapt = index.get(('PoolAdapt', target))
        baselines = [row for (name, budget), row in index.items() if budget == target and name in ('Fixed Prefix', 'PACE-EF', 'SAGE-SLO', 'Random') and row['matched_budget']]
        if pooladapt and pooladapt['matched_budget'] and baselines:
            strongest = max(baselines, key=lambda r: r['nDCG@10'])
            comparisons.append({'method_a': 'PoolAdapt', 'method_b': strongest['method'], 'target_budget': target,
                                'description': 'Descriptive comparison with highest observed baseline nDCG at this matched target; no model or lock selection',
                                'nDCG_delta_a_minus_b': pooladapt['nDCG@10'] - strongest['nDCG@10']})
    return {'title': 'Quality–Cost–Latency Trade-off Analysis', 'points_vs_full': points,
            'matched_comparisons': comparisons,
            'interpretation': 'Descriptive empirical deltas; corrected paired tests are reported separately. No equivalence or Pareto-frontier claim.'}


def paired_statistics(groups: Mapping[tuple[str, int, int | None], Sequence[Mapping[str, Any]]],
                      matched: set[tuple[str, int]], ids: Sequence[str]) -> list[dict[str, Any]]:
    """Predeclared paired two-sided Wilcoxon comparisons, one global Holm family.

    Random comparison uses each query's mean metric over the five fixed seeds.
    Seed-level results and SD remain separately reported. Unmatched TEST points
    are excluded from budget-matched hypotheses without changing their locks.
    """
    vectors = {}
    for key, rows in groups.items():
        verify_population([str(r["query_id"]) for r in rows], ids)
        indexed = {str(r["query_id"]): r for r in rows}
        vectors[key] = {m: np.asarray([indexed[q][m] for q in ids]) for m in METRICS}
    for target in TARGETS:
        vectors[("Random", target, None)] = {m: np.mean([vectors[("Random", target, seed)][m] for seed in SEEDS], axis=0) for m in METRICS}
    pairs = []
    for target in TARGETS:
        names = [name for name in ("Fixed Prefix", "PACE-EF", "PoolAdapt", "SAGE-SLO", "Random") if (name, target) in matched]
        pairs.extend(((name, target, None), ("Full Rerank", 100, None)) for name in names)
        pairs.extend((a, b) for a, b in [
            (("PACE-EF", target, None), ("Fixed Prefix", target, None)),
            *((((name, target, None), (other, target, None))) for name in ("PoolAdapt", "SAGE-SLO") for other in ("Fixed Prefix", "PACE-EF", "Random")),
            (("PoolAdapt", target, None), ("SAGE-SLO", target, None)),
        ] if a[0] in names and b[0] in names)
    results = []
    for left, right in pairs:
        for metric in METRICS:
            x, y = vectors[left][metric], vectors[right][metric]
            nonzero = int(np.count_nonzero(x - y))
            test = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox") if nonzero else None
            results.append({"method_a": left[0], "budget_a": left[1], "method_b": right[0], "budget_b": right[1],
                            "metric": metric, "test": "Wilcoxon signed-rank; two-sided; zero differences excluded",
                            "paired": True, "query_count": len(ids), "nonzero_pair_count": nonzero,
                            "random_construction": "query mean over 5 seeds" if "Random" in (left[0], right[0]) else None,
                            "statistic": float(test.statistic) if test else 0., "p_value": float(test.pvalue) if test else 1.,
                            "mean_difference_a_minus_b": float(np.mean(x - y)),
                            "median_difference_a_minus_b": float(np.median(x - y)),
                            "correction": "Holm; global family across all predeclared metrics and comparisons"})
    running = 0.
    for rank, index in enumerate(sorted(range(len(results)), key=lambda i: results[i]["p_value"])):
        running = max(running, min(1., results[index]["p_value"] * (len(results) - rank)))
        results[index]["p_Holm"] = running
        results[index]["interpretation"] = "statistically significant difference detected" if running < .05 else "no statistically significant difference detected"
    return results
