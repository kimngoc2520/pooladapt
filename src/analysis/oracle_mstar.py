"""Compute oracle-derived query-specific reranking budgets from rerank outputs.

This module deliberately depends only on qrels and ranked-document outputs.  It
does not import, read, or otherwise depend on candidate/pool/query features.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.retrieval_metrics import ndcg_at_k


BUDGETS = (10, 20, 30, 50, 100)
TAUS = (0.95, 0.98, 0.99)
CSV_FIELDS = (
    "query_id", "M_star", "tau", "M_star_is_max", "M_star_is_min",
    "nDCG_at_M_star", "nDCG_full",
)


def prediction_index(predictions: Sequence[Mapping[str, Any]], label: str) -> dict[str, list[str]]:
    """Validate one method's predictions and index its final ranked IDs."""
    indexed: dict[str, list[str]] = {}
    for prediction in predictions:
        if "query_id" not in prediction or "document_ids" not in prediction:
            raise ValueError(f"{label} prediction is missing query_id or document_ids")
        query_id = str(prediction["query_id"])
        if query_id in indexed:
            raise ValueError(f"{label} contains duplicate query ID {query_id!r}")
        document_ids = prediction["document_ids"]
        if not isinstance(document_ids, list):
            raise ValueError(f"{label}, query {query_id!r}: document_ids must be a list")
        indexed[query_id] = [str(document_id) for document_id in document_ids]
    return indexed


def extract_rankings(result: Mapping[str, Any]) -> tuple[dict[int, dict[str, list[str]]], dict[str, list[str]]]:
    """Extract fixed-prefix budgets and the independently recorded Full Rerank."""
    methods = result.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("reranking result must contain a methods mapping")
    budget_rankings: dict[int, dict[str, list[str]]] = {}
    for budget in BUDGETS:
        method_name = f"fixed_prefix_{budget}"
        method = methods.get(method_name)
        if not isinstance(method, Mapping) or not isinstance(method.get("predictions"), list):
            raise ValueError(f"reranking result must include {method_name!r} predictions")
        budget_rankings[budget] = prediction_index(method["predictions"], method_name)
    full = methods.get("full_rerank")
    if not isinstance(full, Mapping) or not isinstance(full.get("predictions"), list):
        raise ValueError("reranking result must include 'full_rerank' predictions")
    return budget_rankings, prediction_index(full["predictions"], "full_rerank")


def compute_mstar_labels(
    qrels: Mapping[str, Mapping[str, int]],
    budget_rankings: Mapping[int, Mapping[str, Sequence[str]]],
    full_rankings: Mapping[str, Sequence[str]],
    tau: float,
) -> list[dict[str, Any]]:
    """Return deterministic M* labels for one retention threshold."""
    if tau not in TAUS:
        raise ValueError(f"tau must be one of {TAUS}; got {tau}")
    expected_ids = set(map(str, qrels))
    if not expected_ids:
        raise ValueError("qrels must not be empty")
    if set(map(str, full_rankings)) != expected_ids:
        raise ValueError("full_rerank query IDs must exactly match qrels")
    for budget in BUDGETS:
        if budget not in budget_rankings:
            raise ValueError(f"missing rankings for M={budget}")
        if set(map(str, budget_rankings[budget])) != expected_ids:
            raise ValueError(f"M={budget} query IDs must exactly match qrels")

    rows: list[dict[str, Any]] = []
    for query_id in sorted(expected_ids):
        relevance = qrels[query_id]
        full_ndcg = ndcg_at_k(full_rankings[query_id], relevance, k=10)
        selected = 100
        selected_ndcg = ndcg_at_k(budget_rankings[100][query_id], relevance, k=10)
        for budget in BUDGETS:
            score = ndcg_at_k(budget_rankings[budget][query_id], relevance, k=10)
            if score >= tau * full_ndcg:
                selected, selected_ndcg = budget, score
                break
        rows.append({
            "query_id": query_id, "M_star": selected, "tau": tau,
            "M_star_is_max": int(selected == 100), "M_star_is_min": int(selected == 10),
            "nDCG_at_M_star": selected_ndcg, "nDCG_full": full_ndcg,
        })
    return rows


def analyse_mstar(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create the predefined distribution and G1 screening summary."""
    if not rows:
        raise ValueError("cannot analyse empty M* labels")
    values = [int(row["M_star"]) for row in rows]
    counts = {str(budget): values.count(budget) for budget in BUDGETS}
    percentages = {str(budget): counts[str(budget)] * 100.0 / len(values) for budget in BUDGETS}
    distinct = len(set(values))
    std = statistics.pstdev(values)
    return {
        "query_count": len(values), "mstar_counts": counts, "mstar_percentages": percentages,
        "mean_M_star": statistics.mean(values), "std_M_star": std,
        "min_M_star": min(values), "max_M_star": max(values),
        "number_of_distinct_M_star_levels": distinct,
        "G1_passes": distinct >= 3 and std > 0,
        "G1_criterion": "number_of_distinct_M_star_levels >= 3 AND std_M_star > 0",
    }


def write_labels(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(analysis: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, sort_keys=True)
        handle.write("\n")
