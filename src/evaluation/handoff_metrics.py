"""Validation and cost definitions for the Improvement 1 TEST handoff."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

ExecutionKey = tuple[str, str, str, str]


def execution_key(row: Mapping[str, Any]) -> ExecutionKey:
    """Return a stable key that preserves genuine repeated-seed executions."""
    return (
        str(row["query_id"]),
        str(row["method"]),
        str(row["operating_point"]),
        "" if row.get("seed") is None else str(row["seed"]),
    )


def validate_unique_execution_keys(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject duplicate log rows without merging legitimate distinct seeds."""
    keys = [execution_key(row) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate execution keys: {duplicates[:5]}")


def population_coverage(
    expected_ids: Iterable[str], observed_ids: Iterable[str]
) -> tuple[set[str], set[str], set[str]]:
    """Return missing, unexpected, and duplicate query IDs without dropping rows."""
    expected = {str(query_id) for query_id in expected_ids}
    observed = [str(query_id) for query_id in observed_ids]
    counts = Counter(observed)
    actual = set(counts)
    return expected - actual, actual - expected, {
        query_id for query_id, count in counts.items() if count > 1
    }


def validate_qrels(
    qrels: Mapping[str, Mapping[str, float]], query_ids: Iterable[str]
) -> None:
    """Require valid finite, nonnegative relevance and positive labels per query."""
    for query_id in map(str, query_ids):
        judgments = qrels.get(query_id)
        if not judgments:
            raise ValueError(f"missing qrels for query {query_id}")
        for document_id, relevance in judgments.items():
            if not str(document_id) or not math.isfinite(float(relevance)) or relevance < 0:
                raise ValueError(f"invalid qrel for query {query_id}")
        if not any(float(relevance) > 0 for relevance in judgments.values()):
            raise ValueError(f"query {query_id} has no positive qrels")


def metrics_match(
    observed: Mapping[str, float], expected: Mapping[str, float],
    metrics: Sequence[str] = ("ndcg_at_10", "recall_at_10", "mrr_at_10"),
    tolerance: float = 1e-10,
) -> bool:
    """Compare query or aggregate metric mappings within an explicit tolerance."""
    return all(
        key in observed
        and key in expected
        and math.isfinite(float(observed[key]))
        and math.isfinite(float(expected[key]))
        and abs(float(observed[key]) - float(expected[key])) <= tolerance
        for key in metrics
    )


def average_k(selected_k: Iterable[int]) -> float:
    """Calculate mean controller-selected K over a non-empty population."""
    values = list(selected_k)
    _validate_pair_counts(values, "selected K")
    return sum(values) / len(values)


def average_actual_reranked_pairs(
    pair_counts: Iterable[int | None],
) -> float | None:
    """Calculate actual reranker pairs/query; return None when all are unavailable.

    Partial missing execution counts are rejected so the denominator cannot silently
    change. Values must come from measured reranker inputs, not selected K.
    """
    values = list(pair_counts)
    if not values:
        raise ValueError("pair counts must not be empty")
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("actual reranked-pair data is incomplete")
    counts = [value for value in values if value is not None]
    _validate_pair_counts(counts, "actual reranked pairs")
    return sum(counts) / len(counts)


def full_rerank_reference(
    pair_counts: Iterable[int], candidate_pool_size: int = 100,
) -> float:
    """Validate Full Rerank consumed the full pool and return its mean pair cost."""
    counts = list(pair_counts)
    _validate_pair_counts(counts, "Full Rerank pairs")
    if not counts or any(count != candidate_pool_size for count in counts):
        raise ValueError("Full Rerank reference must process the full candidate pool")
    return sum(counts) / len(counts)


def compression_ratio(average_method_pairs: float, average_full_pairs: float) -> float:
    """Return method pair consumption divided by the Full Rerank reference."""
    if not math.isfinite(average_method_pairs) or average_method_pairs < 0:
        raise ValueError("method average pairs must be finite and nonnegative")
    if not math.isfinite(average_full_pairs) or average_full_pairs <= 0:
        raise ValueError("Full Rerank average pairs must be finite and positive")
    return average_method_pairs / average_full_pairs


def cost_reduction(ratio: float) -> float:
    """Return relative reranking-pair reduction from a compression ratio."""
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("compression ratio must be finite and nonnegative")
    return 1.0 - ratio


def _validate_pair_counts(values: Sequence[int], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must not be empty")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{label} must contain nonnegative integers")
