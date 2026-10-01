"""Adapted PACE Evidence Frontloading for fixed hybrid candidate pools.

This adaptation uses the repository's BM25-derived lexical statistics and
per-query RRF scores on SciFact pools. It implements Evidence Frontloading
only; it does not reproduce SPLADE-v3, pressure budgeting, or soft anchors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


EPSILON = 1e-8
TIE_TOLERANCE = 1e-12


def bm25_document_contribution(tf: int, document_length: int, k1: float, b: float, avgdl: float) -> float:
    """Return BM25's document-side term factor, without query IDF."""
    if tf <= 0:
        return 0.0
    denominator = tf + k1 * (1.0 - b + b * document_length / avgdl)
    return tf * (k1 + 1.0) / denominator


def normalize_rrf_scores(scores: Sequence[float], epsilon: float = EPSILON) -> list[float]:
    """Min-max normalize RRF scores within one candidate pool."""
    if not scores:
        return []
    low, high = min(scores), max(scores)
    denominator = high - low + epsilon
    return [(score - low) / denominator for score in scores]


def evidence_coverage(
    selected: Sequence[Mapping[str, Any]], query_weights: Mapping[str, float]
) -> tuple[float, dict[str, float]]:
    """Calculate F(S) and each query term's maximum weighted coverage."""
    maxima = {term: 0.0 for term in query_weights}
    for candidate in selected:
        rho = float(candidate["rho"])
        terms = candidate["term_contributions"]
        for term in maxima:
            maxima[term] = max(maxima[term], rho * float(terms.get(term, 0.0)))
    return sum(query_weights[t] * maxima[t] for t in maxima), maxima


def marginal_contribution(
    candidate: Mapping[str, Any], current_maxima: Mapping[str, float], query_weights: Mapping[str, float]
) -> float:
    """Calculate the weighted increase in per-term maximum evidence."""
    rho = float(candidate["rho"])
    terms = candidate["term_contributions"]
    return sum(
        weight * max(rho * float(terms.get(term, 0.0)) - current_maxima.get(term, 0.0), 0.0)
        for term, weight in query_weights.items()
    )


def evidence_frontload(
    candidates: Sequence[Mapping[str, Any]], query_weights: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Greedily order all candidates by marginal evidence coverage.

    Inputs carry only lexical contributions and retrieval-time RRF information;
    no qrels, labels, oracle values, or evaluation metrics are accepted.
    Ties use original RRF rank and then string document ID.
    """
    ids = [str(c["id"]) for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    remaining = [dict(c) for c in candidates]
    remaining.sort(key=lambda c: (int(c["original_rank"]), str(c["id"])))
    maxima = {term: 0.0 for term in query_weights}
    ordered: list[dict[str, Any]] = []
    while remaining:
        best_index = 0
        best_value = marginal_contribution(remaining[0], maxima, query_weights)
        for index in range(1, len(remaining)):
            value = marginal_contribution(remaining[index], maxima, query_weights)
            if value > best_value + TIE_TOLERANCE:
                best_index, best_value = index, value
            elif abs(value - best_value) <= TIE_TOLERANCE:
                tie_key = (int(remaining[index]["original_rank"]), str(remaining[index]["id"]))
                best_key = (int(remaining[best_index]["original_rank"]), str(remaining[best_index]["id"]))
                if tie_key < best_key:
                    best_index, best_value = index, value
        chosen = remaining.pop(best_index)
        chosen["marginal_contribution"] = best_value
        ordered.append(chosen)
        rho = float(chosen["rho"])
        for term in maxima:
            maxima[term] = max(maxima[term], rho * float(chosen["term_contributions"].get(term, 0.0)))
    return ordered


def select_prefix(ordered: Sequence[Mapping[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Return the first K candidates from a complete evidence ordering."""
    if not 0 <= budget <= len(ordered):
        raise ValueError("budget must be between zero and candidate count")
    return [dict(candidate) for candidate in ordered[:budget]]

