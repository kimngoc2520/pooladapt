"""SAGE-inspired query-level adaptive reranking-budget policy.

This adapts the policy shape of Raza et al. (2026), *SAGE: SLO-Aware
Adaptive Retrieval for Production RAG Systems* (arXiv:2608.08237), not its
end-to-end EM-and-latency-SLO experiment.  Inference accepts only existing,
pre-reranking pool/query features.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

BUDGET_GRID: tuple[int, ...] = (10, 20, 30, 50, 100)
FORBIDDEN_FEATURE_NAMES = frozenset(("qrels", "relevance", "m_star", "oracle", "reranker", "cross_encoder", "reranked", "post_reranking", "final_score"))


def budget_bias(budgets: Sequence[int] = BUDGET_GRID) -> np.ndarray:
    """Return g(k)=-(k-mean(G))/std(G), using the population standard deviation."""
    values = np.asarray(budgets, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.std(values) == 0:
        raise ValueError("budgets must contain at least two distinct values")
    return -(values - values.mean()) / values.std()


def calibrated_logits(logits: Sequence[float], temperature: float, budget_bias_lambda: float) -> np.ndarray:
    """Apply z'_k(q)=z_k(q)/T + lambda*g(k)."""
    values = np.asarray(logits, dtype=float)
    if values.shape != (len(BUDGET_GRID),):
        raise ValueError(f"logits must have one value for each budget: {BUDGET_GRID}")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    if not np.isfinite(budget_bias_lambda):
        raise ValueError("budget_bias_lambda must be finite")
    return values / temperature + budget_bias_lambda * budget_bias()


def softmax(scores: Sequence[float]) -> np.ndarray:
    """Numerically stable softmax, for reporting p(k|q)."""
    values = np.asarray(scores, dtype=float)
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def select_budget(logits: Sequence[float], temperature: float = 1.0, budget_bias_lambda: float = 0.0) -> int:
    """Select a legal budget; np.argmax breaks ties toward smaller K."""
    return BUDGET_GRID[int(np.argmax(calibrated_logits(logits, temperature, budget_bias_lambda)))]


def select_sage_slo(candidates: Sequence[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Select the RRF prefix prescribed by an already predicted adaptive budget."""
    if budget not in BUDGET_GRID:
        raise ValueError(f"budget must be one of {BUDGET_GRID}")
    return list(candidates[:budget])


def validate_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Reject labels, IDs, and post-reranking fields from policy inference input."""
    clean = tuple(feature_names)
    if not clean or len(clean) != len(set(clean)):
        raise ValueError("feature names must be non-empty and unique")
    forbidden = [name for name in clean if any(token in name.lower() for token in FORBIDDEN_FEATURE_NAMES)]
    if forbidden or "query_id" in clean or "doc_id" in clean:
        raise ValueError(f"forbidden inference feature(s): {forbidden}")
    return clean


def model_logits(model: Any, feature_row: Mapping[str, float], feature_names: Sequence[str]) -> np.ndarray:
    """Define z_k(q) as log clipped class probabilities in BUDGET_GRID order."""
    names = validate_feature_names(feature_names)
    values = np.asarray([[float(feature_row[name]) for name in names]], dtype=float)
    probabilities = model.predict_proba(values)[0]
    by_class = {int(label): float(probability) for label, probability in zip(model.classes_, probabilities)}
    return np.log(np.asarray([max(by_class.get(budget, 0.0), 1e-12) for budget in BUDGET_GRID]))
