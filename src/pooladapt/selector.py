"""PoolAdapt's query/pool-level adaptive-K selector."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

BUDGET_GRID: tuple[int, ...] = (10, 20, 30, 50, 100)
FROZEN_FEATURES: tuple[str, ...] = (
    "top1_top2_rrf_margin", "top20_overlap", "rank_correlation_union", "score_entropy", "score_gini", "redundancy", "concentration", "sparse_dense_agreement", "sparse_dense_disagreement", "query_length", "num_numeric_tokens", "avg_idf", "max_idf",
)
FORBIDDEN_TOKENS = ("qrel", "relevance", "m_star", "oracle", "label", "reranker", "cross_encoder", "reranked", "post_", "final_score", "query_id", "doc_id")


def budget_bias() -> np.ndarray:
    """Return g(k)=-(k-mean(G))/std(G), with population standard deviation."""
    budgets = np.asarray(BUDGET_GRID, dtype=float)
    return -(budgets - budgets.mean()) / budgets.std()


def calibrated_logits(logits: Sequence[float], temperature: float, budget_bias_lambda: float) -> np.ndarray:
    """Return z'_k(q)=z_k(q)/T+lambda*g(k)."""
    values = np.asarray(logits, dtype=float)
    if values.shape != (len(BUDGET_GRID),):
        raise ValueError("one logit is required for every supported budget")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(budget_bias_lambda):
        raise ValueError("budget_bias_lambda must be finite")
    return values / temperature + budget_bias_lambda * budget_bias()


def validate_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Allow exactly the frozen, pre-reranking query/pool feature vector."""
    names = tuple(feature_names)
    forbidden = [name for name in names if any(token in name.lower() for token in FORBIDDEN_TOKENS)]
    if forbidden:
        raise ValueError(f"forbidden inference feature(s): {forbidden}")
    if names != FROZEN_FEATURES:
        raise ValueError("PoolAdapt requires exactly the frozen 13 Phase 2 features")
    return names


class PoolAdaptSelector:
    """Wrap one fitted probabilistic classifier over all five budget classes."""
    def __init__(self, model: Any, feature_names: Sequence[str] = FROZEN_FEATURES) -> None:
        self.model = model
        self.feature_names = validate_feature_names(feature_names)

    def logits(self, query_features: Mapping[str, float]) -> np.ndarray:
        values = np.asarray([[float(query_features[name]) for name in self.feature_names]], dtype=float)
        probabilities = self.model.predict_proba(values)[0]
        by_budget = {int(label): float(probability) for label, probability in zip(self.model.classes_, probabilities)}
        return np.log(np.asarray([max(by_budget.get(budget, 0.0), 1e-12) for budget in BUDGET_GRID]))

    def predict(self, query_features: Mapping[str, float], temperature: float = 1.0, budget_bias_lambda: float = 0.0) -> int:
        """Predict a budget; argmax provides the required smaller-K tie break."""
        return BUDGET_GRID[int(np.argmax(calibrated_logits(self.logits(query_features), temperature, budget_bias_lambda)))]


def select_rrf_prefix(candidates: Sequence[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Return the unmodified first K RRF candidates; no candidate-level selection."""
    if budget not in BUDGET_GRID:
        raise ValueError(f"budget must be one of {BUDGET_GRID}")
    if len(candidates) < budget:
        raise ValueError("candidate pool is shorter than the requested RRF prefix")
    return list(candidates[:budget])
