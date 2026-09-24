"""Seeded random candidate-selection baseline."""

import random
from collections.abc import Sequence
from typing import Any

from src.reranking.cross_encoder import Reranker


MAX_CANDIDATE_POOL_SIZE = 100


def select_random(candidates: Sequence[dict[str, Any]], budget: int, seed: int = 0) -> list[dict[str, Any]]:
    if len(candidates) > MAX_CANDIDATE_POOL_SIZE:
        raise ValueError("candidate pool must contain at most 100 documents")
    if not 0 <= budget <= len(candidates):
        raise ValueError("budget must be between zero and candidate-pool size")
    return random.Random(seed).sample(list(candidates), budget)


def rerank_random(
    query: str,
    candidates: Sequence[dict[str, Any]],
    budget: int,
    reranker: Reranker,
    seed: int = 0,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Select exactly ``budget`` unique candidates, then rerank only those pairs."""
    return reranker.rerank(query, select_random(candidates, budget, seed=seed), top_k=top_k)
