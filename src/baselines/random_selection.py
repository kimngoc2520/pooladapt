"""Seeded random candidate-selection baseline."""

import hashlib
import random
from collections.abc import Sequence
from typing import Any

from src.reranking.cross_encoder import Reranker


MAX_CANDIDATE_POOL_SIZE = 100


def select_random(
    candidates: Sequence[dict[str, Any]],
    budget: int,
    seed: int = 0,
    query_id: str | None = None,
) -> list[dict[str, Any]]:
    if len(candidates) > MAX_CANDIDATE_POOL_SIZE:
        raise ValueError("candidate pool must contain at most 100 documents")
    if not 0 <= budget <= len(candidates):
        raise ValueError("budget must be between zero and candidate-pool size")
    query_seed = seed
    if query_id is not None:
        digest = hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).digest()
        query_seed = int.from_bytes(digest[:8], "big")
    return random.Random(query_seed).sample(list(candidates), budget)


def rerank_random(
    query: str,
    candidates: Sequence[dict[str, Any]],
    budget: int,
    reranker: Reranker,
    seed: int = 0,
    top_k: int | None = None,
    query_id: str | None = None,
) -> list[dict[str, Any]]:
    """Select exactly ``budget`` unique candidates, then rerank only those pairs."""
    return reranker.rerank(
        query,
        select_random(candidates, budget, seed=seed, query_id=query_id),
        top_k=top_k,
    )
