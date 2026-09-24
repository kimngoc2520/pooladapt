"""Full-pool reranking baseline."""

from collections.abc import Sequence
from typing import Any

from src.reranking.cross_encoder import Reranker


FULL_RERANK_POOL_SIZE = 100


def select_full_pool(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(candidates)


def rerank_full(
    query: str,
    candidates: Sequence[dict[str, Any]],
    reranker: Reranker,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank every candidate in the N=100 reference pool."""
    selected = select_full_pool(candidates)
    if len(selected) > FULL_RERANK_POOL_SIZE:
        raise ValueError("full rerank expects a candidate pool of at most 100 documents")
    return reranker.rerank(query, selected, top_k=top_k)
