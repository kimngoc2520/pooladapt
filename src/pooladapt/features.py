"""Extensible candidate-level retrieval features; no selector is prescribed here."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def characterize_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach source availability and rank-gap signals while preserving candidates."""
    featured: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        sparse_rank, dense_rank = item.get("bm25_rank"), item.get("dense_rank")
        item["sparse_dense_agreement"] = sparse_rank is not None and dense_rank is not None
        item["rank_gap"] = abs(sparse_rank - dense_rank) if item["sparse_dense_agreement"] else None
        featured.append(item)
    return featured
