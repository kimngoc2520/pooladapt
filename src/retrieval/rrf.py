"""Reciprocal-rank fusion for the shared hybrid candidate pool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


RRF_K = 60


def fuse_ranked_lists(
    ranked_lists: Mapping[str, Sequence[Mapping[str, Any]]], top_n: int = 100, k: int = RRF_K
) -> list[dict[str, Any]]:
    """Fuse ranked lists using standard RRF, retaining source ranks and scores.

    Candidates must expose an ``id`` key. A missing candidate contributes zero.
    """
    if k < 0 or top_n < 0:
        raise ValueError("k and top_n must be non-negative")
    pooled: dict[str, dict[str, Any]] = {}
    for source, candidates in ranked_lists.items():
        for rank, candidate in enumerate(candidates, start=1):
            document_id = str(candidate["id"])
            result = pooled.setdefault(document_id, {"id": document_id, "rrf_score": 0.0})
            result.update({key: value for key, value in candidate.items() if key != "id"})
            result[f"{source}_rank"] = rank
            if "score" in candidate:
                result[f"{source}_score"] = candidate["score"]
            result["rrf_score"] += 1 / (k + rank)
    return sorted(pooled.values(), key=lambda item: item["rrf_score"], reverse=True)[:top_n]
