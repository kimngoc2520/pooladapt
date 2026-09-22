"""Protocol shared by every reranking method."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class CrossEncoderReranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        """Score candidates and return them in descending cross-encoder order."""
