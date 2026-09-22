"""BM25 retrieval interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class BM25Retriever(Protocol):
    def retrieve(self, query: str, top_n: int = 100) -> Sequence[dict[str, Any]]:
        """Return ranked sparse-retrieval candidates."""
