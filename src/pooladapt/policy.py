"""Selection-policy abstraction; model design remains an experimental question."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class SelectionPolicy(Protocol):
    def select(self, candidates: Sequence[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        """Choose candidates from the already-characterized candidate pool."""
