"""Candidate-level selection entry point."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .policy import SelectionPolicy


def select_candidates(candidates: Sequence[dict[str, Any]], budget: int, policy: SelectionPolicy) -> list[dict[str, Any]]:
    """Delegate selection of M candidates from N to an explicit policy."""
    if not 0 <= budget <= len(candidates):
        raise ValueError("budget must be between zero and candidate-pool size")
    selected = policy.select(candidates, budget)
    if len(selected) > budget:
        raise ValueError("selection policy exceeded the requested budget")
    return selected
