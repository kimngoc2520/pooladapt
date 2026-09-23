"""Fixed-prefix candidate-selection baseline."""

from collections.abc import Sequence
from typing import Any


MAX_POOL_SIZE = 100


def select(candidates: Sequence[dict[str, Any]], m: int) -> list[dict[str, Any]]:
    """Select the first ``m`` candidates without altering hybrid rank order."""
    if m < 0:
        raise ValueError("m must be non-negative")
    return list(candidates[: min(m, MAX_POOL_SIZE)])


def select_fixed_prefix(candidates: Sequence[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Backward-compatible name for :func:`select`."""
    return select(candidates, m=budget)
