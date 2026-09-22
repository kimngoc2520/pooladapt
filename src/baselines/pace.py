"""PACE baseline boundary; implementation is intentionally deferred."""

from collections.abc import Sequence
from typing import Any


def select_pace(candidates: Sequence[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Reserve the PACE interface without claiming a reproduction yet."""
    raise NotImplementedError("PACE requires a separately validated reproduction protocol")
