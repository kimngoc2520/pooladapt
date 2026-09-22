"""Fixed-prefix candidate-selection baseline."""

from collections.abc import Sequence
from typing import Any


def select_fixed_prefix(candidates: Sequence[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    return list(candidates[:budget])
