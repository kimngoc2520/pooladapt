"""Full-pool reranking baseline."""

from collections.abc import Sequence
from typing import Any


def select_full_pool(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(candidates)
