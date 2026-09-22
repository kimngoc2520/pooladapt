"""Seeded random candidate-selection baseline."""

import random
from collections.abc import Sequence
from typing import Any


def select_random(candidates: Sequence[dict[str, Any]], budget: int, seed: int = 0) -> list[dict[str, Any]]:
    if not 0 <= budget <= len(candidates):
        raise ValueError("budget must be between zero and candidate-pool size")
    return random.Random(seed).sample(list(candidates), budget)
