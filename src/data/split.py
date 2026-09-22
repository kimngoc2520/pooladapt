"""Deterministic split utility for dataset records."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def split_items(items: Sequence[T], validation_fraction: float, seed: int = 0) -> tuple[list[T], list[T]]:
    """Return deterministic train and validation partitions."""
    if not 0 <= validation_fraction <= 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    boundary = round(len(shuffled) * (1 - validation_fraction))
    return shuffled[:boundary], shuffled[boundary:]
