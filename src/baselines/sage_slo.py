"""SAGE-SLO baseline boundary; implementation is intentionally deferred."""

from collections.abc import Sequence
from typing import Any


def select_sage_slo(candidates: Sequence[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Reserve the SAGE-SLO interface without claiming a reproduction yet."""
    raise NotImplementedError("SAGE-SLO requires a separately validated reproduction protocol")
