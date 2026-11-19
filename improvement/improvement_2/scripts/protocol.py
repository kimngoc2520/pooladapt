"""Protocol-defined numerical primitives for the CT2 development analysis.

These functions deliberately keep policy selection independent of qrels and
retrieval outcomes.  They are shared by the runner and deterministic tests.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np

BUDGETS = np.array([10, 20, 30, 50, 100], dtype=int)


def align_probabilities(classes: Iterable[int], probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map estimator probabilities to the fixed budget grid and return a mask."""
    probabilities = np.asarray(probabilities, dtype=float)
    result = np.zeros((len(probabilities), len(BUDGETS)), dtype=float)
    available = np.zeros(len(BUDGETS), dtype=bool)
    for source, label in enumerate(classes):
        target = int(np.flatnonzero(BUDGETS == int(label))[0])
        result[:, target] = probabilities[:, source]
        available[target] = True
    return result, available


def hard_probability(probabilities: np.ndarray) -> np.ndarray:
    """Return P(M* > 10) from a fixed-grid probability matrix."""
    return np.asarray(probabilities, dtype=float)[:, 1:].sum(axis=1)


def shifted_budget(probabilities: np.ndarray, available: np.ndarray, lam: float) -> np.ndarray:
    """Apply the Protocol CT2.1.6 lambda shift without reviving absent classes."""
    logs = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-12, None))
    mask = np.asarray(available, dtype=bool)
    if mask.ndim == 1:
        mask = np.broadcast_to(mask, logs.shape)
    if mask.shape != logs.shape:
        raise ValueError("availability mask must be a budget vector or probability-shaped matrix")
    logs[~mask] = -np.inf
    if not np.any(mask):
        return np.full(len(logs), 10, dtype=int)
    g = -(BUDGETS - BUDGETS.mean()) / BUDGETS.std()
    # np.argmax deterministically chooses the smaller K on exact ties.
    return BUDGETS[np.argmax(logs + lam * g, axis=1)]


def bisection_policy(probabilities: np.ndarray, available: np.ndarray, target: float) -> tuple[np.ndarray, float, float, bool]:
    """Select a pooled-OOF lambda by bisection; report realized discrete cost."""
    low, high = -100.0, 100.0
    candidates: list[tuple[np.ndarray, float, float]] = []
    for _ in range(80):
        lam = (low + high) / 2
        chosen = shifted_budget(probabilities, available, lam)
        mean_k = float(chosen.mean())
        candidates.append((chosen, lam, mean_k))
        if mean_k > target:
            low = lam
        else:
            high = lam
    chosen, lam, mean_k = min(candidates, key=lambda x: (abs(x[2] - target), x[2], abs(x[1]), x[1]))
    return chosen, float(lam), float(mean_k), abs(mean_k - target) <= 0.1


def closed_form_difference(chosen: np.ndarray, quality: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    """Return paired policy-minus-null nDCG differences for one sample.

    The selected-budget frequency is intentionally computed inside this
    function, so a bootstrap caller recomputes the null on every resample.
    """
    quality = np.asarray(quality, dtype=float)
    if indices is not None:
        quality = quality[np.asarray(indices, dtype=int)]
        chosen = np.asarray(chosen, dtype=int)[np.asarray(indices, dtype=int)]
    frequencies = np.array([(chosen == k).mean() for k in BUDGETS])
    policy = quality[np.arange(len(quality)), np.searchsorted(BUDGETS, chosen)]
    null = quality @ frequencies
    return policy - null


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm adjusted p-values in original order."""
    p = np.asarray(list(p_values), dtype=float)
    out = np.zeros(len(p)); previous = 0.0
    for rank, index in enumerate(np.argsort(p)):
        previous = max(previous, min(1.0, (len(p) - rank) * p[index]))
        out[index] = previous
    return out
