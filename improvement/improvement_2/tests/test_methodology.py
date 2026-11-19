"""Deterministic regression tests for the Protocol CT2 numerical primitives."""
from __future__ import annotations

import numpy as np
from improvement.improvement_2.scripts.protocol import (
    BUDGETS, align_probabilities, bisection_policy, closed_form_difference,
    hard_probability, holm_adjust, shifted_budget,
)


def test_fixed_feature_configuration() -> None:
    query = ("query_length", "num_numeric_tokens", "avg_idf", "max_idf")
    pool = ("top1_top2_rrf_margin", "top20_overlap", "rank_correlation_union", "score_entropy", "score_gini", "redundancy", "concentration", "sparse_dense_disagreement")
    assert len(query + pool) == 12 and len(pool) == 8 and len(query) == 4
    assert "sparse_dense_agreement" not in query + pool


def test_probability_alignment_hard_and_mask() -> None:
    p, mask = align_probabilities([10, 50], np.array([[.7, .3]]))
    assert p.tolist() == [[.7, 0., 0., .3, 0.]]
    assert mask.tolist() == [True, False, False, True, False]
    assert hard_probability(p).tolist() == [.3]
    # Missing classes cannot be made available by a positive lambda shift.
    assert shifted_budget(p, mask, -100)[0] in (10, 50)


def test_bisection_and_closed_form_resample() -> None:
    p = np.array([[.9,.1,0,0,0],[.2,.8,0,0,0],[.9,.1,0,0,0],[.2,.8,0,0,0]])
    chosen, _, cost, feasible = bisection_policy(p, np.ones(5, bool), 15)
    assert feasible and cost == 15 and set(chosen) == {10,20}
    quality = np.array([[0, .2, .2, .2, .2],[0,.4,.4,.4,.4],[0,.6,.6,.6,.6],[0,.8,.8,.8,.8]])
    full = closed_form_difference(chosen, quality)
    duplicate = closed_form_difference(chosen, quality, np.array([0,0,0,1]))
    assert full.shape == (4,) and duplicate.shape == (4,)
    assert not np.isclose(full.mean(), duplicate.mean())


def test_holm() -> None:
    assert np.allclose(holm_adjust([.01,.04]), [.02,.04])
