from __future__ import annotations

import pytest

from src.evaluation.cost_metrics import (
    average_reranked_pairs,
    compression_ratio,
    reranked_pairs_per_query,
    total_reranking_pairs,
)


def test_reranked_pairs_per_query_returns_input_count() -> None:
    assert reranked_pairs_per_query(10) == 10
    assert reranked_pairs_per_query(100) == 100


def test_reranked_pairs_per_query_rejects_negative_count() -> None:
    with pytest.raises(ValueError):
        reranked_pairs_per_query(-1)


def test_total_reranking_pairs_sums_query_counts() -> None:
    pair_counts = [10, 20, 30]

    assert total_reranking_pairs(pair_counts) == 60


def test_total_reranking_pairs_returns_zero_for_empty_input() -> None:
    assert total_reranking_pairs([]) == 0


def test_average_reranked_pairs_computes_mean() -> None:
    pair_counts = [10, 20, 30]

    assert average_reranked_pairs(pair_counts) == pytest.approx(20.0)


def test_average_reranked_pairs_returns_zero_for_empty_input() -> None:
    assert average_reranked_pairs([]) == 0.0


def test_compression_ratio_for_full_reranking() -> None:
    assert compression_ratio(100, 100) == pytest.approx(0.0)


def test_compression_ratio_for_half_budget() -> None:
    assert compression_ratio(100, 50) == pytest.approx(0.5)


def test_compression_ratio_for_ten_percent_budget() -> None:
    assert compression_ratio(100, 10) == pytest.approx(0.9)


def test_compression_ratio_rejects_invalid_candidate_pool_size() -> None:
    with pytest.raises(ValueError):
        compression_ratio(0, 10)


def test_compression_ratio_rejects_negative_reranked_count() -> None:
    with pytest.raises(ValueError):
        compression_ratio(100, -1)


def test_compression_ratio_rejects_budget_above_candidate_pool() -> None:
    with pytest.raises(ValueError):
        compression_ratio(100, 101)