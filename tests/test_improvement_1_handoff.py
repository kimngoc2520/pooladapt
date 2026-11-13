"""Targeted tests for Improvement 1 handoff validation and cost definitions."""

from __future__ import annotations

import unittest

from src.evaluation.handoff_metrics import (
    average_actual_reranked_pairs,
    average_k,
    compression_ratio,
    cost_reduction,
    execution_key,
    full_rerank_reference,
    metrics_match,
    population_coverage,
    validate_qrels,
    validate_unique_execution_keys,
)
from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k


class TestImprovement1Handoff(unittest.TestCase):
    def test_execution_key_includes_seed_for_real_repeated_executions(self) -> None:
        first = {"query_id": "q1", "method": "Random", "operating_point": "B10", "seed": 0}
        second = {**first, "seed": 1}

        self.assertNotEqual(execution_key(first), execution_key(second))
        validate_unique_execution_keys([first, second])

    def test_population_coverage_reports_missing_unexpected_and_duplicate_ids(self) -> None:
        missing, unexpected, duplicates = population_coverage(
            ["q1", "q2"], ["q1", "q1", "q3"]
        )

        self.assertEqual(missing, {"q2"})
        self.assertEqual(unexpected, {"q3"})
        self.assertEqual(duplicates, {"q1"})

    def test_missing_predictions_are_not_silently_removed(self) -> None:
        missing, unexpected, duplicates = population_coverage(["q1", "q2"], ["q1"])

        self.assertEqual(missing, {"q2"})
        self.assertFalse(unexpected)
        self.assertFalse(duplicates)

    def test_duplicate_prediction_ids_are_detected(self) -> None:
        _, _, duplicates = population_coverage(["q1"], ["q1", "q1"])

        self.assertEqual(duplicates, {"q1"})

    def test_qrels_require_positive_valid_judgments_for_each_query(self) -> None:
        validate_qrels({"q1": {"d1": 1}}, ["q1"])

        with self.assertRaisesRegex(ValueError, "missing qrels"):
            validate_qrels({}, ["q1"])
        with self.assertRaisesRegex(ValueError, "no positive"):
            validate_qrels({"q1": {"d1": 0}}, ["q1"])
        with self.assertRaisesRegex(ValueError, "invalid qrel"):
            validate_qrels({"q1": {"d1": -1}}, ["q1"])

    def test_query_metric_reproduction_uses_repository_definitions(self) -> None:
        ranking = ["d2", "d1", "d3"]
        relevance = {"d1": 1.0, "d2": 0.0, "d3": 1.0}
        observed = {
            "ndcg_at_10": ndcg_at_k(ranking, relevance),
            "recall_at_10": recall_at_k(ranking, ["d1", "d3"]),
            "mrr_at_10": mrr_at_k(ranking, ["d1", "d3"]),
        }
        expected = {"ndcg_at_10": 0.6934264036172708, "recall_at_10": 1.0, "mrr_at_10": 0.5}

        self.assertTrue(metrics_match(observed, expected, tolerance=1e-12))

    def test_aggregate_metric_consistency_flags_differences(self) -> None:
        self.assertTrue(metrics_match({"ndcg_at_10": 0.5}, {"ndcg_at_10": 0.5}, ("ndcg_at_10",)))
        self.assertFalse(metrics_match({"ndcg_at_10": 0.4}, {"ndcg_at_10": 0.5}, ("ndcg_at_10",)))

    def test_average_k_is_controller_allocation_mean(self) -> None:
        self.assertEqual(average_k([10, 20, 30]), 20)

    def test_actual_pair_mean_uses_execution_counts_not_selected_k(self) -> None:
        self.assertEqual(average_actual_reranked_pairs([9, 19, 29]), 19)
        self.assertEqual(average_k([10, 20, 30]), 20)

    def test_compression_ratio_uses_full_rerank_pairs_denominator(self) -> None:
        self.assertEqual(compression_ratio(25, 100), 0.25)
        self.assertEqual(compression_ratio(100, 100), 1.0)

    def test_cost_reduction_is_one_minus_compression_ratio(self) -> None:
        self.assertEqual(cost_reduction(0.25), 0.75)
        self.assertEqual(cost_reduction(1.0), 0.0)

    def test_full_rerank_reference_requires_full_pool_processing(self) -> None:
        self.assertEqual(full_rerank_reference([100, 100, 100]), 100)

        with self.assertRaisesRegex(ValueError, "full candidate pool"):
            full_rerank_reference([99, 100])

    def test_missing_actual_execution_data_is_explicit(self) -> None:
        self.assertIsNone(average_actual_reranked_pairs([None, None]))

        with self.assertRaisesRegex(ValueError, "incomplete"):
            average_actual_reranked_pairs([10, None])

    def test_duplicate_execution_log_rows_are_rejected_but_seed_runs_are_allowed(self) -> None:
        row = {"query_id": "q1", "method": "Random", "operating_point": "B10", "seed": 0}

        with self.assertRaisesRegex(ValueError, "duplicate execution keys"):
            validate_unique_execution_keys([row, dict(row)])
        validate_unique_execution_keys([row, {**row, "seed": 1}])


if __name__ == "__main__":
    unittest.main()
