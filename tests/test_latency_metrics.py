"""Tests for Phase 1 latency aggregation against the current latency_metrics API."""

from __future__ import annotations

import unittest

from src.evaluation.latency_metrics import (
    aggregate_latency_methods,
    build_latency_report,
    latency_reduction,
    method_latency_samples,
    summarize_latency_samples,
)


def _prediction(query_id: str, rerank: float, ce: float) -> dict[str, float | str]:
    return {
        "query_id": query_id,
        "latency_rerank_seconds": rerank,
        "latency_ce_seconds": ce,
    }


class LatencyMetricsTests(unittest.TestCase):
    def test_mean_p50_and_p95_are_computed_separately_for_rerank_and_ce(self) -> None:
        summary = summarize_latency_samples([1.0, 2.0, 3.0], [0.5, 1.0, 1.5])

        rerank = summary["rerank_latency_seconds"]
        ce = summary["ce_latency_seconds"]

        self.assertEqual(rerank["mean"], 2.0)
        self.assertEqual(rerank["p50"], 2.0)
        self.assertEqual(rerank["p95"], 2.9)
        self.assertEqual(ce["mean"], 1.0)
        self.assertEqual(ce["p50"], 1.0)
        self.assertEqual(ce["p95"], 1.45)

    def test_rerank_and_ce_summaries_are_independent(self) -> None:
        summary = summarize_latency_samples([10.0, 20.0], [1.0, 3.0])

        self.assertEqual(summary["rerank_latency_seconds"]["mean"], 15.0)
        self.assertEqual(summary["ce_latency_seconds"]["mean"], 2.0)
        self.assertNotEqual(
            summary["rerank_latency_seconds"],
            summary["ce_latency_seconds"],
        )

    def test_latency_reduction_is_one_minus_method_over_full(self) -> None:
        full = {"mean": 10.0, "p50": 10.0, "p95": 10.0}
        method = {"mean": 5.0, "p50": 5.0, "p95": 5.0}
        self.assertEqual(latency_reduction(method, full), {"mean": 0.5, "p50": 0.5, "p95": 0.5})

    def test_latency_reduction_uses_matching_statistics(self) -> None:
        full = {"mean": 10.0, "p50": 8.0, "p95": 20.0}
        method = {"mean": 5.0, "p50": 6.0, "p95": 10.0}
        self.assertEqual(latency_reduction(method, full), {"mean": 0.5, "p50": 0.25, "p95": 0.5})

    def test_full_rerank_is_required_as_reduction_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "full_rerank"):
            aggregate_latency_methods(
                {
                    "fixed_prefix_10": {
                        "predictions": [_prediction("q1", 5.0, 4.0)],
                    }
                }
            )

    def test_full_rerank_has_no_latency_reduction(self) -> None:
        summaries = aggregate_latency_methods(
            {
                "full_rerank": {
                    "predictions": [_prediction("q1", 10.0, 8.0)],
                },
                "fixed_prefix_10": {
                    "predictions": [_prediction("q1", 5.0, 4.0)],
                },
            }
        )

        self.assertIsNone(summaries["full_rerank"]["latency_reduction"])
        self.assertEqual(
            summaries["fixed_prefix_10"]["latency_reduction"],
            {"mean": 0.5, "p50": 0.5, "p95": 0.5},
        )
        self.assertEqual(summaries["full_rerank"]["rerank_latency_seconds"]["mean"], 10.0)
        self.assertEqual(summaries["full_rerank"]["ce_latency_seconds"]["mean"], 8.0)

    def test_reduction_uses_rerank_latency_not_ce_latency(self) -> None:
        summaries = aggregate_latency_methods(
            {
                "full_rerank": {
                    "predictions": [_prediction("q1", 10.0, 2.0)],
                },
                "fixed_prefix_10": {
                    "predictions": [_prediction("q1", 5.0, 1.9)],
                },
            }
        )

        self.assertEqual(
            summaries["fixed_prefix_10"]["latency_reduction"],
            {"mean": 0.5, "p50": 0.5, "p95": 0.5},
        )
        self.assertEqual(summaries["fixed_prefix_10"]["ce_latency_seconds"]["mean"], 1.9)

    def test_zero_full_rerank_latency_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-zero"):
            latency_reduction(
                {"mean": 1.0, "p50": 1.0, "p95": 1.0},
                {"mean": 0.0, "p50": 1.0, "p95": 1.0},
            )

    def test_missing_summary_statistic_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing 'p95'"):
            latency_reduction({"mean": 1.0, "p50": 1.0}, {"mean": 2.0, "p50": 2.0, "p95": 2.0})

    def test_empty_samples_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one latency sample"):
            summarize_latency_samples([], [])

    def test_mismatched_latency_samples_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "counts must match"):
            summarize_latency_samples([1.0], [1.0, 2.0])

    def test_empty_predictions_raise_during_aggregation(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one latency sample"):
            aggregate_latency_methods({"full_rerank": {"predictions": []}})

    def test_missing_latency_fields_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "rerank and cross-encoder latency fields"):
            method_latency_samples([{"query_id": "q1", "latency_rerank_seconds": 1.0}])

    def test_non_sequence_predictions_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "predictions sequence"):
            aggregate_latency_methods({"full_rerank": {"predictions": "invalid"}})

    def test_method_latency_samples_extract_paired_fields(self) -> None:
        rerank, ce = method_latency_samples(
            [
                _prediction("q1", 10.0, 8.0),
                _prediction("q2", 12.0, 9.0),
            ]
        )
        self.assertEqual(rerank, [10.0, 12.0])
        self.assertEqual(ce, [8.0, 9.0])

    def test_build_latency_report_aggregates_prediction_structure(self) -> None:
        report = build_latency_report(
            {
                "dataset": "synthetic",
                "candidate_pool_size": 100,
                "top_k": 10,
                "methods": {
                    "full_rerank": {
                        "predictions": [_prediction("q1", 10.0, 8.0)],
                    },
                    "fixed_prefix_10": {
                        "predictions": [_prediction("q1", 5.0, 4.0)],
                    },
                },
            }
        )

        self.assertEqual(report["dataset"], "synthetic")
        self.assertEqual(report["candidate_pool_size"], 100)
        self.assertEqual(report["top_k"], 10)

        full = report["methods"]["full_rerank"]
        fixed = report["methods"]["fixed_prefix_10"]
        self.assertIsNone(full["latency_reduction"])
        self.assertEqual(fixed["rerank_latency_seconds"]["mean"], 5.0)
        self.assertEqual(fixed["ce_latency_seconds"]["mean"], 4.0)
        self.assertEqual(fixed["latency_reduction"], {"mean": 0.5, "p50": 0.5, "p95": 0.5})

    def test_build_latency_report_requires_methods_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "methods mapping"):
            build_latency_report({"dataset": "synthetic"})


if __name__ == "__main__":
    unittest.main()
