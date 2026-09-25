"""Tests for Phase 1 latency aggregation and reporting data."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.run_latency_evaluation import main as run_latency_evaluation
from src.evaluation.latency_metrics import (
    aggregate_latency_methods,
    build_latency_report,
    latency_reduction,
    summarize_latency_samples,
)


class LatencyMetricsTests(unittest.TestCase):
    def test_latency_summary_contains_mean_p50_and_p95(self) -> None:
        summary = summarize_latency_samples([1.0, 2.0, 3.0], [0.5, 1.0, 1.5])
        self.assertEqual(summary["rerank_latency_seconds"], {"mean": 2.0, "p50": 2.0, "p95": 2.9})
        self.assertEqual(summary["ce_latency_seconds"], {"mean": 1.0, "p50": 1.0, "p95": 1.45})

    def test_latency_reduction_is_like_for_like(self) -> None:
        full = {"mean": 10.0, "p50": 10.0, "p95": 10.0}
        method = {"mean": 5.0, "p50": 5.0, "p95": 5.0}
        self.assertEqual(latency_reduction(method, full), {"mean": 0.5, "p50": 0.5, "p95": 0.5})

    def test_non_uniform_latency_uses_each_matching_statistic(self) -> None:
        full = {"mean": 10.0, "p50": 8.0, "p95": 20.0}
        method = {"mean": 5.0, "p50": 6.0, "p95": 10.0}
        self.assertEqual(latency_reduction(method, full), {"mean": 0.5, "p50": 0.25, "p95": 0.5})

    def test_missing_full_rerank_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "full_rerank"):
            aggregate_latency_methods({"fixed_prefix_10": {"predictions": []}})

    def test_zero_full_latency_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-zero"):
            latency_reduction({"mean": 1.0, "p50": 1.0, "p95": 1.0}, {"mean": 0.0, "p50": 1.0, "p95": 1.0})

    def test_mismatched_latency_samples_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "counts must match"):
            summarize_latency_samples([1.0], [1.0, 2.0])

    def test_prediction_structure_is_aggregated(self) -> None:
        report = build_latency_report(
            {
                "dataset": "scifact",
                "candidate_pool_size": 100,
                "top_k": 10,
                "methods": {
                    "full_rerank": {
                        "predictions": [
                            {"query_id": "q1", "latency_rerank_seconds": 10.0, "latency_ce_seconds": 8.0}
                        ]
                    },
                    "fixed_prefix_10": {
                        "predictions": [
                            {"query_id": "q1", "latency_rerank_seconds": 5.0, "latency_ce_seconds": 4.0}
                        ]
                    },
                },
            }
        )
        fixed = report["methods"]["fixed_prefix_10"]
        self.assertEqual(fixed["rerank_latency_seconds"]["mean"], 5.0)
        self.assertEqual(fixed["ce_latency_seconds"]["mean"], 4.0)
        self.assertEqual(fixed["latency_reduction"], {"mean": 0.5, "p50": 0.5, "p95": 0.5})

    def test_reporting_cli_writes_fixture_report(self) -> None:
        fixture = {
            "dataset": "scifact",
            "candidate_pool_size": 100,
            "top_k": 10,
            "methods": {
                "full_rerank": {
                    "predictions": [
                        {"query_id": "q1", "latency_rerank_seconds": 10.0, "latency_ce_seconds": 8.0}
                    ]
                },
                "fixed_prefix_10": {
                    "predictions": [
                        {"query_id": "q1", "latency_rerank_seconds": 5.0, "latency_ce_seconds": 4.0}
                    ]
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            predictions_path = temporary_path / "predictions.json"
            output_path = temporary_path / "latency.json"
            predictions_path.write_text(json.dumps(fixture), encoding="utf-8")
            with patch.object(
                sys,
                "argv",
                ["run_latency_evaluation.py", "--predictions", str(predictions_path), "--output", str(output_path)],
            ):
                run_latency_evaluation()
            report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["methods"]["full_rerank"]["latency_reduction"], None)
        self.assertEqual(report["methods"]["fixed_prefix_10"]["latency_reduction"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
