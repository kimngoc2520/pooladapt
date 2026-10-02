import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.baselines.sage_slo import BUDGET_GRID, budget_bias, calibrated_logits, select_budget, validate_feature_names


class SageSloTests(unittest.TestCase):
    def test_normalized_budget_bias(self):
        expected = -(np.asarray(BUDGET_GRID, dtype=float) - np.mean(BUDGET_GRID)) / np.std(BUDGET_GRID)
        np.testing.assert_allclose(budget_bias(), expected)

    def test_calibrated_logits_formula(self):
        logits = np.arange(5, dtype=float)
        np.testing.assert_allclose(calibrated_logits(logits, 2.0, 3.0), logits / 2.0 + 3.0 * budget_bias())

    def test_lambda_directionality_and_tie_breaking(self):
        zeros = np.zeros(5)
        self.assertEqual(select_budget(zeros, budget_bias_lambda=1), 10)
        self.assertEqual(select_budget(zeros, budget_bias_lambda=-1), 100)
        self.assertEqual(select_budget(zeros), 10)

    def test_valid_budget_and_temperature_is_not_hard_control_alone(self):
        logits = [-4, -3, -2, -1, 0]
        self.assertIn(select_budget(logits, temperature=0.1), BUDGET_GRID)
        self.assertEqual(select_budget(logits, temperature=0.1), select_budget(logits, temperature=10))

    def test_forbidden_inputs_and_existing_feature_names(self):
        valid = ("score_entropy", "query_length", "sparse_dense_agreement")
        self.assertEqual(validate_feature_names(valid), valid)
        with self.assertRaises(ValueError): validate_feature_names(("qrels",))
        with self.assertRaises(ValueError): validate_feature_names(("final_reranker_score",))

    def test_lambdas_can_change_average_budget(self):
        logits = [np.zeros(5), np.zeros(5), np.zeros(5)]
        low = np.mean([select_budget(x, budget_bias_lambda=-5) for x in logits])
        high = np.mean([select_budget(x, budget_bias_lambda=5) for x in logits])
        self.assertLess(high, low)

    def test_output_schema(self):
        fields = ["query_id", "predicted_budget", "temperature", "budget_bias_lambda", "nDCG@10", "Recall@10", "MRR@10", "reranked_pairs", "compression_ratio", "reranking_latency"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sage.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
                writer.writerow(dict(zip(fields, ["q", 10, 1.0, 0.0, .5, .5, .5, 10, .9, .01])))
            with path.open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(tuple(row), tuple(fields)); self.assertIn(int(row["predicted_budget"]), BUDGET_GRID); self.assertGreaterEqual(float(row["reranking_latency"]), 0)
