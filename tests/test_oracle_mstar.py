import inspect
import unittest

import src.analysis.oracle_mstar as oracle_mstar
from src.analysis.oracle_mstar import analyse_mstar, compute_mstar_labels, prediction_index


class OracleMstarTests(unittest.TestCase):
    def setUp(self):
        self.qrels = {"q1": {"a": 1}, "q2": {"b": 3, "c": 1}}
        self.full = {"q1": ["a"], "q2": ["b", "c"]}
        self.rankings = {
            10: {"q1": ["a"], "q2": ["x"]},
            20: {"q1": ["a"], "q2": ["b", "c"]},
            30: {"q1": ["a"], "q2": ["b", "c"]},
            50: {"q1": ["a"], "q2": ["b", "c"]},
            100: {"q1": ["a"], "q2": ["b", "c"]},
        }

    def test_minimum_budget_and_audit_values(self):
        rows = compute_mstar_labels(self.qrels, self.rankings, self.full, 0.98)
        first = next(row for row in rows if row["query_id"] == "q1")
        self.assertEqual(first["M_star"], 10)
        self.assertEqual(first["M_star_is_min"], 1)
        self.assertEqual(first["M_star_is_max"], 0)
        self.assertEqual(first["nDCG_at_M_star"], 1.0)
        self.assertEqual(first["nDCG_full"], 1.0)

    def test_selects_first_budget_meeting_threshold(self):
        rows = compute_mstar_labels(self.qrels, self.rankings, self.full, 0.98)
        self.assertEqual(rows[1]["M_star"], 20)

    def test_threshold_comparison_and_fallback_to_maximum(self):
        rankings = {budget: dict(values) for budget, values in self.rankings.items()}
        rankings[10]["q2"] = ["b", "x", "c"]
        # q2's M=10 nDCG is below full but clears .95, not .98.
        self.assertEqual(compute_mstar_labels(self.qrels, rankings, self.full, 0.95)[1]["M_star"], 10)
        self.assertEqual(compute_mstar_labels(self.qrels, rankings, self.full, 0.98)[1]["M_star"], 20)
        rankings[20]["q2"] = ["x"]
        rankings[30]["q2"] = ["x"]
        rankings[50]["q2"] = ["x"]
        rankings[100]["q2"] = ["x"]
        row = compute_mstar_labels(self.qrels, rankings, self.full, 0.99)[1]
        self.assertEqual(row["M_star"], 100)
        self.assertEqual(row["M_star_is_max"], 1)

    def test_duplicate_queries_are_rejected(self):
        with self.assertRaises(ValueError):
            prediction_index([
                {"query_id": "q1", "document_ids": ["a"]},
                {"query_id": "q1", "document_ids": ["a"]},
            ], "synthetic")

    def test_deterministic_output_and_g1(self):
        first = compute_mstar_labels(self.qrels, self.rankings, self.full, 0.98)
        second = compute_mstar_labels(self.qrels, self.rankings, self.full, 0.98)
        self.assertEqual(first, second)
        analysis = analyse_mstar(first)
        self.assertFalse(analysis["G1_passes"])
        self.assertEqual(analysis["number_of_distinct_M_star_levels"], 2)

    def test_oracle_has_no_feature_file_dependency(self):
        source = inspect.getsource(oracle_mstar)
        for forbidden in ("candidate_features.csv", "pool_features.csv", "query_features.csv", "candidate_labels.csv"):
            self.assertNotIn(forbidden, source)
