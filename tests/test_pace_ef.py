import inspect
import unittest

from src.baselines.pace_ef import (
    bm25_document_contribution,
    evidence_coverage,
    evidence_frontload,
    marginal_contribution,
    normalize_rrf_scores,
    select_prefix,
)
from experiments.run_pace_ef import BUDGETS, RESULT_FIELDS


class PaceEvidenceFrontloadingTests(unittest.TestCase):
    def test_bm25_term_factor_matches_formula_and_zero_tf(self):
        expected = 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 10 / 20))
        self.assertAlmostEqual(bm25_document_contribution(2, 10, 1.2, 0.75, 20), expected)
        self.assertEqual(bm25_document_contribution(0, 10, 1.2, 0.75, 20), 0.0)

    def test_rrf_normalization_and_constant_pool(self):
        self.assertEqual(normalize_rrf_scores([1, 2, 3]), [0.0, 1 / (2 + 1e-8), 2 / (2 + 1e-8)])
        self.assertEqual(normalize_rrf_scores([4, 4]), [0.0, 0.0])

    def test_coverage_and_per_term_maxima(self):
        selected = [
            {"rho": 0.5, "term_contributions": {"a": 2, "b": 1}},
            {"rho": 1.0, "term_contributions": {"a": 1, "b": 3}},
        ]
        value, maxima = evidence_coverage(selected, {"a": 2, "b": 1})
        self.assertEqual(maxima, {"a": 1.0, "b": 3.0})
        self.assertEqual(value, 5.0)

    def test_marginal_increase_and_redundant_candidate(self):
        weights = {"a": 2.0, "b": 1.0}
        maxima = {"a": 1.0, "b": 0.5}
        candidate = {"rho": 1.0, "term_contributions": {"a": 2.0, "b": 0.5}}
        self.assertEqual(marginal_contribution(candidate, maxima, weights), 2.0)
        self.assertEqual(marginal_contribution({"rho": 1.0, "term_contributions": {"a": 1.0}}, maxima, weights), 0.0)

    def test_greedy_order_updates_coverage_and_preserves_candidates(self):
        candidates = [
            {"id": "a", "original_rank": 1, "rho": 1, "term_contributions": {"x": 1, "y": 0}},
            {"id": "b", "original_rank": 2, "rho": 1, "term_contributions": {"x": 0, "y": 2}},
            {"id": "c", "original_rank": 3, "rho": 1, "term_contributions": {"x": 0.5, "y": 1}},
        ]
        ordered = evidence_frontload(candidates, {"x": 1, "y": 1})
        self.assertEqual([c["id"] for c in ordered], ["b", "a", "c"])
        self.assertEqual(len(ordered), 3)
        self.assertEqual({c["id"] for c in ordered}, {c["id"] for c in candidates})

    def test_deterministic_tie_uses_original_rank_then_id(self):
        candidates = [
            {"id": "z", "original_rank": 2, "rho": 1, "term_contributions": {"x": 1}},
            {"id": "b", "original_rank": 1, "rho": 1, "term_contributions": {"x": 1}},
            {"id": "a", "original_rank": 1, "rho": 1, "term_contributions": {"x": 1}},
        ]
        self.assertEqual([c["id"] for c in evidence_frontload(candidates, {"x": 1})], ["a", "b", "z"])
        self.assertEqual(
            [c["id"] for c in evidence_frontload(candidates, {"x": 1})],
            [c["id"] for c in evidence_frontload(candidates, {"x": 1})],
        )

    def test_candidate_preservation_and_prefix_budgets(self):
        candidates = [
            {"id": str(i), "original_rank": i + 1, "rho": i / 99, "term_contributions": {"x": i}}
            for i in range(100)
        ]
        ordered = evidence_frontload(candidates, {"x": 1})
        self.assertEqual(len(ordered), 100)
        self.assertEqual(len({c["id"] for c in ordered}), 100)
        self.assertEqual({c["id"] for c in ordered}, {c["id"] for c in candidates})
        for budget in (10, 20, 30, 50, 100):
            self.assertEqual(len(select_prefix(ordered, budget)), budget)

    def test_selector_has_no_qrels_or_labels_input(self):
        parameters = set(inspect.signature(evidence_frontload).parameters)
        self.assertEqual(parameters, {"candidates", "query_weights"})
        self.assertFalse({"qrels", "is_relevant", "mstar", "labels"} & parameters)

    def test_required_results_schema_and_query_budget_cardinality(self):
        self.assertEqual(
            RESULT_FIELDS,
            ("query_id", "budget", "nDCG@10", "Recall@10", "MRR@10", "reranked_pairs", "compression_ratio", "reranking_latency"),
        )
        pairs = {(query_id, budget) for query_id in ("q1", "q2") for budget in BUDGETS}
        self.assertEqual(len(pairs), 2 * 5)


if __name__ == "__main__":
    unittest.main()
