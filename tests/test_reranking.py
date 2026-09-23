import unittest

from src.baselines.full_rerank import rerank_full, select_full_pool
from src.baselines.random_selection import rerank_random, select_random
from src.evaluation.cost_metrics import average_reranked_pairs, compression_ratio, reranked_pairs_per_query
from src.evaluation.efficiency_metrics import latency_summary, measure_latency
from src.reranking.cross_encoder import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.pairs = []

    def predict(self, pairs):
        self.pairs = list(pairs)
        return [float(index) for index, _ in enumerate(self.pairs)]


class RerankingBaselineTests(unittest.TestCase):
    def test_full_pool_keeps_every_candidate(self) -> None:
        candidates = [{"id": "a"}, {"id": "b"}]
        self.assertEqual(select_full_pool(candidates), candidates)

    def test_cross_encoder_scores_and_limits_candidates(self) -> None:
        model = FakeCrossEncoder()
        reranker = CrossEncoderReranker(model=model, top_k=2)
        results = reranker.rerank("query", [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}, {"id": "c", "text": "C"}])
        self.assertEqual([result["id"] for result in results], ["c", "b"])
        self.assertEqual(len(model.pairs), 3)
        self.assertTrue(all("reranker_score" in result for result in results))

    def test_full_rerank_sends_all_100_candidates_to_model(self) -> None:
        model = FakeCrossEncoder()
        candidates = [{"id": str(index), "text": str(index)} for index in range(100)]
        rerank_full("query", candidates, CrossEncoderReranker(model=model, top_k=10))
        self.assertEqual(len(model.pairs), 100)

    def test_random_selection_is_seeded_unique_and_bounded(self) -> None:
        candidates = [{"id": str(index)} for index in range(100)]
        first = select_random(candidates, 20, seed=7)
        second = select_random(candidates, 20, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len({candidate["id"] for candidate in first}), 20)

    def test_random_rerank_only_scores_selected_budget(self) -> None:
        model = FakeCrossEncoder()
        candidates = [{"id": str(index), "text": str(index)} for index in range(100)]
        rerank_random("query", candidates, 10, CrossEncoderReranker(model=model), seed=3, top_k=5)
        self.assertEqual(len(model.pairs), 10)

    def test_cost_and_latency_metrics(self) -> None:
        self.assertEqual(reranked_pairs_per_query(["a", "b"]), 2)
        self.assertEqual(average_reranked_pairs([10, 20]), 15)
        self.assertEqual(compression_ratio(100, 20), 0.8)
        summary = latency_summary([1.0, 2.0, 3.0])
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["p50"], 2.0)
        self.assertGreaterEqual(measure_latency(lambda: None)["mean"], 0.0)
