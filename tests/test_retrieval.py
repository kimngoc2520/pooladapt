import unittest

import numpy as np

from src.baselines.fixed_prefix import select
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.rrf import fuse_ranked_lists


CORPUS = {
    "a": {"title": "Alpha", "text": "cats and dogs"},
    "b": {"title": "Beta", "text": "dogs only"},
    "c": {"title": "Gamma", "text": "birds only"},
}


class FakeEncoder:
    def encode(self, texts, **_: object):
        return np.asarray([[len(text), sum(map(ord, text)) % 17] for text in texts], dtype=np.float32)


class RRFTests(unittest.TestCase):
    def test_fusion_records_ranks_and_limits_pool(self) -> None:
        pool = fuse_ranked_lists({"bm25": [{"id": "a", "score": 2.0}], "dense": [{"id": "a", "score": 1.0}, {"id": "b"}]}, top_n=1)
        self.assertEqual(pool[0]["id"], "a")
        self.assertEqual(pool[0]["bm25_rank"], 1)
        self.assertEqual(pool[0]["dense_rank"], 1)

    def test_rrf_returns_at_most_100_unique_ranked_ids(self) -> None:
        candidates = [{"id": str(index), "score": float(index)} for index in range(150)]
        pool = fuse_ranked_lists({"bm25": candidates, "dense": list(reversed(candidates))})
        self.assertEqual(len(pool), 100)
        self.assertEqual(len({candidate["id"] for candidate in pool}), 100)
        self.assertEqual([candidate["rank"] for candidate in pool], list(range(1, 101)))


class RetrieverTests(unittest.TestCase):
    def test_bm25_returns_at_most_100_documents_with_scores_and_ranks(self) -> None:
        results = BM25Retriever(CORPUS).retrieve("dogs", top_n=100)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["rank"], 1)
        self.assertIn("score", results[0])

    def test_dense_returns_at_most_100_documents(self) -> None:
        results = DenseRetriever(CORPUS, model=FakeEncoder()).retrieve("dogs", top_n=100)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["rank"], 1)


class FixedPrefixTests(unittest.TestCase):
    def test_selects_requested_prefix_in_hybrid_order(self) -> None:
        candidates = [{"id": str(index)} for index in range(100)]
        selected = select(candidates, m=20)
        self.assertEqual(selected, candidates[:20])

    def test_never_selects_more_than_hybrid_pool_size(self) -> None:
        candidates = [{"id": str(index)} for index in range(150)]
        self.assertEqual(len(select(candidates, m=150)), 100)
