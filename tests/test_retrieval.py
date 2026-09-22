import unittest

from src.retrieval.rrf import fuse_ranked_lists


class RRFTests(unittest.TestCase):
    def test_fusion_records_ranks_and_limits_pool(self) -> None:
        pool = fuse_ranked_lists({"bm25": [{"id": "a", "score": 2.0}], "dense": [{"id": "a", "score": 1.0}, {"id": "b"}]}, top_n=1)
        self.assertEqual(pool[0]["id"], "a")
        self.assertEqual(pool[0]["bm25_rank"], 1)
        self.assertEqual(pool[0]["dense_rank"], 1)
