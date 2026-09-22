import unittest

from src.baselines.full_rerank import select_full_pool


class RerankingBaselineTests(unittest.TestCase):
    def test_full_pool_keeps_every_candidate(self) -> None:
        candidates = [{"id": "a"}, {"id": "b"}]
        self.assertEqual(select_full_pool(candidates), candidates)
