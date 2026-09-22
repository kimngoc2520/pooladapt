import unittest

from src.pooladapt.features import characterize_candidates


class PoolCharacterizationTests(unittest.TestCase):
    def test_rank_gap_is_recorded_when_both_sources_retrieve(self) -> None:
        result = characterize_candidates([{"id": "a", "bm25_rank": 2, "dense_rank": 5}])
        self.assertEqual(result[0]["rank_gap"], 3)
