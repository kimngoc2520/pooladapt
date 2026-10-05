import unittest

import numpy as np

from src.pooladapt.features import characterize_candidates
from src.pooladapt.selector import BUDGET_GRID, FROZEN_FEATURES, PoolAdaptSelector, budget_bias, calibrated_logits, select_rrf_prefix, validate_feature_names


class PoolCharacterizationTests(unittest.TestCase):
    def test_rank_gap_is_recorded_when_both_sources_retrieve(self) -> None:
        result = characterize_candidates([{"id": "a", "bm25_rank": 2, "dense_rank": 5}])
        self.assertEqual(result[0]["rank_gap"], 3)

    def test_grid_bias_logits_and_tie_break(self) -> None:
        self.assertEqual(BUDGET_GRID, (10, 20, 30, 50, 100))
        np.testing.assert_allclose(budget_bias(), -(np.asarray(BUDGET_GRID)-np.mean(BUDGET_GRID))/np.std(BUDGET_GRID))
        logits=np.arange(5, dtype=float)
        np.testing.assert_allclose(calibrated_logits(logits,2,3),logits/2+3*budget_bias())
        class Uniform:
            classes_=np.asarray(BUDGET_GRID)
            def predict_proba(self, values): return np.asarray([[.2]*5])
        row={name: 1.0 for name in FROZEN_FEATURES}
        selector=PoolAdaptSelector(Uniform())
        self.assertEqual(selector.predict(row),10)
        self.assertEqual(selector.predict(row,budget_bias_lambda=2),10)
        self.assertEqual(selector.predict(row,budget_bias_lambda=-2),100)

    def test_frozen_features_and_prefix_only(self) -> None:
        self.assertEqual(validate_feature_names(FROZEN_FEATURES), FROZEN_FEATURES)
        with self.assertRaises(ValueError): validate_feature_names(FROZEN_FEATURES[:-1])
        with self.assertRaises(ValueError): validate_feature_names(("qrels",))
        candidates=[{"id":str(i),"rank":i} for i in range(100)]
        selected=select_rrf_prefix(candidates,30)
        self.assertEqual(selected,candidates[:30])
        self.assertEqual([x["id"] for x in selected], [str(i) for i in range(30)])
        with self.assertRaises(ValueError): select_rrf_prefix(candidates,11)
