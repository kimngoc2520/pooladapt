import csv
import tempfile
import unittest
from pathlib import Path

from src.analysis.correlation import analyze, holm_adjust, load_feature_sets, multicollinearity


class FeatureMstarAnalysisTests(unittest.TestCase):
    def _write(self, path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_candidate_mean_aggregation_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = []
            for query_id, offset in (("q1", 0), ("q2", 10), ("q3", 20)):
                for index in range(100):
                    candidates.append({"query_id": query_id, "doc_id": f"{query_id}-{index}", "bm25_rank": offset + index, "dense_rank": index, "rank_displacement": offset, "bm25_score": index, "dense_score": index, "local_rrf_margin": .1})
            self._write(root / "candidate.csv", list(candidates[0]), candidates)
            self._write(root / "pool.csv", ["query_id", "score_entropy"], [{"query_id": q, "score_entropy": i} for i, q in enumerate(("q1", "q2", "q3"))])
            self._write(root / "query.csv", ["query_id", "query_length", "entity_count"], [{"query_id": q, "query_length": i + 1, "entity_count": ""} for i, q in enumerate(("q1", "q2", "q3"))])
            features = load_feature_sets(root / "candidate.csv", root / "pool.csv", root / "query.csv")
            self.assertEqual(features["candidate"]["q1"]["bm25_rank"], 49.5)
            rows = analyze(features, [{"query_id": q, "M_star": str((i + 1) * 10), "tau": "0.95"} for i, q in enumerate(("q1", "q2", "q3"))], "095")
            candidate_row = next(row for row in rows if row["feature_level"] == "candidate" and row["feature_name"] == "bm25_rank")
            self.assertEqual(candidate_row["aggregation_method"], "mean_N100")
            self.assertEqual(candidate_row["number_of_rows_used"], candidate_row["number_of_unique_queries"])
            self.assertNotIn("entity_count", features["query"]["q1"])

    def test_candidate_pool_size_and_duplicate_mstar_assertions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root / "candidate.csv", ["query_id", "doc_id", "bm25_rank"], [{"query_id": "q", "doc_id": "d", "bm25_rank": 1}])
            self._write(root / "pool.csv", ["query_id", "x"], [{"query_id": "q", "x": 1}])
            self._write(root / "query.csv", ["query_id", "x"], [{"query_id": "q", "x": 1}])
            with self.assertRaises(ValueError):
                load_feature_sets(root / "candidate.csv", root / "pool.csv", root / "query.csv")
        with self.assertRaises(ValueError):
            analyze({"pool": {"q": {"query_id": "q", "x": 1}}}, [{"query_id": "q", "M_star": "10", "tau": "0.95"}, {"query_id": "q", "M_star": "20", "tau": "0.95"}], "095")

    def test_spearman_holm_selection_and_multicollinearity(self):
        features = {"candidate": {"q1": {"query_id": "q1", "x": 1}, "q2": {"query_id": "q2", "x": 2}, "q3": {"query_id": "q3", "x": 3}}}
        labels = [{"query_id": f"q{i}", "M_star": str(i * 10), "tau": "0.95"} for i in range(1, 4)]
        result = analyze(features, labels, "095")[0]
        self.assertEqual(result["spearman_rho"], 1.0)
        self.assertTrue(result["selected"])
        self.assertEqual(holm_adjust([.01, .04]), [.02, .04])
        correlated = multicollinearity({"pool": {"q1": {"query_id": "q1", "a": 1, "b": 2}, "q2": {"query_id": "q2", "a": 2, "b": 4}, "q3": {"query_id": "q3", "a": 3, "b": 6}}})
        self.assertEqual(len(correlated), 1)
