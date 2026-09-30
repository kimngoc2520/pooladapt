import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.development_reranking import METHOD_BUDGETS, validate_artifact
from src.data.scifact_phase2 import SPLIT_SEED, prepare_split, split_original_train, validate_split


class SciFactPhase2PreparationTests(unittest.TestCase):
    def test_fixed_seed_ratio_and_complete_disjoint_assignment(self):
        source = {str(index) for index in range(809)}
        train, validation = split_original_train(source)
        self.assertEqual(SPLIT_SEED, 42)
        self.assertEqual((len(train), len(validation)), (647, 162))
        self.assertEqual((train, validation), split_original_train(source))
        validate_split(set(train), set(validation), source, {"locked-test"})
        with self.assertRaises(ValueError):
            validate_split({"a"}, {"a"}, {"a"}, set())

    def test_split_qrels_are_filtered_and_lock_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scifact"
            (root / "qrels").mkdir(parents=True)
            train_rows = "query-id\tcorpus-id\tscore\n" + "\n".join(f"{index}\td{index}\t1" for index in range(10)) + "\n"
            (root / "qrels" / "train.tsv").write_text(train_rows, encoding="utf-8")
            (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nlocked\td\t1\n", encoding="utf-8")
            metadata = prepare_split(root)
            self.assertEqual((metadata["train_query_count"], metadata["validation_query_count"]), (8, 2))
            self.assertEqual((root / "qrels" / "train.tsv").read_text(encoding="utf-8"), train_rows)
            self.assertEqual((root / "qrels" / "train_split.tsv").read_text(encoding="utf-8").splitlines()[0], "query-id\tcorpus-id\tscore")
            self.assertEqual(metadata, prepare_split(root))
            locked = json.loads((root / "splits" / "split_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(locked["random_seed"], 42)

    def test_artifact_schema_alignment_latency_and_k100_comparison(self):
        expected = {"q1", "q2"}
        methods = {}
        for name, budget in METHOD_BUDGETS.items():
            methods[name] = {"predictions": [{"query_id": query_id, "document_ids": [f"{query_id}-d"], "reranked_pairs": budget, "latency_rerank_seconds": 0.1, "latency_ce_seconds": 0.05} for query_id in sorted(expected)]}
        artifact = {"dataset": "scifact", "split": "train", "candidate_pool_size": 100, "top_k": 10, "methods": methods}
        result = validate_artifact(artifact, expected, "train")
        self.assertTrue(result["full_rerank_equals_fixed_prefix_100"])
        methods["fixed_prefix_100"]["predictions"][0]["document_ids"] = ["different"]
        self.assertFalse(validate_artifact(artifact, expected, "train")["full_rerank_equals_fixed_prefix_100"])
        methods["fixed_prefix_10"]["predictions"].append(methods["fixed_prefix_10"]["predictions"][0].copy())
        with self.assertRaises(ValueError):
            validate_artifact(artifact, expected, "train")
