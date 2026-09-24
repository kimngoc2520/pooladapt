import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_baselines import _quality_metrics, build_candidate_pool, build_parser, run_query_baselines, summarize_records, write_result_files
from src.data import load_beir_dataset


class FakeRetriever:
    def __init__(self, document_ids):
        self.document_ids = document_ids

    def retrieve(self, query, top_n=100):
        return [{"id": document_id, "score": float(top_n - index)} for index, document_id in enumerate(self.document_ids[:top_n])]


class FakeReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_k=None):
        candidates = list(candidates)
        self.calls.append((query, candidates))
        return [{**candidate, "score": float(index)} for index, candidate in enumerate(reversed(candidates))][:top_k]


class BaselineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.corpus = {str(index): {"title": f"Title {index}", "text": f"Document {index}"} for index in range(100)}
        self.pool = build_candidate_pool("query", self.corpus, FakeRetriever(list(self.corpus)), FakeRetriever(list(reversed(self.corpus))))

    def test_candidate_pool_is_shared_and_has_cross_encoder_text(self):
        self.assertEqual(len(self.pool), 100)
        self.assertEqual(len({candidate["id"] for candidate in self.pool}), 100)
        self.assertTrue(all(candidate["text"].startswith("Document") for candidate in self.pool))

    def test_baselines_score_only_their_selected_budgets(self):
        reranker = FakeReranker()
        records = run_query_baselines("q1", "query", self.pool, reranker, {"0": 1}, budgets=(10, 20, 30, 50), seed=7)
        self.assertEqual(len(records), 9)
        self.assertEqual([len(candidates) for _, candidates in reranker.calls], [100, 10, 10, 20, 20, 30, 30, 50, 50])
        self.assertEqual([record["reranked_pairs"] for record in records], [100, 10, 10, 20, 20, 30, 30, 50, 50])
        self.assertEqual(records[1]["compression_ratio"], 0.9)
        self.assertEqual(records[7]["compression_ratio"], 0.5)
        self.assertTrue(all(record["ndcg_at_10"] is not None for record in records))

    def test_random_selection_is_reproducible(self):
        first, second = FakeReranker(), FakeReranker()
        run_query_baselines("q1", "query", self.pool, first, budgets=(10,), seed=9)
        run_query_baselines("q1", "query", self.pool, second, budgets=(10,), seed=9)
        self.assertEqual(first.calls[2][1], second.calls[2][1])

    def test_beir_qrels_and_ranked_ids_are_normalized_for_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qrels").mkdir()
            (root / "corpus.jsonl").write_text(json.dumps({"_id": "31715818", "text": "document"}) + "\n", encoding="utf-8")
            (root / "queries.jsonl").write_text(json.dumps({"_id": 1, "text": "query"}) + "\n", encoding="utf-8")
            (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\n1\t31715818\t1\n", encoding="utf-8")
            _, queries, qrels = load_beir_dataset(root, qrels_split="test")

        self.assertIn("1", queries)
        self.assertEqual(qrels, {"1": {"31715818": 1}})
        metrics = _quality_metrics([{"id": 31715818}], qrels["1"])
        self.assertEqual(metrics, {"ndcg_at_10": 1.0, "recall_at_10": 1.0, "mrr_at_10": 1.0})

    def test_writes_raw_and_summary_csv_to_requested_output_directory(self):
        records = run_query_baselines("q1", "query", self.pool, FakeReranker(), {"0": 1}, budgets=(10, 20), seed=7)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "custom_results"
            raw_path, summary_path = write_result_files(records, output_dir)
            with raw_path.open(encoding="utf-8", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            with summary_path.open(encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))

        self.assertEqual(raw_path.parent, output_dir)
        self.assertEqual(len(raw_rows), 5)
        self.assertEqual(list(raw_rows[0]), ["query_id", "method", "budget", "candidate_pool_size", "reranked_pairs", "compression_ratio", "latency_seconds", "ndcg_at_10", "recall_at_10", "mrr_at_10", "top_k_doc_ids"])
        self.assertEqual(raw_rows[0]["budget"], "100")
        self.assertEqual(raw_rows[0]["reranked_pairs"], "100")
        self.assertEqual(len(summary_rows), 5)
        self.assertEqual(summary_rows[0]["mean_reranked_pairs"], "100.0")

    def test_summary_aggregation_averages_matching_method_and_budget(self):
        summaries = summarize_records([
            {"method": "fixed_prefix", "budget": 10, "ndcg_at_10": 0.2, "recall_at_10": 0.4, "mrr_at_10": 0.5, "reranked_pairs": 10, "compression_ratio": 0.9, "latency_seconds": 2.0},
            {"method": "fixed_prefix", "budget": 10, "ndcg_at_10": 0.6, "recall_at_10": 0.8, "mrr_at_10": 1.0, "reranked_pairs": 10, "compression_ratio": 0.9, "latency_seconds": 4.0},
        ])
        self.assertEqual(summaries, [{"method": "fixed_prefix", "budget": 10, "mean_ndcg_at_10": 0.4, "mean_recall_at_10": 0.6000000000000001, "mean_mrr_at_10": 0.75, "mean_reranked_pairs": 10.0, "mean_compression_ratio": 0.9, "mean_latency_seconds": 3.0}])

    def test_output_dir_argument_is_parsed(self):
        self.assertEqual(build_parser().parse_args(["--output-dir", "results/test_run"]).output_dir, Path("results/test_run"))
