"""Measure SciFact test-relevance coverage in the hybrid Top-100 candidate pool."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_beir_dataset
from src.evaluation.retrieval_metrics import recall_at_k
from src.retrieval import DEFAULT_MODEL, BM25Retriever, DenseRetriever, fuse_ranked_lists


CANDIDATE_POOL_SIZE = 100
TEST_QUERY_COUNT = 300
RANK_BUCKETS = (("rank_1_10", 1, 10), ("rank_11_20", 11, 20), ("rank_21_50", 21, 50), ("rank_51_100", 51, 100))


def parse_arguments() -> argparse.Namespace:
    """Parse diagnostics inputs and output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Dense retriever model name.")
    parser.add_argument("--output", type=Path, default=Path("results/phase1/candidate_pool_diagnostics.csv"))
    return parser.parse_args()


def bucket_for_rank(rank: int | None) -> str:
    """Map a candidate-pool rank to its reported diagnostic bucket."""
    if rank is None:
        return "missing"
    for name, start, end in RANK_BUCKETS:
        if start <= rank <= end:
            return name
    raise ValueError(f"Candidate rank {rank} is outside the Top-{CANDIDATE_POOL_SIZE} pool")


def summarize_pool(pool: Sequence[Mapping[str, Any]], relevant_ids: Mapping[str, int]) -> tuple[float, bool, dict[str, int]]:
    """Calculate recall, hit status, and relevant-document rank buckets for one query."""
    if len(pool) != CANDIDATE_POOL_SIZE:
        raise ValueError(f"Hybrid candidate pool must contain exactly {CANDIDATE_POOL_SIZE} candidates, got {len(pool)}")
    ranks = {str(candidate["id"]): int(candidate["rank"]) for candidate in pool}
    bucket_counts = {name: 0 for name, _, _ in RANK_BUCKETS}
    bucket_counts["missing"] = 0
    for document_id in relevant_ids:
        bucket_counts[bucket_for_rank(ranks.get(str(document_id)))] += 1
    pool_ids = list(ranks)
    return recall_at_k(pool_ids, relevant_ids, k=CANDIDATE_POOL_SIZE), bool(set(pool_ids) & set(relevant_ids)), bucket_counts


def write_csv(path: Path, values: Mapping[str, int | float]) -> None:
    """Write scalar diagnostics in a compact, spreadsheet-friendly CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(values.items())


def main() -> None:
    """Build test-query hybrid pools and report their pre-reranking coverage."""
    args = parse_arguments()
    corpus, queries, qrels = load_beir_dataset(args.dataset_dir, qrels_split="test")
    if len(qrels) != TEST_QUERY_COUNT:
        raise ValueError(f"Expected {TEST_QUERY_COUNT} SciFact test queries, found {len(qrels)}")
    missing_queries = set(qrels) - set(queries)
    if missing_queries:
        raise ValueError(f"Test qrels refer to query IDs absent from queries.jsonl: {sorted(missing_queries)}")

    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, model_name=args.model)
    recall_scores: list[float] = []
    hit_count = 0
    bucket_counts = {name: 0 for name, _, _ in RANK_BUCKETS}
    bucket_counts["missing"] = 0

    for query_id, relevant_ids in qrels.items():
        pool = fuse_ranked_lists(
            {
                "bm25": bm25.retrieve(queries[query_id], top_n=CANDIDATE_POOL_SIZE),
                "dense": dense.retrieve(queries[query_id], top_n=CANDIDATE_POOL_SIZE),
            },
            top_n=CANDIDATE_POOL_SIZE,
        )
        recall, has_hit, query_buckets = summarize_pool(pool, relevant_ids)
        recall_scores.append(recall)
        hit_count += int(has_hit)
        for bucket, count in query_buckets.items():
            bucket_counts[bucket] += count

    total_relevant = sum(bucket_counts.values())
    found_relevant = total_relevant - bucket_counts["missing"]
    values: dict[str, int | float] = {
        "test_query_count": len(qrels),
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "candidate_pool_recall_at_100": sum(recall_scores) / len(recall_scores),
        "pool_hit_rate_at_100": hit_count / len(qrels),
        "total_relevant_documents": total_relevant,
        "relevant_documents_found_in_top_100": found_relevant,
        "relevant_documents_missing_from_top_100": bucket_counts["missing"],
    }
    for name, _, _ in RANK_BUCKETS:
        values[f"{name}_count"] = bucket_counts[name]
        values[f"{name}_percentage"] = bucket_counts[name] / total_relevant if total_relevant else 0.0
    values["missing_count"] = bucket_counts["missing"]
    values["missing_percentage"] = bucket_counts["missing"] / total_relevant if total_relevant else 0.0
    write_csv(args.output, values)

    print(f"Test queries: {values['test_query_count']}")
    print(f"Candidate pool size: {values['candidate_pool_size']}")
    print(f"Candidate Pool Recall@100: {values['candidate_pool_recall_at_100']:.6f}")
    print(f"Pool Hit Rate@100: {values['pool_hit_rate_at_100']:.6f}")
    print(f"Relevant documents: {values['total_relevant_documents']} total, {found_relevant} found, {bucket_counts['missing']} missing")
    print("Relevant-document rank distribution:")
    for name, _, _ in RANK_BUCKETS:
        print(f"  {name.removeprefix('rank_').replace('_', '-')}: {bucket_counts[name]} ({values[f'{name}_percentage']:.2%})")
    print(f"  missing: {bucket_counts['missing']} ({values['missing_percentage']:.2%})")
    print(f"Saved diagnostics: {args.output}")


if __name__ == "__main__":
    main()
