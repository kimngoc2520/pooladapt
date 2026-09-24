"""Run reproducible Phase 1 baselines on a shared SciFact hybrid candidate pool.

Every baseline receives the same BM25 + dense RRF pool. ``--queries 0`` runs
every query in the selected qrels split; the default is a one-query smoke test.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.fixed_prefix import select_fixed_prefix
from src.baselines.full_rerank import rerank_full, select_full_pool
from src.baselines.random_selection import rerank_random, select_random
from src.data import load_beir_dataset
from src.evaluation.cost_metrics import average_reranked_pairs, compression_ratio, reranked_pairs_per_query
from src.evaluation.efficiency_metrics import latency_summary
from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.reranking.cross_encoder import DEFAULT_MODEL_NAME, CrossEncoderReranker, Reranker
from src.retrieval.rrf import fuse_ranked_lists


DEFAULT_BUDGETS = (10, 20, 30, 50)
RAW_RESULT_FIELDS = ["query_id", "method", "budget", "candidate_pool_size", "reranked_pairs", "compression_ratio", "latency_seconds", "ndcg_at_10", "recall_at_10", "mrr_at_10", "top_k_doc_ids"]
SUMMARY_RESULT_FIELDS = ["method", "budget", "mean_ndcg_at_10", "mean_recall_at_10", "mean_mrr_at_10", "mean_reranked_pairs", "mean_compression_ratio", "mean_latency_seconds"]
METHOD_ORDER = {"full_rerank": 0, "fixed_prefix": 1, "random_selection": 2}


def build_candidate_pool(query: str, corpus: Mapping[str, Mapping[str, Any]], bm25: Any, dense: Any, pool_size: int = 100) -> list[dict[str, Any]]:
    """Retrieve, fuse, and enrich the shared pool with Cross-Encoder text."""
    fused = fuse_ranked_lists(
        {"bm25": bm25.retrieve(query, top_n=pool_size), "dense": dense.retrieve(query, top_n=pool_size)}, top_n=pool_size
    )
    return [{**candidate, "title": corpus[candidate["id"]].get("title", ""), "text": corpus[candidate["id"]].get("text", "")} for candidate in fused]


def _quality_metrics(ranked: Sequence[dict[str, Any]], relevance: Mapping[str, int] | None) -> dict[str, float | None]:
    if not relevance:
        return {"ndcg_at_10": None, "recall_at_10": None, "mrr_at_10": None}
    ranked_ids = [str(result["id"]) for result in ranked]
    relevant_ids = [document_id for document_id, score in relevance.items() if score > 0]
    return {"ndcg_at_10": ndcg_at_k(ranked_ids, relevance, k=10), "recall_at_10": recall_at_k(ranked_ids, relevant_ids, k=10), "mrr_at_10": mrr_at_k(ranked_ids, relevant_ids, k=10)}


def _run_one(query_id: str, method: str, budget: int, query: str, pool: Sequence[dict[str, Any]], selected: Sequence[dict[str, Any]], rerank: Any, relevance: Mapping[str, int] | None) -> dict[str, Any]:
    started = time.perf_counter()
    results = rerank()
    elapsed = time.perf_counter() - started
    pairs = reranked_pairs_per_query(selected)
    return {
        "query_id": query_id, "method": method, "budget": budget, "candidate_pool_size": len(pool), "reranked_pairs": pairs,
        "compression_ratio": compression_ratio(len(pool), pairs), "latency_seconds": latency_summary([elapsed])["mean"],
        "top_k_doc_ids": " ".join(str(result["id"]) for result in results), **_quality_metrics(results, relevance),
    }


def run_query_baselines(query_id: str, query: str, pool: Sequence[dict[str, Any]], reranker: Reranker, relevance: Mapping[str, int] | None = None, budgets: Sequence[int] = DEFAULT_BUDGETS, seed: int = 0, top_k: int = 10) -> list[dict[str, Any]]:
    """Run every Phase 1 baseline against one already-built shared pool."""
    if not pool:
        raise ValueError("candidate pool must not be empty")
    full_selected = select_full_pool(pool)
    records = [_run_one(query_id, "full_rerank", len(full_selected), query, pool, full_selected, lambda: rerank_full(query, pool, reranker, top_k=top_k), relevance)]
    for budget in budgets:
        if not 0 <= budget <= len(pool):
            raise ValueError(f"budget {budget} must be between zero and pool size {len(pool)}")
        fixed_selected = select_fixed_prefix(pool, budget)
        records.append(_run_one(query_id, "fixed_prefix", budget, query, pool, fixed_selected, lambda selected=fixed_selected: reranker.rerank(query, selected, top_k=top_k), relevance))
        random_selected = select_random(pool, budget, seed=seed)
        records.append(_run_one(query_id, "random_selection", budget, query, pool, random_selected, lambda: rerank_random(query, pool, budget, reranker, seed=seed, top_k=top_k), relevance))
    return records


def summarize_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate deterministic method-and-budget summaries for CSV export."""
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record["method"]), int(record["budget"])), []).append(record)

    def mean(group: Sequence[Mapping[str, Any]], field: str) -> float | None:
        values = [float(record[field]) for record in group if record[field] is not None]
        return sum(values) / len(values) if values else None

    return [
        {
            "method": method,
            "budget": budget,
            "mean_ndcg_at_10": mean(group, "ndcg_at_10"),
            "mean_recall_at_10": mean(group, "recall_at_10"),
            "mean_mrr_at_10": mean(group, "mrr_at_10"),
            "mean_reranked_pairs": average_reranked_pairs(record["reranked_pairs"] for record in group),
            "mean_compression_ratio": mean(group, "compression_ratio"),
            "mean_latency_seconds": mean(group, "latency_seconds"),
        }
        for (method, budget), group in sorted(
            grouped.items(),
            key=lambda item: (METHOD_ORDER.get(item[0][0], 3), item[0][1]),
        )
    ]


def _write_csv(path: Path, fieldnames: Sequence[str], records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_result_files(records: Sequence[Mapping[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Write stable Phase 1 raw and aggregate CSV files, creating ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "phase1_baselines_scifact.csv"
    summary_path = output_dir / "phase1_summary_scifact.csv"
    sorted_records = sorted(
        records,
        key=lambda record: (
            str(record["query_id"]),
            METHOD_ORDER.get(str(record["method"]), 3),
            int(record["budget"]),
        ),
    )
    _write_csv(raw_path, RAW_RESULT_FIELDS, sorted_records)
    _write_csv(summary_path, SUMMARY_RESULT_FIELDS, summarize_records(sorted_records))
    return raw_path, summary_path


def _write_records(records: Sequence[Mapping[str, Any]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=RAW_RESULT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data" / "scifact")
    parser.add_argument("--queries", type=int, default=1, help="Evaluation queries to run; 0 runs all.")
    parser.add_argument("--qrels-split", default="test")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-encoder-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.queries < 0 or args.top_k < 0:
        parser.error("--queries and --top-k must be non-negative")

    # Keep ``--help`` and fake-reranker unit tests independent of optional
    # retrieval/model packages; real experiment runs load these implementations.
    from src.retrieval import BM25Retriever, DenseRetriever

    corpus, queries, qrels = load_beir_dataset(args.dataset_dir, qrels_split=args.qrels_split)
    query_items = [(query_id, query) for query_id, query in queries.items() if query_id in qrels]
    if args.queries:
        query_items = query_items[: args.queries]
    if not query_items:
        raise SystemExit("No queries matched the selected qrels split.")
    print(f"Loading dense model: {args.dense_model}", file=sys.stderr)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, model_name=args.dense_model)
    reranker = CrossEncoderReranker(model_name=args.cross_encoder_model, top_k=args.top_k)
    records = []
    for query_index, (query_id, query) in enumerate(query_items):
        pool = build_candidate_pool(query, corpus, bm25, dense)
        if len(pool) != 100:
            raise RuntimeError(f"query {query_id} produced {len(pool)} candidates; expected 100")
        records.extend(run_query_baselines(query_id, query, pool, reranker, qrels.get(query_id), seed=args.seed + query_index, top_k=args.top_k))
    _write_records(records)
    raw_path, summary_path = write_result_files(records, args.output_dir)
    print(f"wrote_raw_results={raw_path}", file=sys.stderr)
    print(f"wrote_summary_results={summary_path}", file=sys.stderr)
    print(f"summary,queries={len(query_items)},mean_reranked_pairs={average_reranked_pairs(record['reranked_pairs'] for record in records):.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
