"""Run adapted PACE-EF on the Phase 2 train + validation population."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.pace_ef import (
    bm25_document_contribution,
    evidence_frontload,
    normalize_rrf_scores,
    select_prefix,
)
from src.data import load_beir_dataset
from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.reranking import CrossEncoderReranker
from src.retrieval.bm25 import BM25Retriever, tokenize


POOL_SIZE = 100
BUDGETS = (10, 20, 30, 50, 100)
RESULT_FIELDS = (
    "query_id", "budget", "nDCG@10", "Recall@10", "MRR@10",
    "reranked_pairs", "compression_ratio", "reranking_latency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--candidate-log", type=Path, default=Path("results/phase2/01_candidate_logging/candidate_features.csv"))
    parser.add_argument("--query-ids", type=Path, default=Path("results/phase2/01_candidate_logging/train_validation_query_ids.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2/08_pace_ef/pace_ef_results.csv"))
    parser.add_argument("--audit-output", type=Path, default=Path("results/phase2/08_pace_ef/pace_ef_ordering_audit.csv"))
    parser.add_argument("--queries", type=int, default=0, help="Optional development subset for smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_corpus, queries, _ = load_beir_dataset(args.dataset_dir)
    corpus = {
        doc_id: {"title": str(document.get("title", "")), "text": str(document.get("text", ""))}
        for doc_id, document in raw_corpus.items()
    }
    with args.query_ids.open(encoding="utf-8-sig", newline="") as handle:
        query_ids = [row["query_id"] for row in csv.DictReader(handle)]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("development query ID list contains duplicates")
    if not set(query_ids) <= set(queries):
        raise ValueError("development IDs must be present in query data")
    if args.queries:
        if args.queries < 0:
            raise ValueError("--queries must be non-negative")
        query_ids = query_ids[:args.queries]

    logged: dict[str, list[dict[str, str]]] = {}
    with args.candidate_log.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["query_id"] in set(query_ids):
                logged.setdefault(row["query_id"], []).append(row)
    bm25 = BM25Retriever(corpus)
    reranker = CrossEncoderReranker()
    reranker.warm_up()
    result_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for query_id in query_ids:
        pool_rows = sorted(logged.get(query_id, []), key=lambda row: int(row["rrf_rank"]))
        if len(pool_rows) != POOL_SIZE or len({row["doc_id"] for row in pool_rows}) != POOL_SIZE:
            raise ValueError(f"{query_id}: expected 100 unique logged RRF candidates")
        query_terms = list(dict.fromkeys(tokenize(queries[query_id])))
        query_weights = {
            term: float(bm25._index.idf[term])
            for term in query_terms
            if term in bm25._index.idf
        }
        rrfs = [float(row["rrf_score"]) for row in pool_rows]
        rhos = normalize_rrf_scores(rrfs)
        selector_candidates = []
        for row, rho in zip(pool_rows, rhos):
            doc_id = row["doc_id"]
            if doc_id not in corpus:
                raise KeyError(f"candidate {doc_id} is missing from corpus")
            doc_tokens = tokenize(f"{corpus[doc_id].get('title', '')} {corpus[doc_id].get('text', '')}")
            frequencies = Counter(doc_tokens)
            term_contributions = {
                term: bm25_document_contribution(
                    frequencies[term], len(doc_tokens), bm25._index.k1,
                    bm25._index.b, bm25._index.avgdl,
                )
                for term in query_weights
            }
            candidate = dict(corpus[doc_id])
            candidate.update({
                "id": doc_id,
                "rrf_score": float(row["rrf_score"]),
                "original_rank": int(row["rrf_rank"]),
                "rho": rho,
                "term_contributions": term_contributions,
            })
            selector_candidates.append(candidate)

        ordered = evidence_frontload(selector_candidates, query_weights)
        ordered_ids = [str(candidate["id"]) for candidate in ordered]
        original_ids = {row["doc_id"] for row in pool_rows}
        if len(ordered) != POOL_SIZE or len(set(ordered_ids)) != POOL_SIZE or set(ordered_ids) != original_ids:
            raise ValueError(f"{query_id}: Evidence Frontloading did not preserve the exact RRF pool")
        for rank, candidate in enumerate(ordered, start=1):
            audit_rows.append({
                "query_id": query_id,
                "original_rank": candidate["original_rank"],
                "frontloaded_rank": rank,
                "document_id": candidate["id"],
                "marginal_contribution": candidate["marginal_contribution"],
            })

        for budget in BUDGETS:
            selected = select_prefix(ordered, budget)
            reranked = reranker.rerank(queries[query_id], selected, top_k=10)
            ranked_ids = [str(candidate["id"]) for candidate in reranked]
            predictions.append({
                "query_id": query_id,
                "budget": budget,
                "document_ids": ranked_ids,
                "reranking_latency": reranker.last_latency_rerank_seconds,
            })

    # Relevance data enters only after candidate ordering and every CE call finish.
    _, _, qrels = load_beir_dataset(args.dataset_dir, qrels_split="train")
    if not set(query_ids) <= set(qrels):
        raise ValueError("development IDs must be present in train qrels for evaluation")
    for prediction in predictions:
        query_id, budget = prediction["query_id"], prediction["budget"]
        ranked_ids, relevance = prediction["document_ids"], qrels[query_id]
        result_rows.append({
            "query_id": query_id,
            "budget": budget,
            "nDCG@10": ndcg_at_k(ranked_ids, relevance, k=10),
            "Recall@10": recall_at_k(ranked_ids, relevance, k=10),
            "MRR@10": mrr_at_k(ranked_ids, relevance, k=10),
            "reranked_pairs": budget,
            "compression_ratio": 1.0 - budget / POOL_SIZE,
            "reranking_latency": prediction["reranking_latency"],
        })
    if len(result_rows) != len(query_ids) * len(BUDGETS):
        raise ValueError("result row count does not match the development queries and budgets")
    expected_pairs = {(query_id, budget) for query_id in query_ids for budget in BUDGETS}
    actual_pairs = {(row["query_id"], row["budget"]) for row in result_rows}
    if actual_pairs != expected_pairs:
        raise ValueError("results must contain exactly one row per query and budget")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(result_rows)
    with args.audit_output.open("w", encoding="utf-8", newline="") as handle:
        fields = ("query_id", "original_rank", "frontloaded_rank", "document_id", "marginal_contribution")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"Wrote {len(result_rows)} rows ({len(query_ids)} queries x {len(BUDGETS)} budgets) to {args.output}")


if __name__ == "__main__":
    main()
