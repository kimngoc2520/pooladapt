"""Run Phase 1 SciFact hybrid-retrieval reranking baselines."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.fixed_prefix import select
from src.baselines.full_rerank import rerank_full
from src.baselines.random_selection import rerank_random
from src.data import load_beir_dataset
from src.reranking import CrossEncoderReranker
from src.retrieval import DEFAULT_MODEL, BM25Retriever, DenseRetriever, fuse_ranked_lists


CANDIDATE_POOL_SIZE = 100


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for a baseline experiment run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=0, help="Number of queries to run; 0 runs all queries.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Dense retriever model name.")
    parser.add_argument("--top-k", type=int, default=10, help="Final results retained per query.")
    parser.add_argument("--prefix-budgets", type=int, nargs="+", default=[10, 20, 30, 50])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/phase1/baseline_predictions.json"))
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid experiment settings before models are loaded."""
    if args.queries < 0:
        raise ValueError("--queries must be non-negative")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if not args.prefix_budgets:
        raise ValueError("--prefix-budgets must contain at least one budget")
    if any(not 0 <= budget <= CANDIDATE_POOL_SIZE for budget in args.prefix_budgets):
        raise ValueError(f"--prefix-budgets must be between 0 and {CANDIDATE_POOL_SIZE}")


def enrich_candidates(candidates: Sequence[dict[str, Any]], corpus: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach corpus fields required by the cross-encoder to fused candidates."""
    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        document_id = str(candidate["id"])
        if document_id not in corpus:
            raise KeyError(f"Fused candidate {document_id!r} is absent from the corpus")
        result = dict(corpus[document_id])
        result.update(candidate)
        result["id"] = document_id
        enriched.append(result)
    return enriched


def document_ids(ranked_candidates: Sequence[dict[str, Any]]) -> list[str]:
    """Return JSON-safe document IDs from a reranked result list."""
    return [str(candidate["id"]) for candidate in ranked_candidates]


def main() -> None:
    """Execute all Phase 1 baselines and save query-level predictions."""
    args = parse_arguments()
    validate_arguments(args)

    corpus, queries, _ = load_beir_dataset(args.dataset_dir)
    query_items = list(queries.items())
    if args.queries:
        query_items = query_items[: args.queries]

    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, model_name=args.model)
    reranker = CrossEncoderReranker()
    methods: dict[str, dict[str, Any]] = {"full_rerank": {"predictions": []}}
    for budget in args.prefix_budgets:
        methods[f"fixed_prefix_{budget}"] = {"predictions": []}
        methods[f"random_{budget}"] = {"predictions": []}

    for query_id, query in query_items:
        fused = fuse_ranked_lists(
            {
                "bm25": bm25.retrieve(query, top_n=CANDIDATE_POOL_SIZE),
                "dense": dense.retrieve(query, top_n=CANDIDATE_POOL_SIZE),
            },
            top_n=CANDIDATE_POOL_SIZE,
        )
        if len(fused) != CANDIDATE_POOL_SIZE:
            raise ValueError(
                f"Query {query_id!r} produced {len(fused)} hybrid candidates; "
                f"Phase 1 requires exactly {CANDIDATE_POOL_SIZE}."
            )
        candidates = enrich_candidates(fused, corpus)

        full_results = rerank_full(query, candidates, reranker, top_k=args.top_k)
        methods["full_rerank"]["predictions"].append(
            {"query_id": str(query_id), "document_ids": document_ids(full_results), "reranked_pairs": len(candidates)}
        )

        for budget in args.prefix_budgets:
            prefix_results = reranker.rerank(query, select(candidates, budget), top_k=args.top_k)
            methods[f"fixed_prefix_{budget}"]["predictions"].append(
                {"query_id": str(query_id), "document_ids": document_ids(prefix_results), "reranked_pairs": budget}
            )
            random_results = rerank_random(query, candidates, budget, reranker, seed=args.seed, top_k=args.top_k)
            methods[f"random_{budget}"]["predictions"].append(
                {"query_id": str(query_id), "document_ids": document_ids(random_results), "reranked_pairs": budget}
            )

    output = {
        "dataset": "scifact",
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "top_k": args.top_k,
        "seed": args.seed,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
