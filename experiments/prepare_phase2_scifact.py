"""Create the locked SciFact development split and Oracle-ready reranking outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.development_reranking import METHOD_BUDGETS, validate_artifact
from src.baselines.fixed_prefix import select
from src.baselines.full_rerank import rerank_full
from src.data import load_beir_dataset
from src.data.scifact_phase2 import prepare_split, read_qrel_rows, query_ids
from src.reranking import CrossEncoderReranker
from src.reranking.cross_encoder import DEFAULT_MODEL_NAME
from src.retrieval import BM25Retriever, DEFAULT_MODEL, DenseRetriever, fuse_ranked_lists


def enrich(candidates: list[dict[str, Any]], corpus: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**corpus[str(candidate["id"])], **candidate, "id": str(candidate["id"])} for candidate in candidates]


def verify_retrievers(corpus: dict[str, dict[str, Any]], bm25: BM25Retriever, dense: DenseRetriever) -> None:
    corpus_ids = set(corpus)
    if set(bm25.document_ids) != corpus_ids:
        raise ValueError("BM25 index does not cover the complete SciFact corpus")
    if set(dense.document_ids) != corpus_ids or len(dense._document_embeddings) != len(corpus):
        raise ValueError("Dense index does not cover the complete SciFact corpus")


def rerank_split(split: str, query_ids_for_split: list[str], qrels_path: Path, corpus: dict[str, dict[str, Any]], queries: dict[str, str], bm25: BM25Retriever, dense: DenseRetriever, reranker: CrossEncoderReranker, model_name: str) -> dict[str, Any]: 
    qrel_ids = query_ids(read_qrel_rows(qrels_path)) 
    expected_ids = set(query_ids_for_split) 
    if qrel_ids != expected_ids or not expected_ids <= set(queries): 
        raise ValueError(f"{split} split IDs do not match qrels/queries") 
    methods = {name: {"predictions": []} for name in METHOD_BUDGETS} 

    for query_id in tqdm(query_ids_for_split, desc=f"Reranking {split}", unit="query"): 
        query = queries[query_id] 
        pool = fuse_ranked_lists(
            {
                "bm25": bm25.retrieve(query, top_n=100),
                "dense": dense.retrieve(query, top_n=100)
            },
            top_n=100
        ) 

        if len(pool) != 100 or len({candidate["id"] for candidate in pool}) != 100: 
            raise ValueError(
                f"{split} query {query_id}: expected 100 unique hybrid candidates"
            ) 

        candidates = enrich(pool, corpus) 

        full = rerank_full(query, candidates, reranker, top_k=10) 
        methods["full_rerank"]["predictions"].append(
            _record(query_id, full, 100, reranker)
        ) 

        for budget in (10, 20, 30, 50): 
            ranked = reranker.rerank(
                query,
                select(candidates, budget),
                top_k=10
            ) 
            methods[f"fixed_prefix_{budget}"]["predictions"].append(
                _record(query_id, ranked, budget, reranker)
            )

        methods["fixed_prefix_100"]["predictions"].append(
            _record(query_id, full, 100, reranker)
        )

    artifact: dict[str, Any] = {
        "dataset": "scifact",
        "split": split,
        "num_queries": len(query_ids_for_split),
        "candidate_pool_size": 100,
        "top_k": 10,
        "metadata": {
            "dataset": "SciFact",
            "split": split,
            "source_split": "original_train",
            "candidate_pool_size": 100,
            "retrieval_method": "BM25 + Dense + RRF",
            "dense_model": model_name,
            "reranker_model": DEFAULT_MODEL_NAME,
            "methods": {
                name: {
                    "K": budget,
                    "method": (
                        "Full Rerank"
                        if name == "full_rerank"
                        else "Fixed Prefix"
                    )
                }
                for name, budget in METHOD_BUDGETS.items()
            }
        },
        "methods": methods
    }

    comparison = validate_artifact(artifact, expected_ids, split) 
    artifact["metadata"]["full_rerank_vs_fixed_prefix_100"] = comparison 
    return artifact

def _record(query_id: str, ranked: list[dict[str, Any]], budget: int, reranker: CrossEncoderReranker) -> dict[str, Any]:
    return {"query_id": str(query_id), "document_ids": [str(candidate["id"]) for candidate in ranked], "reranked_pairs": budget, "latency_rerank_seconds": float(reranker.last_latency_rerank_seconds), "latency_ce_seconds": float(reranker.last_latency_ce_seconds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2/development_reranking"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prepare-only", action="store_true", help="Create/verify locked split and qrels without model inference.")
    args = parser.parse_args()
    split = prepare_split(args.dataset_dir)
    print(f"Locked split verified: train={split['train_query_count']}, validation={split['validation_query_count']}")
    if args.prepare_only:
        return
    corpus, queries, _ = load_beir_dataset(args.dataset_dir)
    bm25, dense = BM25Retriever(corpus), DenseRetriever(corpus, model_name=args.model)
    verify_retrievers(corpus, bm25, dense)
    reranker = CrossEncoderReranker()
    reranker.warm_up()
    qrels_dir = args.dataset_dir / "qrels"
    for name, ids, qrels_name in (("train", split["train_ids"], "train_split.tsv"), ("validation", split["validation_ids"], "validation.tsv")):
        artifact = rerank_split(name, ids, qrels_dir / qrels_name, corpus, queries, bm25, dense, reranker, args.model)
        output = args.output_dir / name / "reranking_results.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
