"""Evaluate Phase 1 SciFact baseline predictions against BEIR test qrels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_beir_dataset
from src.evaluation.cost_metrics import (
    average_reranked_pairs,
    compression_ratio,
    reranked_pairs_per_query,
    total_reranking_pairs,
)
from src.evaluation.latency_metrics import build_latency_report
from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k


CANDIDATE_POOL_SIZE = 100
FINAL_TOP_K = 10


def parse_arguments() -> argparse.Namespace:
    """Parse paths for the Phase 1 prediction and evaluation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/phase1/baseline_predictions.json"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/scifact"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/baseline_evaluation.json"),
    )
    return parser.parse_args()


def expected_budget(method_name: str, candidate_pool_size: int) -> int:
    """Return the Phase 1 reranking budget encoded by a method name."""
    if method_name == "full_rerank":
        return candidate_pool_size

    if method_name.startswith("fixed_prefix_") or method_name.startswith("random_"):
        try:
            return int(method_name.rsplit("_", maxsplit=1)[1])
        except ValueError as error:
            raise ValueError(
                f"Method {method_name!r} does not end in an integer budget"
            ) from error

    raise ValueError(f"Unknown Phase 1 baseline method: {method_name!r}")


def load_predictions(path: Path) -> dict[str, Any]:
    """Load and minimally validate a baseline prediction JSON artifact."""
    with path.open(encoding="utf-8") as handle:
        predictions = json.load(handle)

    if predictions.get("candidate_pool_size") != CANDIDATE_POOL_SIZE:
        raise ValueError(f"candidate_pool_size must be {CANDIDATE_POOL_SIZE}")

    if predictions.get("top_k") != FINAL_TOP_K:
        raise ValueError(f"top_k must be {FINAL_TOP_K} for Phase 1 evaluation")

    if not isinstance(predictions.get("methods"), Mapping):
        raise ValueError("predictions must contain a methods mapping")

    return predictions


def validate_method_predictions(
    method_name: str,
    predictions: Sequence[Mapping[str, Any]],
    top_k: int,
    candidate_pool_size: int,
) -> None:
    """Validate prediction shape, final rank depth, IDs, and reranking budgets."""
    budget = expected_budget(method_name, candidate_pool_size)

    if not 0 <= budget <= candidate_pool_size:
        raise ValueError(
            f"Method {method_name!r} has invalid reranking budget {budget}"
        )

    query_ids: set[str] = set()

    for prediction in predictions:
        if not {"query_id", "document_ids", "reranked_pairs"} <= prediction.keys():
            raise ValueError(
                f"Method {method_name!r} has an incomplete prediction"
            )

        query_id = str(prediction["query_id"])

        if query_id in query_ids:
            raise ValueError(
                f"Method {method_name!r} contains duplicate query ID {query_id!r}"
            )

        query_ids.add(query_id)

        document_ids = prediction["document_ids"]

        if not isinstance(document_ids, list) or len(document_ids) > top_k:
            raise ValueError(
                f"Method {method_name!r}, query {query_id!r} "
                f"exceeds top_k={top_k}"
            )

        if prediction["reranked_pairs"] != budget:
            raise ValueError(
                f"Method {method_name!r}, query {query_id!r} has "
                f"reranked_pairs={prediction['reranked_pairs']}; "
                f"expected {budget}"
            )


def evaluate_method(
    method_name: str,
    predictions: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, int]],
    top_k: int,
    candidate_pool_size: int,
) -> dict[str, Any]:
    """Evaluate one method, retaining missing-qrel IDs as validation metadata."""
    validate_method_predictions(
        method_name,
        predictions,
        top_k,
        candidate_pool_size,
    )

    matched_predictions = [
        prediction
        for prediction in predictions
        if str(prediction["query_id"]) in qrels
    ]

    missing_query_ids = sorted(
        {str(prediction["query_id"]) for prediction in predictions}
        - set(qrels)
    )

    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []

    for prediction in matched_predictions:
        query_id = str(prediction["query_id"])
        ranked_ids = [
            str(document_id)
            for document_id in prediction["document_ids"]
        ]
        relevance = qrels[query_id]

        ndcg_scores.append(
            ndcg_at_k(ranked_ids, relevance, k=top_k)
        )
        recall_scores.append(
            recall_at_k(ranked_ids, relevance, k=top_k)
        )
        mrr_scores.append(
            mrr_at_k(ranked_ids, relevance, k=top_k)
        )

    pair_counts = [
        reranked_pairs_per_query(
            int(prediction["reranked_pairs"])
        )
        for prediction in matched_predictions
    ]

    average_pairs = average_reranked_pairs(pair_counts)

    return {
        "quality_query_count": len(matched_predictions),
        "prediction_query_count": len(predictions),
        "missing_qrels_query_count": len(missing_query_ids),
        "missing_qrels_query_ids": missing_query_ids,
        "ndcg_at_10": (
            sum(ndcg_scores) / len(ndcg_scores)
            if ndcg_scores
            else 0.0
        ),
        "recall_at_10": (
            sum(recall_scores) / len(recall_scores)
            if recall_scores
            else 0.0
        ),
        "mrr_at_10": (
            sum(mrr_scores) / len(mrr_scores)
            if mrr_scores
            else 0.0
        ),
        "average_reranked_pairs_per_query": average_pairs,
        "total_reranking_pairs": total_reranking_pairs(pair_counts),
        "compression_ratio": (
            sum(
                compression_ratio(candidate_pool_size, count)
                for count in pair_counts
            )
            / len(pair_counts)
            if pair_counts
            else 0.0
        ),
    }


def print_table(results: Mapping[str, Mapping[str, Any]]) -> None:
    """Print a compact Phase 1 quality, cost, and latency comparison table."""
    print(
        "Method | nDCG@10 | Recall@10 | MRR@10 | "
        "Avg Pairs | Total Pairs | Compression | "
        "L_rerank Mean | L_rerank P50 | L_rerank P95 | "
        "L_CE Mean | L_CE P50 | L_CE P95 | "
        "LR Mean | LR P50 | LR P95"
    )
    print(
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:"
    )

    for method_name, result in results.items():
        latency = result["latency"]

        rerank = latency["rerank_latency_seconds"]
        ce = latency["ce_latency_seconds"]
        reduction = latency["latency_reduction"]

        if reduction is None:
            reduction_cells = ("-", "-", "-")
        else:
            reduction_cells = tuple(
                f"{reduction[key]:.6f}"
                for key in ("mean", "p50", "p95")
            )

        print(
            f"{method_name} | "
            f"{result['ndcg_at_10']:.6f} | "
            f"{result['recall_at_10']:.6f} | "
            f"{result['mrr_at_10']:.6f} | "
            f"{result['average_reranked_pairs_per_query']:.2f} | "
            f"{result['total_reranking_pairs']} | "
            f"{result['compression_ratio']:.2%} | "
            f"{rerank['mean']:.6f} | "
            f"{rerank['p50']:.6f} | "
            f"{rerank['p95']:.6f} | "
            f"{ce['mean']:.6f} | "
            f"{ce['p50']:.6f} | "
            f"{ce['p95']:.6f} | "
            + " | ".join(reduction_cells)
        )


def save_csv(
    results: Mapping[str, Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Save the Phase 1 comparison metrics as CSV."""
    csv_path = output_path.with_suffix(".csv")

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "method",
                "ndcg_at_10",
                "recall_at_10",
                "mrr_at_10",
                "average_reranked_pairs_per_query",
                "total_reranking_pairs",
                "compression_ratio",
                "rerank_latency_mean",
                "rerank_latency_p50",
                "rerank_latency_p95",
                "ce_latency_mean",
                "ce_latency_p50",
                "ce_latency_p95",
                "latency_reduction_mean",
                "latency_reduction_p50",
                "latency_reduction_p95",
            ]
        )

        for method_name, result in results.items():
            latency = result["latency"]

            rerank = latency["rerank_latency_seconds"]
            ce = latency["ce_latency_seconds"]
            reduction = latency["latency_reduction"]

            writer.writerow(
                [
                    method_name,
                    result["ndcg_at_10"],
                    result["recall_at_10"],
                    result["mrr_at_10"],
                    result["average_reranked_pairs_per_query"],
                    result["total_reranking_pairs"],
                    result["compression_ratio"],
                    rerank["mean"],
                    rerank["p50"],
                    rerank["p95"],
                    ce["mean"],
                    ce["p50"],
                    ce["p95"],
                    None if reduction is None else reduction["mean"],
                    None if reduction is None else reduction["p50"],
                    None if reduction is None else reduction["p95"],
                ]
            )


def main() -> None:
    """Run Phase 1 baseline evaluation and save machine-readable results."""
    args = parse_arguments()

    prediction_data = load_predictions(args.predictions)

    _, _, qrels = load_beir_dataset(
        args.dataset_dir,
        qrels_split="test",
    )

    results: dict[str, dict[str, Any]] = {}

    # Quality + cost evaluation.
    for method_name, method_data in tqdm(
        prediction_data["methods"].items(),
        total=len(prediction_data["methods"]),
        desc="Evaluating methods",
        unit="method",
    ):
        method_predictions = method_data.get("predictions")

        if not isinstance(method_predictions, list):
            raise ValueError(
                f"Method {method_name!r} must contain a predictions list"
            )

        results[method_name] = evaluate_method(
            method_name,
            method_predictions,
            qrels,
            prediction_data["top_k"],
            prediction_data["candidate_pool_size"],
        )

    # Latency evaluation.
    # The latency measurements were already collected by run_baselines.py,
    # so this step only aggregates the recorded values. It does not rerun
    # retrieval or Cross-Encoder inference.
    latency_report = build_latency_report(prediction_data)

    for method_name, latency_result in latency_report["methods"].items():
        if method_name not in results:
            raise ValueError(
                f"Latency report contains unknown method {method_name!r}"
            )

        results[method_name]["latency"] = latency_result

    output = {
        "dataset": prediction_data.get("dataset"),
        "qrels_split": "test",
        "candidate_pool_size": prediction_data["candidate_pool_size"],
        "top_k": prediction_data["top_k"],
        "qrels_query_count": len(qrels),
        "methods": results,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
        )
        handle.write("\n")

    save_csv(results, args.output)

    print_table(results)

    for method_name, result in results.items():
        if result["missing_qrels_query_count"]:
            print(
                f"WARNING: {method_name} has "
                f"{result['missing_qrels_query_count']} prediction query IDs "
                "without test qrels; quality and cost metrics use "
                "matched queries only."
            )


if __name__ == "__main__":
    main()
