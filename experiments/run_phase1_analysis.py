"""Generate reproducible Phase 1 analysis from the official SciFact test run.

This script consumes recorded predictions and evaluation results; it never runs
retrieval or reranking.  Its validation deliberately rejects any population
other than the 300-query SciFact test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_beir_dataset
from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k


TEST_QUERY_COUNT = 300
TOP_K = 10
FULL_METHOD = "full_rerank"
PREFIX_METHODS = [f"fixed_prefix_{budget}" for budget in (10, 20, 30, 50)]
RANDOM_METHODS = [f"random_{budget}" for budget in (10, 20, 30, 50)]
METHODS = [FULL_METHOD, *PREFIX_METHODS, *RANDOM_METHODS]
METRICS = ("ndcg_at_10", "recall_at_10", "mrr_at_10")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("results/phase1/baseline_predictions.json"))
    parser.add_argument("--evaluation", type=Path, default=Path("results/phase1/baseline_evaluation.json"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase1"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_population(predictions: dict[str, Any], qrels: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return predictions indexed by method/query after strict split checks."""
    if len(qrels) != TEST_QUERY_COUNT:
        raise ValueError(f"Expected {TEST_QUERY_COUNT} test qrels, found {len(qrels)}")
    if predictions.get("candidate_pool_size") != 100 or predictions.get("top_k") != TOP_K:
        raise ValueError("Phase 1 requires candidate_pool_size=100 and top_k=10")
    expected_ids = set(qrels)
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    methods = predictions.get("methods", {})
    if set(methods) != set(METHODS):
        raise ValueError(f"Expected exactly Phase 1 methods: {METHODS}")
    for method in METHODS:
        rows = methods[method].get("predictions", [])
        ids = [str(row["query_id"]) for row in rows]
        if len(rows) != TEST_QUERY_COUNT or len(ids) != len(set(ids) ) or set(ids) != expected_ids:
            raise ValueError(f"{method} does not contain the same 300 unique test query IDs")
        indexed[method] = {str(row["query_id"]): row for row in rows}
    return indexed


def query_metrics(prediction: dict[str, Any], relevance: dict[str, int]) -> dict[str, float]:
    ids = [str(document_id) for document_id in prediction["document_ids"]]
    return {
        "ndcg_at_10": ndcg_at_k(ids, relevance, k=TOP_K),
        "recall_at_10": recall_at_k(ids, relevance, k=TOP_K),
        "mrr_at_10": mrr_at_k(ids, relevance, k=TOP_K),
    }


def build_per_query(indexed: dict[str, dict[str, dict[str, Any]]], qrels: dict[str, dict[str, int]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    metric_index: dict[str, dict[str, dict[str, float]]] = {method: {} for method in METHODS}
    rows: list[dict[str, Any]] = []
    for query_id in sorted(qrels, key=lambda value: int(value) if value.isdigit() else value):
        row: dict[str, Any] = {"query_id": query_id}
        for method in METHODS:
            scores = query_metrics(indexed[method][query_id], qrels[query_id])
            metric_index[method][query_id] = scores
            for metric, value in scores.items():
                row[f"{method}_{metric}"] = value
        full = metric_index[FULL_METHOD][query_id]
        for method in PREFIX_METHODS + RANDOM_METHODS:
            for metric in METRICS:
                row[f"{method}_{metric}_delta_vs_full"] = metric_index[method][query_id][metric] - full[metric]
            row[f"{method}_preserves_full_metrics"] = all(metric_index[method][query_id][metric] == full[metric] for metric in METRICS)
        rows.append(row)
    return rows, metric_index


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm(p_values: list[float]) -> list[float]:
    adjusted = [0.0] * len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    running = 0.0
    total = len(p_values)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - rank)))
        adjusted[index] = running
    return adjusted


def statistics(metric_index: dict[str, dict[str, dict[str, float]]], query_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in PREFIX_METHODS:
        for metric in METRICS:
            full_values = [metric_index[FULL_METHOD][query_id][metric] for query_id in query_ids]
            method_values = [metric_index[method][query_id][metric] for query_id in query_ids]
            differences = [a - b for a, b in zip(full_values, method_values)]
            nonzero = sum(value != 0 for value in differences)
            if nonzero:
                test = wilcoxon(full_values, method_values, alternative="two-sided", zero_method="wilcox")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            else:
                statistic, p_value = 0.0, 1.0
            rows.append({"metric": metric, "comparison": f"{FULL_METHOD} vs {method}", "test": "Wilcoxon signed-rank (two-sided; zero differences excluded)", "statistic": statistic, "p_value": p_value, "nonzero_pair_count": nonzero, "query_count": len(query_ids)})
    for row, adjusted in zip(rows, holm([row["p_value"] for row in rows])):
        row["holm_adjusted_p_value"] = adjusted
    return rows


def failures(indexed: dict[str, dict[str, dict[str, Any]]], metric_index: dict[str, dict[str, dict[str, float]]], query_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in PREFIX_METHODS + RANDOM_METHODS:
        selection = "fixed_prefix" if method.startswith("fixed_prefix") else "random"
        for query_id in query_ids:
            full, observed = metric_index[FULL_METHOD][query_id], metric_index[method][query_id]
            delta = observed["ndcg_at_10"] - full["ndcg_at_10"]
            if delta < 0:
                rows.append({"method": method, "selection": selection, "query_id": query_id, "full_ndcg_at_10": full["ndcg_at_10"], "method_ndcg_at_10": observed["ndcg_at_10"], "ndcg_at_10_delta_vs_full": delta, "recall_at_10_delta_vs_full": observed["recall_at_10"] - full["recall_at_10"], "mrr_at_10_delta_vs_full": observed["mrr_at_10"] - full["mrr_at_10"], "full_document_ids": json.dumps(indexed[FULL_METHOD][query_id]["document_ids"]), "method_document_ids": json.dumps(indexed[method][query_id]["document_ids"])})
    return sorted(rows, key=lambda row: (row["method"], row["ndcg_at_10_delta_vs_full"], row["query_id"]))


def pareto(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in [FULL_METHOD, *PREFIX_METHODS]:
        result = evaluation["methods"][method]
        latency, reduction = result["latency"]["rerank_latency_seconds"], result["latency"]["latency_reduction"]
        rows.append({"method": method, "ndcg_at_10": result["ndcg_at_10"], "recall_at_10": result["recall_at_10"], "mrr_at_10": result["mrr_at_10"], "reranked_pairs_per_query": result["average_reranked_pairs_per_query"], "compression_ratio": result["compression_ratio"], "rerank_latency_mean_seconds": latency["mean"], "rerank_latency_p50_seconds": latency["p50"], "rerank_latency_p95_seconds": latency["p95"], "latency_reduction_mean": None if reduction is None else reduction["mean"], "latency_reduction_p50": None if reduction is None else reduction["p50"], "latency_reduction_p95": None if reduction is None else reduction["p95"]})
    return rows


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height, margin = 720, 420, 65
    max_latency = max(float(row["rerank_latency_mean_seconds"]) for row in rows)
    points = []
    for row in rows:
        x = margin + (float(row["rerank_latency_mean_seconds"]) / max_latency) * (width - 2 * margin)
        y = height - margin - (float(row["ndcg_at_10"]) / 0.7) * (height - 2 * margin)
        points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"/><text x="{x + 8:.1f}" y="{y - 8:.1f}" font-size="12">{row["method"]}</text>')
    path.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/><text x="{width/2}" y="{height-18}" text-anchor="middle">Mean reranking latency (seconds)</text><text x="20" y="{height/2}" text-anchor="middle" transform="rotate(-90 20 {height/2})">nDCG@10</text>{''.join(points)}</svg>''', encoding="utf-8")


def main() -> None:
    args = arguments()
    predictions, evaluation = load_json(args.predictions), load_json(args.evaluation)
    _, _, qrels = load_beir_dataset(args.dataset_dir, qrels_split="test")
    indexed = validate_population(predictions, qrels)
    per_query, metric_index = build_per_query(indexed, qrels)
    query_ids = [row["query_id"] for row in per_query]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_query_analysis.csv", per_query)
    failure_rows = failures(indexed, metric_index, query_ids)
    write_csv(args.output_dir / "failure_analysis.csv", failure_rows)
    statistical_rows = statistics(metric_index, query_ids)
    write_csv(args.output_dir / "statistical_analysis.csv", statistical_rows)
    pareto_rows = pareto(evaluation)
    write_csv(args.output_dir / "pareto_analysis.csv", pareto_rows)
    write_svg(args.output_dir / "pareto_quality_latency.svg", pareto_rows)
    summary = {"dataset": "scifact", "qrels_split": "test", "official_population_size": len(query_ids), "candidate_pool_size": 100, "top_k": TOP_K, "methods_evaluated": METHODS, "source_predictions": str(args.predictions), "source_evaluation": str(args.evaluation), "analysis_script": str(Path(__file__).resolve()), "per_query_analysis": str(args.output_dir / "per_query_analysis.csv"), "failure_analysis": str(args.output_dir / "failure_analysis.csv"), "statistical_analysis": str(args.output_dir / "statistical_analysis.csv"), "pareto_analysis": str(args.output_dir / "pareto_analysis.csv"), "pareto_visualization": str(args.output_dir / "pareto_quality_latency.svg"), "statistical_results": statistical_rows, "validation": {"status": "passed", "same_300_unique_test_query_ids_for_all_methods": True}, "limitations": ["Candidate-pool ranks are not recorded in the prediction artifact, so failure rows report output rankings and metric deltas but do not make candidate-rank claims."]}
    (args.output_dir / "phase1_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {len(query_ids)} SciFact test queries; wrote Phase 1 analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
