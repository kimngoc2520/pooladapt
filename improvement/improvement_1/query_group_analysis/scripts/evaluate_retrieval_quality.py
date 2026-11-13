"""Summarize pre-reranking retrieval quality within frozen query groups.

This is a read-only evaluation of the saved BM25, Dense, and RRF rankings. It
does not rerun retrieval and does not use relevance labels to assign groups.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import LOCKED, OUT, RECON, ROOT, read_csv, write_csv_new
except ImportError:  # Direct script execution from this scripts directory.
    from common import LOCKED, OUT, RECON, ROOT, read_csv, write_csv_new

GROUP_COLUMNS = ("redundancy_group", "sparse_dense_agreement_group")
EXPECTED_GROUPS = {
    "redundancy_group": {"redundancy_low", "redundancy_high"},
    "sparse_dense_agreement_group": {"low_agreement", "high_agreement"},
}
GROUP_FILES = {
    "BM25": ("bm25_test_top100.csv", "bm25_rank"),
    "Dense": ("dense_test_top100.csv", "dense_rank"),
    "RRF": ("rrf_test_top100.csv", "rrf_rank"),
}
OUTPUT = OUT / "evaluation" / "pre_reranking_retrieval_quality_by_group.csv"
QRELS = ROOT / "data" / "scifact" / "qrels" / "test.tsv"
EXPECTED_QUERY_COUNT = 300
EXPECTED_GROUP_SIZE = 150
METRICS = ("nDCG@10", "Recall@10", "MRR@10")
OUTPUT_COLUMNS = (
    "grouping_strategy", "group_name", "query_count", "retrieval_method", "metric",
    "mean", "median", "std_population", "min", "max", "metric_cutoff", "qrels_source",
    "ranking_source", "reconstruction_run_id", "rrf_alignment_gate", "aggregation",
)


def validate_locked_ids(locked_ids: list[Any]) -> set[str]:
    """Require exactly 300 nonempty, unique locked TEST query IDs."""
    normalized = [str(query_id).strip() if query_id is not None else "" for query_id in locked_ids]
    query_ids = set(normalized)
    if (len(normalized) != EXPECTED_QUERY_COUNT or len(query_ids) != EXPECTED_QUERY_COUNT
            or any(not query_id for query_id in normalized)):
        raise ValueError("Expected exactly 300 nonempty, unique locked TEST query IDs")
    return query_ids


def validate_assignments(
    assignments: list[dict[str, str]], query_ids: set[str]
) -> dict[str, dict[str, str]]:
    """Validate exact assignment coverage and the two balanced groups per split."""
    if len(assignments) != EXPECTED_QUERY_COUNT:
        raise ValueError("Expected exactly one group-assignment row per locked TEST query")
    required = {"query_id", *GROUP_COLUMNS}
    assignment_by_query: dict[str, dict[str, str]] = {}
    for row in assignments:
        if not required.issubset(row):
            raise ValueError(f"Group assignments are missing required columns: {sorted(required - set(row))}")
        query_id = str(row["query_id"]).strip()
        if not query_id or query_id in assignment_by_query:
            raise ValueError("Group assignments contain a missing or duplicate query ID")
        assignment_by_query[query_id] = row
    if set(assignment_by_query) != query_ids:
        raise ValueError("Group assignments do not match the locked TEST IDs exactly")

    for column, expected_groups in EXPECTED_GROUPS.items():
        values = [str(row[column]).strip() for row in assignment_by_query.values()]
        if any(not value for value in values):
            raise ValueError(f"{column} contains a missing group assignment")
        counts = Counter(values)
        if set(counts) != expected_groups:
            raise ValueError(f"{column} must contain exactly groups {sorted(expected_groups)}")
        if any(counts[group] != EXPECTED_GROUP_SIZE for group in expected_groups):
            raise ValueError(f"{column} must contain exactly {EXPECTED_GROUP_SIZE} queries per group")
    return assignment_by_query


def validate_output_rows(output_rows: list[dict[str, Any]]) -> None:
    """Require the current 36-row method-by-group-by-metric output schema."""
    if len(output_rows) != 36:
        raise ValueError(f"Expected exactly 36 output rows; found {len(output_rows)}")
    if any(tuple(row) != OUTPUT_COLUMNS for row in output_rows):
        raise ValueError("Output rows do not match the required output schema and column order")
    expected_keys = {
        (group_column, group_name, method, metric)
        for group_column, group_names in EXPECTED_GROUPS.items()
        for group_name in group_names
        for method in GROUP_FILES
        for metric in METRICS
    }
    actual_keys = [
        (row["grouping_strategy"], row["group_name"], row["retrieval_method"], row["metric"])
        for row in output_rows
    ]
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != expected_keys:
        raise ValueError("Output rows do not contain each expected group/method/metric combination exactly once")
    for row in output_rows:
        if int(row["query_count"]) != EXPECTED_GROUP_SIZE or int(row["metric_cutoff"]) != 10:
            raise ValueError("Output rows have an unexpected group size or metric cutoff")
        for column in ("mean", "median", "std_population", "min", "max"):
            value = float(row[column])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Output metric {column} is non-finite or outside [0, 1]")


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    """Load positive SciFact relevance judgments for metric evaluation."""
    relevance: dict[str, dict[str, float]] = defaultdict(dict)
    seen_judgments: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"query-id", "corpus-id", "score"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected TEST qrels schema: {reader.fieldnames}")
        for row in reader:
            query_id = str(row["query-id"]).strip()
            corpus_id = str(row["corpus-id"]).strip()
            score = float(row["score"])
            if not query_id or not corpus_id or not math.isfinite(score) or score < 0:
                raise ValueError("TEST qrels contain a missing ID or invalid relevance score")
            key = (query_id, corpus_id)
            if key in seen_judgments:
                raise ValueError(f"Duplicate TEST qrels judgment for query/document {query_id}/{corpus_id}")
            seen_judgments.add(key)
            if score > 0:
                relevance[query_id][corpus_id] = score
    return dict(relevance)


def load_rankings(path: Path, rank_column: str, query_ids: set[str]) -> dict[str, list[str]]:
    """Load and validate one saved Top-100 retrieval ranking per TEST query."""
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in read_csv(path):
        query_id = str(row["query_id"]).strip()
        doc_id = str(row["doc_id"]).strip()
        if not query_id or not doc_id:
            raise ValueError(f"{path.name} contains a missing query or document ID")
        if query_id not in query_ids:
            raise ValueError(f"Unexpected query ID {query_id} in {path.name}")
        try:
            rank = int(row[rank_column])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path.name} contains an invalid {rank_column}") from error
        grouped[query_id].append((rank, doc_id))
    if set(grouped) != query_ids:
        raise ValueError(f"{path.name} does not cover all locked TEST IDs")

    rankings: dict[str, list[str]] = {}
    for query_id, ranked_docs in grouped.items():
        ranked_docs.sort(key=lambda item: item[0])
        ranks = [rank for rank, _ in ranked_docs]
        doc_ids = [doc_id for _, doc_id in ranked_docs]
        if ranks != list(range(1, 101)) or len(set(doc_ids)) != 100:
            raise ValueError(f"{path.name} has invalid Top-100 rows for query {query_id}")
        rankings[query_id] = doc_ids
    return rankings


def main() -> None:
    """Write descriptive pre-reranking metrics by preassigned group."""
    import numpy as np

    from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k

    lock = json.loads(LOCKED.read_text(encoding="utf-8"))
    query_ids = validate_locked_ids(lock["test_ids"])

    assignments = read_csv(OUT / "groups" / "query_group_assignments.csv")
    assignment_by_query = validate_assignments(assignments, query_ids)
    manifest = json.loads((RECON / "reconstruction_manifest.json").read_text(encoding="utf-8"))
    if manifest["integrity"]["pool_validation"]["classification"] != "GATE_A_EXACT_MATCH":
        raise ValueError("Frozen-pool RRF metrics require the exact-order alignment gate")

    qrels = load_qrels(QRELS)
    if set(qrels) != query_ids:
        raise ValueError("TEST qrels do not cover the exact locked TEST query IDs")

    output_rows: list[dict[str, Any]] = []
    for method, (filename, rank_column) in GROUP_FILES.items():
        rankings = load_rankings(RECON / filename, rank_column, query_ids)
        per_query: dict[str, dict[str, float]] = {}
        for query_id in sorted(query_ids):
            docs = rankings[query_id]
            rel = qrels[query_id]
            per_query[query_id] = {
                "nDCG@10": ndcg_at_k(docs, rel, k=10),
                "Recall@10": recall_at_k(docs, rel, k=10),
                "MRR@10": mrr_at_k(docs, rel, k=10),
            }
        for group_column in GROUP_COLUMNS:
            for group_name in sorted(EXPECTED_GROUPS[group_column]):
                members = sorted(qid for qid in query_ids if assignment_by_query[qid][group_column] == group_name)
                for metric in METRICS:
                    values = np.asarray([per_query[qid][metric] for qid in members], dtype=float)
                    output_rows.append({
                        "grouping_strategy": group_column,
                        "group_name": group_name,
                        "query_count": len(members),
                        "retrieval_method": method,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "std_population": float(values.std(ddof=0)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "metric_cutoff": 10,
                        "qrels_source": "data/scifact/qrels/test.tsv",
                        "ranking_source": f"improvement/improvement_1/query_group_analysis/reconstruction/{filename}",
                        "reconstruction_run_id": manifest["retrieval_run_id"],
                        "rrf_alignment_gate": manifest["integrity"]["pool_validation"]["classification"],
                        "aggregation": "query-level descriptive statistics; no hypothesis test",
                    })

    validate_output_rows(output_rows)
    write_csv_new(OUTPUT, output_rows, list(OUTPUT_COLUMNS))
    print(json.dumps({
        "output": str(OUTPUT),
        "query_count": len(query_ids),
        "rows": len(output_rows),
        "retrieval_methods": list(GROUP_FILES),
        "grouping_strategies": list(GROUP_COLUMNS),
        "reretrieval_or_reranking_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
