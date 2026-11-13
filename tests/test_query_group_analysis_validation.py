"""Focused integrity checks for supplementary QGA retrieval-quality inputs."""
from __future__ import annotations

import unittest
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from improvement.improvement_1.query_group_analysis.scripts.evaluate_retrieval_quality import (
    EXPECTED_GROUPS,
    EXPECTED_QUERY_COUNT,
    EXPECTED_GROUP_SIZE,
    GROUP_FILES,
    GROUP_COLUMNS,
    METRICS,
    OUTPUT_COLUMNS,
    load_rankings,
    validate_assignments,
    validate_locked_ids,
    validate_output_rows,
)


def make_assignments() -> list[dict[str, str]]:
    """Build a balanced synthetic assignment table with the production labels."""
    rows = []
    for index in range(EXPECTED_QUERY_COUNT):
        rows.append({
            "query_id": f"q{index}",
            "redundancy_group": "redundancy_low" if index < EXPECTED_GROUP_SIZE else "redundancy_high",
            "sparse_dense_agreement_group": "low_agreement" if index < EXPECTED_GROUP_SIZE else "high_agreement",
        })
    return rows


def make_output_rows() -> list[dict[str, object]]:
    """Build the complete expected output design with dummy valid metrics."""
    rows = []
    for column in GROUP_COLUMNS:
        for group_name in EXPECTED_GROUPS[column]:
            for method in GROUP_FILES:
                for metric in METRICS:
                    rows.append({
                        "grouping_strategy": column,
                        "group_name": group_name,
                        "query_count": EXPECTED_GROUP_SIZE,
                        "retrieval_method": method,
                        "metric": metric,
                        "mean": 0.5,
                        "median": 0.5,
                        "std_population": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "metric_cutoff": 10,
                        "qrels_source": "qrels.tsv",
                        "ranking_source": "ranking.csv",
                        "reconstruction_run_id": "run-id",
                        "rrf_alignment_gate": "GATE_A_EXACT_MATCH",
                        "aggregation": "query-level descriptive statistics; no hypothesis test",
                    })
    return [{column: row[column] for column in OUTPUT_COLUMNS} for row in rows]


class TestQueryGroupAnalysisValidation(unittest.TestCase):
    def test_locked_ids_must_be_exactly_300_unique_nonempty_ids(self) -> None:
        ids = [f"q{index}" for index in range(EXPECTED_QUERY_COUNT)]
        self.assertEqual(len(validate_locked_ids(ids)), EXPECTED_QUERY_COUNT)
        with self.assertRaises(ValueError):
            validate_locked_ids(ids[:-1])
        with self.assertRaises(ValueError):
            validate_locked_ids(ids[:-1] + [ids[0]])
        with self.assertRaises(ValueError):
            validate_locked_ids(ids[:-1] + [None])

    def test_rankings_require_exact_query_coverage_and_unique_contiguous_ranks(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ranking.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("query_id", "doc_id", "bm25_rank"))
                writer.writeheader()
                writer.writerows({"query_id": "q1", "doc_id": f"d{i}", "bm25_rank": i} for i in range(1, 101))
            self.assertEqual(len(load_rankings(path, "bm25_rank", {"q1"})["q1"]), 100)

            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("query_id", "doc_id", "bm25_rank"))
                writer.writeheader()
                writer.writerows({"query_id": "q1", "doc_id": f"d{i}", "bm25_rank": i} for i in range(1, 100))
            with self.assertRaises(ValueError):
                load_rankings(path, "bm25_rank", {"q1"})

            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("query_id", "doc_id", "bm25_rank"))
                writer.writeheader()
                writer.writerows(
                    {"query_id": "q1", "doc_id": f"d{i}", "bm25_rank": i if i < 100 else 99}
                    for i in range(1, 101)
                )
            with self.assertRaises(ValueError):
                load_rankings(path, "bm25_rank", {"q1"})

    def test_assignments_require_exact_coverage_and_balanced_expected_groups(self) -> None:
        rows = make_assignments()
        query_ids = {row["query_id"] for row in rows}
        self.assertEqual(len(validate_assignments(rows, query_ids)), EXPECTED_QUERY_COUNT)

        duplicated = rows[:-1] + [rows[0]]
        with self.assertRaises(ValueError):
            validate_assignments(duplicated, query_ids)

        wrong_label = [dict(row) for row in rows]
        wrong_label[0]["sparse_dense_agreement_group"] = "rank_agreement_high"
        with self.assertRaises(ValueError):
            validate_assignments(wrong_label, query_ids)

        missing_group = [dict(row) for row in rows]
        missing_group[0]["sparse_dense_agreement_group"] = ""
        with self.assertRaises(ValueError):
            validate_assignments(missing_group, query_ids)

        unbalanced = [dict(row) for row in rows]
        unbalanced[0]["redundancy_group"] = "redundancy_high"
        with self.assertRaises(ValueError):
            validate_assignments(unbalanced, query_ids)

    def test_output_requires_exact_schema_and_36_unique_design_rows(self) -> None:
        rows = make_output_rows()
        self.assertEqual(len(rows), 36)
        validate_output_rows(rows)
        with self.assertRaises(ValueError):
            validate_output_rows(rows[:-1])

        duplicate = [dict(row) for row in rows]
        duplicate[-1] = dict(duplicate[0])
        with self.assertRaises(ValueError):
            validate_output_rows(duplicate)

        wrong_schema = [dict(row) for row in rows]
        wrong_schema[0]["unexpected"] = "value"
        with self.assertRaises(ValueError):
            validate_output_rows(wrong_schema)


if __name__ == "__main__":
    unittest.main()
