"""Assign interpretable, outcome-independent groups from pre-reranking features."""
from __future__ import annotations

import json
from typing import Any

try:
    from .common import OUT, write_csv_new
except ImportError:
    from common import OUT, write_csv_new
import numpy as np
import pandas as pd

FEATURES_DIR = OUT / "features"
GROUPS_DIR = OUT / "groups"
FORBIDDEN_TOKENS = ("ndcg", "recall", "mrr", "qrel", "relevance", "label", "pooladapt", "fixed_prefix", "latency", "cost", "rerank")
MIN_GROUP_N = 30


def validate_pre_reranking_columns(columns: list[str]) -> None:
    """Reject outcome-derived or treatment-result columns from group inputs."""
    forbidden = [column for column in columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
    if forbidden:
        raise ValueError(f"Outcome-derived or method outcome columns are forbidden in query-group inputs: {forbidden}")


def validate_assignment_coverage(query_ids: list[str], frozen_ids: list[str]) -> None:
    """Require each frozen TEST query exactly once in group assignment."""
    if len(query_ids) != len(set(query_ids)) or set(query_ids) != set(frozen_ids) or len(frozen_ids) != 300:
        raise ValueError("Group assignment does not cover the 300 frozen TEST IDs exactly once")


def median_groups(values: pd.Series, feature: str, prefix: str) -> tuple[pd.Series, float, dict[str, int]]:
    """Split at the empirical median and enforce useful minimum group size."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"Grouping feature {feature} contains missing or non-finite values")
    threshold = float(numeric.median())
    labels = pd.Series(np.where(numeric <= threshold, f"{prefix}_low", f"{prefix}_high"), index=values.index)
    counts = labels.value_counts().to_dict()
    if len(counts) != 2 or min(counts.values()) < MIN_GROUP_N:
        raise ValueError(f"Median split for {feature} yields undersized/empty groups: {counts}; refusing tiny subgroups")
    return labels, threshold, {str(k): int(v) for k, v in counts.items()}


def main() -> None:
    """Create one frozen TEST assignment row and subgroup distribution table."""
    feature_path = FEATURES_DIR / "test_query_features.csv"
    manifest_path = OUT / "reconstruction" / "reconstruction_manifest.json"
    if not feature_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Feature matrix and reconstruction manifest must exist before grouping")
    frame = pd.read_csv(feature_path, dtype={"query_id": str})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(frame) != 300 or frame["query_id"].nunique() != 300:
        raise ValueError("Group construction requires exact coverage of 300 unique frozen TEST queries")
    validate_pre_reranking_columns(list(frame.columns))
    gate = manifest["integrity"]["pool_validation"]["classification"]

    # Select the most interpretable top-20 redundancy feature with a valid split.
    redundancy_candidates = ["pool_top20_mean_pairwise_lexical_jaccard", "pool_top20_mean_max_lexical_jaccard",
                             "pool_top100_mean_pairwise_lexical_jaccard", "pool_top100_mean_max_lexical_jaccard"]
    selected_redundancy = None
    for candidate in redundancy_candidates:
        values = pd.to_numeric(frame[candidate], errors="coerce")
        if values.notna().all() and values.nunique() > 1:
            labels, threshold, counts = median_groups(values, candidate, "redundancy")
            selected_redundancy = (candidate, labels, threshold, counts)
            break
    if selected_redundancy is None:
        raise ValueError("No non-constant frozen-pool redundancy metric can support a valid median split")
    redundancy_feature, redundancy_labels, redundancy_threshold, redundancy_counts = selected_redundancy
    frame["redundancy_group"] = redundancy_labels
    frame["redundancy_group_feature"] = redundancy_feature
    frame["redundancy_group_threshold_median"] = redundancy_threshold
    frame["redundancy_group_rule"] = f"low <= median ({redundancy_threshold:.12g}); high > median"

    grouping_definitions: list[dict[str, Any]] = [{
        "grouping_strategy": "redundancy_group", "feature": redundancy_feature,
        "threshold": redundancy_threshold, "method": "empirical median split; tie values assigned low",
        "source": "frozen Phase 2 candidate pool; corpus-based lexical feature", "counts": redundancy_counts,
    }]
    frame["sparse_dense_agreement_group"] = "not_assigned_validation_gate"
    frame["sparse_dense_agreement_group_feature"] = "component_top20_jaccard"
    frame["sparse_dense_agreement_group_threshold_median"] = np.nan
    frame["sparse_dense_agreement_group_rule"] = "not assigned because reconstructed RRF did not reach Gate A"
    if gate == "GATE_A_EXACT_MATCH":
        agreement_labels, agreement_threshold, agreement_counts = median_groups(
            frame["component_top20_jaccard"], "component_top20_jaccard", "agreement")
        # A larger intersection/Jaccard means greater rank-list agreement.
        agreement_labels = agreement_labels.str.replace("agreement_low", "low_agreement", regex=False).str.replace("agreement_high", "high_agreement", regex=False)
        frame["sparse_dense_agreement_group"] = agreement_labels
        frame["sparse_dense_agreement_group_threshold_median"] = agreement_threshold
        frame["sparse_dense_agreement_group_rule"] = f"low agreement <= median ({agreement_threshold:.12g}); high agreement > median"
        grouping_definitions.append({
            "grouping_strategy": "sparse_dense_agreement_group", "feature": "component_top20_jaccard",
            "threshold": agreement_threshold, "method": "empirical median split; ties assigned low; higher Jaccard denotes higher agreement",
            "source": "reconstructed BM25/Dense Top-20; used only because frozen RRF ordering passed Gate A",
            "counts": agreement_counts,
        })

    assignments_path = GROUPS_DIR / "query_group_assignments.csv"
    if assignments_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {assignments_path}")
    write_csv_new(assignments_path, frame.to_dict(orient="records"), list(frame.columns))

    summary_rows: list[dict[str, Any]] = []
    groups_to_summarize = [("redundancy_group", redundancy_feature)]
    if gate == "GATE_A_EXACT_MATCH":
        groups_to_summarize.append(("sparse_dense_agreement_group", "component_top20_jaccard"))
    for group_col, feature in groups_to_summarize:
        for group_name, group_frame in frame.groupby(group_col, sort=True):
            values = pd.to_numeric(group_frame[feature], errors="coerce").dropna()
            summary_rows.append({
                "grouping_strategy": group_col, "group_name": group_name, "feature": feature,
                "sample_count": len(group_frame), "test_population_percentage": len(group_frame) / 300 * 100,
                "feature_mean": float(values.mean()), "feature_median": float(values.median()),
                "feature_std": float(values.std(ddof=0)), "feature_p10": float(values.quantile(.1)),
                "feature_p25": float(values.quantile(.25)), "feature_p75": float(values.quantile(.75)),
                "feature_p90": float(values.quantile(.9)), "threshold": grouping_definitions[0 if group_col == "redundancy_group" else 1]["threshold"],
                "grouping_methodology_version": "median-split-v1", "rrf_alignment_gate": gate,
                "feature_provenance": "frozen_pool_direct" if group_col == "redundancy_group" else "reconstructed_components_gate_a",
            })
    write_csv_new(GROUPS_DIR / "group_distribution_summary.csv", summary_rows, list(summary_rows[0]))
    methodology = {
        "version": "median-split-v1", "test_queries": 300, "outcome_leakage": "none; only pre-reranking features and locked IDs read",
        "rrf_alignment_gate": gate, "minimum_group_size": MIN_GROUP_N, "definitions": grouping_definitions,
        "rule_for_gate_b_or_c": "Sparse-Dense agreement groups are omitted because reconstructed component rankings are not validated as characteristics of frozen candidate pools.",
        "interpretation": "Median splits are descriptive query strata. They are not learned from reranking outcomes and do not establish causal effect modification.",
    }
    method_path = GROUPS_DIR / "grouping_methodology.md"
    if method_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {method_path}")
    method_path.write_text("# Query Grouping Methodology\n\n```json\n" + json.dumps(methodology, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"assignments": len(frame), "gate": gate, "groups": grouping_definitions}, indent=2))


if __name__ == "__main__":
    main()
