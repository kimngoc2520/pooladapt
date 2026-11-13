"""Compare frozen PoolAdapt and Fixed Prefix outcomes in preassigned groups."""
from __future__ import annotations

import json
from typing import Any

try:
    from .common import LOCKED, OUT, RECON, ROOT, write_csv_new
except ImportError:
    from common import LOCKED, OUT, RECON, ROOT, write_csv_new
import numpy as np
import pandas as pd

GROUPS = OUT / "groups"
EVALUATION = OUT / "evaluation"
AUDIT = OUT / "audit"
QUALITY_METRICS = ["nDCG@10", "Recall@10", "MRR@10"]
COST_METRICS = ["reranked_pairs", "reranking_latency"]
ALL_METRICS = QUALITY_METRICS + COST_METRICS
TARGETS = [10, 20, 30, 50]
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42


def paired_bootstrap_ci(left: np.ndarray, right: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return paired mean difference and query-level percentile 95% CI."""
    if left.shape != right.shape or not len(left):
        raise ValueError("Paired bootstrap vectors must be nonempty and the same shape")
    delta = left - right
    indices = rng.integers(0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta)))
    means = delta[indices].mean(axis=1)
    return float(delta.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main() -> None:
    """Evaluate matched frozen records without reranking or metric regeneration."""
    metric_path = ROOT / "results" / "phase2" / "evaluation" / "final_gpu" / "test_query_metrics.csv"
    prediction_path = ROOT / "results" / "phase2" / "evaluation" / "final_gpu" / "test_predictions.jsonl"
    group_path = GROUPS / "query_group_assignments.csv"
    manifest_path = RECON / "reconstruction_manifest.json"
    if not all(path.is_file() for path in (metric_path, prediction_path, group_path, manifest_path)):
        raise FileNotFoundError("Frozen metrics, predictions, reconstruction manifest, and group assignments are required")
    if any(path.exists() for path in (EVALUATION / "subgroup_comparison.csv", EVALUATION / "subgroup_comparison_report.md")):
        raise FileExistsError("Refusing to overwrite existing subgroup evaluation output")
    locked = json.loads(LOCKED.read_text(encoding="utf-8"))
    ids = [str(qid) for qid in locked["test_ids"]]
    if len(ids) != 300 or len(set(ids)) != 300:
        raise ValueError("Locked TEST population is not 300 unique query IDs")
    assignments = pd.read_csv(group_path, dtype={"query_id": str})
    if len(assignments) != 300 or set(assignments["query_id"]) != set(ids):
        raise ValueError("Group assignment does not cover the locked TEST IDs exactly")
    forbidden_group_cols = [column for column in assignments.columns if any(token in column.lower() for token in
        ("ndcg", "recall", "mrr", "qrel", "relevance", "label", "pooladapt", "fixed_prefix", "latency", "cost", "rerank"))]
    if forbidden_group_cols:
        raise ValueError(f"Evaluation outcome fields leaked into group assignments: {forbidden_group_cols}")
    metrics = pd.read_csv(metric_path, dtype={"query_id": str})
    metrics["target_budget"] = pd.to_numeric(metrics["target_budget"], errors="raise").astype(int)
    selected = metrics[metrics["method"].isin(["PoolAdapt", "Fixed Prefix"])].copy()
    if selected["seed"].notna().any() and (selected["seed"].astype(str).str.strip() != "").any():
        raise ValueError("PoolAdapt/Fixed Prefix frozen records have unexpected non-empty seed dimension")
    if set(selected["target_budget"].unique()) != set(TARGETS):
        raise ValueError("PoolAdapt/Fixed Prefix do not have all four frozen target budgets")
    if set(selected["query_id"]) != set(ids):
        raise ValueError("PoolAdapt/Fixed Prefix outcome rows do not cover all 300 TEST IDs")
    duplicated = selected.duplicated(["method", "target_budget", "query_id"], keep=False)
    if duplicated.any():
        raise ValueError("Duplicate frozen method/budget/query records require protocol resolution")
    for method in ("PoolAdapt", "Fixed Prefix"):
        method_rows = selected[selected["method"] == method]
        for budget in TARGETS:
            rows = method_rows[method_rows["target_budget"] == budget]
            if len(rows) != 300 or set(rows["query_id"]) != set(ids):
                raise ValueError(f"{method}, target budget {budget}: expected exact coverage of 300 frozen queries")
    for column in ALL_METRICS:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
        if not np.isfinite(selected[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Frozen evaluation metric {column} contains non-finite values")

    joined = selected.merge(assignments, on="query_id", how="left", validate="many_to_one", indicator=True)
    if not (joined["_merge"] == "both").all() or len(joined) != len(selected):
        raise ValueError("Evaluation outcomes and independent group assignments failed exact ID alignment")
    group_columns = [column for column in ("redundancy_group", "sparse_dense_agreement_group") if column in assignments.columns and assignments[column].nunique() > 1 and not assignments[column].eq("not_assigned_validation_gate").all()]
    if not group_columns:
        raise ValueError("No valid query grouping strategy is available for subgroup evaluation")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output_rows: list[dict[str, Any]] = []
    for group_col in group_columns:
        for group_name, group_assign in assignments.groupby(group_col, sort=True, dropna=False):
            query_ids = sorted(group_assign["query_id"].astype(str))
            if len(query_ids) < 30:
                raise ValueError(f"Subgroup {group_col}={group_name} has only {len(query_ids)} queries")
            for budget in TARGETS:
                indexed: dict[str, dict[str, dict[str, float]]] = {}
                for method in ("PoolAdapt", "Fixed Prefix"):
                    sub = selected[(selected["method"] == method) & (selected["target_budget"] == budget)].set_index("query_id")
                    indexed[method] = {qid: {metric: float(sub.loc[qid, metric]) for metric in ALL_METRICS} for qid in query_ids}
                for metric in ALL_METRICS:
                    left = np.asarray([indexed["PoolAdapt"][qid][metric] for qid in query_ids], dtype=float)
                    right = np.asarray([indexed["Fixed Prefix"][qid][metric] for qid in query_ids], dtype=float)
                    difference, ci_low, ci_high = paired_bootstrap_ci(left, right, rng)
                    mean_pool, mean_fixed = float(left.mean()), float(right.mean())
                    output_rows.append({
                        "grouping_strategy": group_col, "group_name": group_name,
                        "query_count": len(query_ids), "test_population_percentage": len(query_ids) / 300 * 100,
                        "target_budget": budget, "metric": metric,
                        "PoolAdapt_mean": mean_pool, "PoolAdapt_median": float(np.median(left)), "PoolAdapt_std": float(np.std(left, ddof=0)),
                        "Fixed_Prefix_mean": mean_fixed, "Fixed_Prefix_median": float(np.median(right)), "Fixed_Prefix_std": float(np.std(right, ddof=0)),
                        "difference_PoolAdapt_minus_FixedPrefix": difference,
                        "relative_difference_percent": (difference / abs(mean_fixed) * 100) if mean_fixed != 0 else None,
                        "paired_query_bootstrap_95ci_lower": ci_low, "paired_query_bootstrap_95ci_upper": ci_high,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
                        "paired": True, "seed_dimension": "not applicable; frozen PoolAdapt and Fixed Prefix rows have blank seed",
                        "budget_interpretation": "matched target average budget; PoolAdapt realized pairs vary by query, Fixed Prefix uses prefix K",
                        "outcome_source": "frozen results/phase2/evaluation/final_gpu/test_query_metrics.csv",
                        "rrf_alignment_gate": json.loads(manifest_path.read_text(encoding="utf-8"))["integrity"]["pool_validation"]["classification"],
                    })
    write_csv_new(EVALUATION / "subgroup_comparison.csv", output_rows, list(output_rows[0]))

    report = [
        "# PoolAdapt vs Fixed Prefix: Query-Group Comparison", "",
        "## Protocol and alignment", "",
        f"- Frozen records are aligned by `query_id`; all comparisons use the same {len(ids)} frozen TEST IDs.",
        "- Method names are `PoolAdapt` and `Fixed Prefix`; target budgets 10/20/30/50 identify matched operating points.",
        "- `seed` is blank for both methods in the saved query-level protocol; there is one row per method, target budget, and query. No seed averaging is applied.",
        "- Outcomes come only from the existing frozen `test_query_metrics.csv`. No reranking or evaluation outcomes were regenerated.",
        "- Quality metrics are the protocol's nDCG@10, Recall@10, and MRR@10; Recall@100 is unavailable in these frozen per-query records. Efficiency metrics are reranked pairs and measured reranking latency.",
        "- Mean differences are paired by query. 95% percentile intervals use 2,000 query-level bootstrap replicates, preserving each query's PoolAdapt/Fixed Prefix pairing (seed 42).",
        "- Differences are descriptive. The analysis does not establish causality; subgroups were formed from pre-reranking features without outcomes.",
        "", "## Group results", "",
        "Detailed means, medians, standard deviations, paired differences, bootstrap intervals, and budgets are in `subgroup_comparison.csv`.",
        "", "## Interpretation limits", "",
        "Budget operating points are matched by the saved target_budget label; PoolAdapt's realized per-query reranked_pairs may vary, while Fixed Prefix uses the target prefix. Latency is the recorded reranking latency for each query and reflects the frozen execution environment. Small groups and descriptive bootstrap intervals should not be read as confirmatory causal evidence.",
    ]
    report_path = EVALUATION / "subgroup_comparison_report.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    rec_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    integrity = [
        "# Data Integrity Audit", "",
        "- Frozen TEST population is sourced from `locked_configurations.json`; 300 unique IDs were validated.",
        "- Corpus, queries, and test qrels SHA-256 values were checked against the Phase 2 execution manifest before retrieval.",
        "- Test qrels were read only for hash integrity in retrieval preflight; feature and grouping code did not read qrels or relevance labels.",
        f"- Frozen prediction SHA-256: `{rec_manifest['frozen_reference_hashes']['test_predictions.jsonl']}`.",
        f"- Frozen query-metric SHA-256: `{rec_manifest['frozen_reference_hashes']['test_query_metrics.csv']}`.",
        f"- BM25 integrity: {'passed' if rec_manifest['integrity']['bm25_valid'] else 'failed'}; rows={rec_manifest['integrity']['bm25_rows']}.",
        f"- Dense integrity: {'passed' if rec_manifest['integrity']['dense_valid'] else 'failed'}; rows={rec_manifest['integrity']['dense_rows']}.",
        f"- RRF pool alignment: `{rec_manifest['integrity']['pool_validation']['classification']}`; ordered exact {rec_manifest['integrity']['pool_validation']['exact_ordered_match_count']}/300, set exact {rec_manifest['integrity']['pool_validation']['exact_set_match_count']}/300.",
        "- No existing predictions, qrels, query metrics, execution manifests, or Improvement 1 artifacts were written by these scripts.",
    ]
    leakage = [
        "# Leakage Audit", "",
        "- Feature extraction reads frozen pool IDs, corpus document text, query text, reconstructed retrieval outputs, and BM25 IDF statistics.",
        "- Feature extraction does not open qrels, labels, candidate outcomes, reranking results, latency, or cost records.",
        "- Group thresholds use feature medians only; outcome columns are rejected from the group input schema.",
        "- Evaluation outcomes are joined only after group assignments are frozen, by exact query_id; evaluation columns are never used to define groups.",
        "- Component-based grouping is available only if reconstructed RRF achieves Gate A. Under Gate B/C, only direct frozen-pool features may define groups.",
        "- Independent training/development data are not used as TEST substitutes; no outcome-derived Easy/Hard labels are reused.",
    ]
    for filename, content in (("data_integrity_audit.md", integrity), ("leakage_audit.md", leakage)):
        path = AUDIT / filename
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(content) + "\n", encoding="utf-8")

    completion = [
        "# Query-Group Analysis Completion", "",
        f"- Reconstruction run: `{rec_manifest['retrieval_run_id']}`.",
        f"- Frozen RRF validation: `{rec_manifest['integrity']['pool_validation']['classification']}`.",
        f"- TEST groups evaluated: {', '.join(group_columns)}.",
        f"- Subgroup comparison rows: {len(output_rows)}; matched budgets: {TARGETS}; bootstraps: {BOOTSTRAP_REPLICATES} paired query resamples.",
        "- Evaluation is descriptive and observational; no causal claim is made.",
        "- Frozen prediction, qrels, evaluation metrics, manifests, and Improvement 1 outputs were preserved.",
        "- Relevant tests should be run separately after artifact generation; this workflow does not rerun the full experiment.",
    ]
    completion_path = AUDIT / "completion_report.md"
    if completion_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {completion_path}")
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_text("\n".join(completion) + "\n", encoding="utf-8")
    print(json.dumps({"groups_evaluated": group_columns, "comparison_rows": len(output_rows),
                      "queries": len(ids), "budgets": TARGETS, "bootstraps": BOOTSTRAP_REPLICATES}, indent=2))


if __name__ == "__main__":
    main()
