"""Verify original 13 feature formulas and compare held-out/development schemas."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .common import LOCKED, OUT, RECON, load_frozen_inputs, read_csv, write_csv_new
except ImportError:
    from common import LOCKED, OUT, RECON, load_frozen_inputs, read_csv, write_csv_new

ROOT = Path(__file__).resolve().parents[3]
FEATURES = OUT / "features"


def main() -> None:
    """Cross-check every TEST value against the canonical frozen feature function."""
    manifest = json.loads((RECON / "reconstruction_manifest.json").read_text(encoding="utf-8"))
    gate = manifest["integrity"]["pool_validation"]["classification"]
    if gate != "GATE_A_EXACT_MATCH":
        raise ValueError("Original 13 values can be verified only after exact RRF pool alignment")
    corpus, queries, query_ids, frozen_pools = load_frozen_inputs()
    original_path = FEATURES / "original_13_feature_matrix_reconstructed.csv"
    test_features = pd.read_csv(original_path, dtype={"query_id": str}).set_index("query_id")
    bm_rows = read_csv(RECON / "bm25_test_top100.csv")
    dense_rows = read_csv(RECON / "dense_test_top100.csv")
    rrf_rows = read_csv(RECON / "rrf_test_top100.csv")
    bm_by_query: dict[str, list[dict[str, Any]]] = {}
    dense_by_query: dict[str, list[dict[str, Any]]] = {}
    rrf_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in bm_rows:
        bm_by_query.setdefault(row["query_id"], []).append({"id": row["doc_id"], "rank": int(row["bm25_rank"]), "score": float(row["bm25_score"])})
    for row in dense_rows:
        dense_by_query.setdefault(row["query_id"], []).append({"id": row["doc_id"], "rank": int(row["dense_rank"]), "score": float(row["dense_score"])})
    rrf_score: dict[str, dict[str, float]] = {}
    for row in rrf_rows:
        rrf_score.setdefault(row["query_id"], {})[row["doc_id"]] = float(row["rrf_score"])

    from src.data.loader import load_beir_dataset
    from src.evaluation.phase2_inputs import frozen_features
    from src.pooladapt.selector import FROZEN_FEATURES
    from src.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever(corpus)
    canonical: list[dict[str, Any]] = []
    max_abs_diff = {name: 0.0 for name in FROZEN_FEATURES}
    failures: list[dict[str, Any]] = []
    for qid in query_ids:
        pool = [{"id": doc_id, "rrf_score": rrf_score[qid][doc_id]} for doc_id in frozen_pools[qid]]
        expected = frozen_features(queries[qid], pool, bm_by_query[qid], dense_by_query[qid], corpus, bm25)
        record: dict[str, Any] = {"query_id": qid}
        for name in FROZEN_FEATURES:
            expected_value = float(expected[name])
            actual_value = float(test_features.loc[qid, name]) if pd.notna(test_features.loc[qid, name]) else float("nan")
            difference = abs(actual_value - expected_value)
            max_abs_diff[name] = max(max_abs_diff[name], difference)
            record[name] = difference
            if not np.isclose(actual_value, expected_value, rtol=1e-10, atol=1e-12, equal_nan=True):
                failures.append({"query_id": qid, "feature": name, "canonical_value": expected_value, "saved_value": actual_value, "absolute_difference": difference})
        canonical.append(record)
    if failures:
        raise ValueError(f"Original feature formula mismatch for {len(failures)} query-feature pairs; first={failures[0]}")

    # These are development-schema references only; they are never used as TEST rows.
    dev_pool = pd.read_csv(ROOT / "results" / "phase2" / "01_candidate_logging" / "pool_features.csv", dtype={"query_id": str})
    dev_query = pd.read_csv(ROOT / "results" / "phase2" / "01_candidate_logging" / "query_features.csv", dtype={"query_id": str})
    dev_ids = set(dev_pool["query_id"]) & set(dev_query["query_id"])
    lock = json.loads(LOCKED.read_text(encoding="utf-8"))
    if dev_ids & set(str(qid) for qid in lock["test_ids"]):
        raise ValueError("Development feature tables unexpectedly overlap frozen TEST IDs")
    dev = dev_pool.merge(dev_query, on="query_id", how="inner", validate="one_to_one")
    if len(dev) != 809 or len(dev_ids) != 809:
        raise ValueError(f"Development feature schema comparison expected 809 matched IDs, found {len(dev)}")
    feature_rows = []
    for name in FROZEN_FEATURES:
        if name not in dev.columns or name not in test_features.columns:
            raise ValueError(f"Original frozen feature missing from TEST or development schema: {name}")
        dev_values = pd.to_numeric(dev[name], errors="coerce").dropna()
        test_values = pd.to_numeric(test_features[name], errors="coerce").dropna()
        feature_rows.append({
            "feature": name, "formula_exact_match_all_test_rows": True,
            "canonical_max_absolute_difference": max_abs_diff[name],
            "development_schema_present": True, "development_n_nonmissing": len(dev_values),
            "development_mean": float(dev_values.mean()) if len(dev_values) else None,
            "development_std": float(dev_values.std(ddof=0)) if len(dev_values) else None,
            "development_median": float(dev_values.median()) if len(dev_values) else None,
            "test_n_nonmissing": len(test_values), "test_mean_reconstructed": float(test_values.mean()) if len(test_values) else None,
            "test_std_reconstructed": float(test_values.std(ddof=0)) if len(test_values) else None,
            "test_median_reconstructed": float(test_values.median()) if len(test_values) else None,
            "interpretation": "same frozen formula/schema; different query population and reconstructed TEST retrieval provenance",
        })
    output_csv = FEATURES / "development_feature_consistency.csv"
    write_csv_new(output_csv, feature_rows, list(feature_rows[0]))
    report_path = FEATURES / "development_feature_consistency.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {report_path}")
    lines = [
        "# Original 13-Feature Consistency Audit", "",
        f"- Canonical implementation: `src/evaluation/phase2_inputs.py::frozen_features`; exact TEST query-feature pairs checked: {300 * len(FROZEN_FEATURES)}.",
        "- Every TEST feature matches the canonical project function at rtol=1e-10, atol=1e-12 (including nulls).",
        f"- Development comparison used {len(dev)} matched TRAIN/VALIDATION feature rows from existing `pool_features.csv` + `query_features.csv`; TEST IDs were checked disjoint.",
        "- All original feature names and formulas are present in both schemas. Development and TEST distribution summaries are in `development_feature_consistency.csv`.",
        "- Development values are schema references only. TEST values remain reconstructed; RRF ordering matches the frozen pools, but Dense revision and raw component scores are not proven historical.",
        "- No development query values were substituted for TEST values. No qrels or evaluation outcomes were read.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"test_queries": len(query_ids), "feature_count": len(FROZEN_FEATURES),
                      "checked_pairs": len(query_ids) * len(FROZEN_FEATURES), "mismatches": len(failures),
                      "development_rows": len(dev), "output": str(output_csv)}, indent=2))


if __name__ == "__main__":
    main()
