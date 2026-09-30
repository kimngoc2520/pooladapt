from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Input / output paths
# ---------------------------------------------------------------------------

POOL_FEATURES_PATH = (
    PROJECT_ROOT
    / "results"
    / "phase2"
    / "01_candidate_logging"
    / "pool_features.csv"
)

QUERY_FEATURES_PATH = (
    PROJECT_ROOT
    / "results"
    / "phase2"
    / "01_candidate_logging"
    / "query_features.csv"
)

MSTAR_PATH = (
    PROJECT_ROOT
    / "results"
    / "phase2"
    / "02_oracle"
    / "mstar_labels_tau099.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "phase2"
    / "05_easy_hard_analysis"
)

OUTPUT_PATH = OUTPUT_DIR / "feature_easy_hard_analysis.csv"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

ID_COLUMN = "query_id"
MSTAR_COLUMN = "M_star"


POOL_FEATURE_COLUMNS = [
    "top1_top2_rrf_margin",
    "top20_overlap",
    "rank_correlation_union",
    "score_entropy",
    "score_gini",
    "redundancy",
    "concentration",
    "sparse_dense_agreement",
    "sparse_dense_disagreement",
]

QUERY_FEATURE_COLUMNS = [
    "query_length",
    "num_numeric_tokens",
    "entity_count",
    "avg_idf",
    "max_idf",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: str | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(number):
        return None

    return number


def holm_adjust(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni adjustment.

    Returns adjusted p-values in the original feature order.
    """
    n = len(p_values)

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: item[1],
    )

    adjusted = [1.0] * n
    running_max = 0.0

    for rank, (original_index, p_value) in enumerate(indexed):
        adjusted_value = min(1.0, (n - rank) * p_value)
        running_max = max(running_max, adjusted_value)
        adjusted[original_index] = running_max

    return adjusted


def auc_with_orientation(
    easy_values: list[float],
    hard_values: list[float],
) -> tuple[float, float]:
    """
    Compute univariate AUC for predicting Hard=1.

    Returns:
        auc_raw:
            Standard ROC-AUC using the original feature direction.

        auc_directional:
            max(AUC, 1-AUC), so that stronger separation is always
            closer to 1 regardless of whether Hard has higher or lower
            feature values.
    """
    values = np.asarray(
        easy_values + hard_values,
        dtype=float,
    )

    # Constant feature: AUC is undefined / uninformative.
    if len(np.unique(values)) < 2:
        return float("nan"), float("nan")

    labels = np.asarray(
        [0] * len(easy_values) + [1] * len(hard_values),
        dtype=int,
    )

    # Need both classes.
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    auc_raw = roc_auc_score(labels, values)
    auc_directional = max(auc_raw, 1.0 - auc_raw)

    return float(auc_raw), float(auc_directional)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("EASY vs HARD FEATURE ANALYSIS")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Check input files
    # -----------------------------------------------------------------------

    input_paths = [
        POOL_FEATURES_PATH,
        QUERY_FEATURES_PATH,
        MSTAR_PATH,
    ]

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file: {path}"
            )

    # -----------------------------------------------------------------------
    # Load artifacts
    # -----------------------------------------------------------------------

    pool_rows = read_csv(POOL_FEATURES_PATH)
    query_rows = read_csv(QUERY_FEATURES_PATH)
    mstar_rows = read_csv(MSTAR_PATH)

    pool_by_id = {
        row[ID_COLUMN]: row
        for row in pool_rows
    }

    query_by_id = {
        row[ID_COLUMN]: row
        for row in query_rows
    }

    mstar_by_id = {
        row[ID_COLUMN]: row
        for row in mstar_rows
    }

    # Only analyze queries available in all required artifacts.
    common_ids = (
        set(pool_by_id)
        & set(query_by_id)
        & set(mstar_by_id)
    )

    if not common_ids:
        raise RuntimeError(
            "No common query IDs across input artifacts."
        )

    print(f"Pool feature queries:  {len(pool_by_id)}")
    print(f"Query feature queries: {len(query_by_id)}")
    print(f"M* queries:            {len(mstar_by_id)}")
    print(f"Common queries:         {len(common_ids)}")

    # -----------------------------------------------------------------------
    # Define features
    # -----------------------------------------------------------------------

    feature_columns = (
        POOL_FEATURE_COLUMNS
        + QUERY_FEATURE_COLUMNS
    )

    results = []

    # -----------------------------------------------------------------------
    # Analyze each feature
    # -----------------------------------------------------------------------

    for feature in feature_columns:
        easy_values: list[float] = []
        hard_values: list[float] = []

        missing_count = 0

        for query_id in sorted(common_ids):
            # ---------------------------------------------------------------
            # M* label
            # ---------------------------------------------------------------

            mstar_value = safe_float(
                mstar_by_id[query_id].get(MSTAR_COLUMN)
            )

            if mstar_value is None:
                missing_count += 1
                continue

            # ---------------------------------------------------------------
            # Feature value
            # ---------------------------------------------------------------

            if feature in pool_by_id[query_id]:
                raw_value = pool_by_id[query_id][feature]

            elif feature in query_by_id[query_id]:
                raw_value = query_by_id[query_id][feature]

            else:
                missing_count += 1
                continue

            value = safe_float(raw_value)

            if value is None:
                missing_count += 1
                continue

            # ---------------------------------------------------------------
            # Easy / Hard grouping
            # ---------------------------------------------------------------

            if mstar_value == 10:
                easy_values.append(value)

            elif mstar_value > 10:
                hard_values.append(value)

        # -------------------------------------------------------------------
        # Cannot compare if one group is empty
        # -------------------------------------------------------------------

        if not easy_values or not hard_values:
            results.append(
                {
                    "feature": feature,
                    "n_easy": len(easy_values),
                    "n_hard": len(hard_values),
                    "easy_mean": "",
                    "hard_mean": "",
                    "easy_median": "",
                    "hard_median": "",
                    "mean_difference_hard_minus_easy": "",
                    "median_difference_hard_minus_easy": "",
                    "mannwhitney_u": "",
                    "p_value": "",
                    "p_holm": "",
                    "auc_raw": "",
                    "auc_directional": "",
                    "direction": "",
                    "missing_count": missing_count,
                }
            )

            continue

        # -------------------------------------------------------------------
        # Mann-Whitney U test
        # -------------------------------------------------------------------

        statistic, p_value = mannwhitneyu(
            easy_values,
            hard_values,
            alternative="two-sided",
        )

        # -------------------------------------------------------------------
        # AUC
        # -------------------------------------------------------------------

        auc_raw, auc_directional = auc_with_orientation(
            easy_values,
            hard_values,
        )

        # -------------------------------------------------------------------
        # Direction
        # -------------------------------------------------------------------

        if np.isnan(auc_raw):
            direction = ""

        elif auc_raw >= 0.5:
            direction = "hard_higher"

        else:
            direction = "hard_lower"

        # -------------------------------------------------------------------
        # Store result
        # -------------------------------------------------------------------

        results.append(
            {
                "feature": feature,
                "n_easy": len(easy_values),
                "n_hard": len(hard_values),
                "easy_mean": float(
                    np.mean(easy_values)
                ),
                "hard_mean": float(
                    np.mean(hard_values)
                ),
                "easy_median": float(
                    np.median(easy_values)
                ),
                "hard_median": float(
                    np.median(hard_values)
                ),
                "mean_difference_hard_minus_easy": float(
                    np.mean(hard_values)
                    - np.mean(easy_values)
                ),
                "median_difference_hard_minus_easy": float(
                    np.median(hard_values)
                    - np.median(easy_values)
                ),
                "mannwhitney_u": float(statistic),
                "p_value": float(p_value),
                "p_holm": "",
                "auc_raw": auc_raw,
                "auc_directional": auc_directional,
                "direction": direction,
                "missing_count": missing_count,
            }
        )

    # -----------------------------------------------------------------------
    # Holm correction
    # -----------------------------------------------------------------------

    p_values = [
        row["p_value"]
        if row["p_value"] != ""
        else 1.0
        for row in results
    ]

    adjusted_p_values = holm_adjust(p_values)

    for row, adjusted in zip(
        results,
        adjusted_p_values,
    ):
        row["p_holm"] = float(adjusted)

    # -----------------------------------------------------------------------
    # Sort by directional AUC
    # -----------------------------------------------------------------------

    def auc_sort_key(row: dict) -> float:
        value = row["auc_directional"]

        if value == "":
            return -1.0

        if isinstance(value, float) and np.isnan(value):
            return -1.0

        return float(value)

    results.sort(
        key=auc_sort_key,
        reverse=True,
    )

    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "feature",
        "n_easy",
        "n_hard",
        "easy_mean",
        "hard_mean",
        "easy_median",
        "hard_median",
        "mean_difference_hard_minus_easy",
        "median_difference_hard_minus_easy",
        "mannwhitney_u",
        "p_value",
        "p_holm",
        "auc_raw",
        "auc_directional",
        "direction",
        "missing_count",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------------------------
    # Count Easy / Hard only within analysis population
    # -----------------------------------------------------------------------

    n_easy = 0
    n_hard = 0

    for query_id in common_ids:
        mstar_value = safe_float(
            mstar_by_id[query_id].get(MSTAR_COLUMN)
        )

        if mstar_value is None:
            continue

        if mstar_value == 10:
            n_easy += 1

        elif mstar_value > 10:
            n_hard += 1

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------

    print()
    print(f"Easy queries (in analysis): {n_easy}")
    print(f"Hard queries (in analysis): {n_hard}")

    print()
    print("Top features by directional AUC:")
    print("-" * 110)

    print(
        f"{'Feature':35s} "
        f"{'Easy med':>10s} "
        f"{'Hard med':>10s} "
        f"{'p_Holm':>12s} "
        f"{'AUC':>8s} "
        f"{'Direction':>14s}"
    )

    print("-" * 110)

    for row in results:
        # Robust against missing / undefined statistics.
        if (
            row["easy_median"] == ""
            or row["hard_median"] == ""
            or row["auc_directional"] == ""
        ):
            continue

        auc_value = float(row["auc_directional"])

        if np.isnan(auc_value):
            continue

        print(
            f"{row['feature']:35s} "
            f"{float(row['easy_median']):10.4f} "
            f"{float(row['hard_median']):10.4f} "
            f"{float(row['p_holm']):12.4g} "
            f"{auc_value:8.4f} "
            f"{row['direction']:>14s}"
        )

    print("-" * 110)
    print()
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()