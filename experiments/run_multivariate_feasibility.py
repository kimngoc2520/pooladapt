"""Evaluate development-only multivariate Easy-vs-Hard feasibility.

This is an additional feasibility experiment only; it is not a G2 substitute.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = PROJECT_ROOT / "results/phase2/01_candidate_logging/pool_features.csv"
QUERY_PATH = PROJECT_ROOT / "results/phase2/01_candidate_logging/query_features.csv"
MSTAR_PATH = PROJECT_ROOT / "results/phase2/02_oracle/mstar_labels_tau099.csv"
DEVELOPMENT_IDS_PATH = PROJECT_ROOT / "results/phase2/01_candidate_logging/train_validation_query_ids.csv"
OUTPUT_DIR = PROJECT_ROOT / "results/phase2/06_multivariate_selector"

ID_COLUMN = "query_id"
MSTAR_COLUMN = "M_star"
PREDEFINED_TOP1 = ["concentration"]
PREDEFINED_TOP3 = ["concentration", "avg_idf", "top1_top2_rrf_margin"]
EXPECTED_N, EXPECTED_EASY, EXPECTED_HARD = 809, 757, 52
RANDOM_STATE, N_PERMUTATIONS = 42, 100


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def index_unique(rows: list[dict[str, str]], source: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        query_id = row.get(ID_COLUMN)
        if not query_id:
            raise ValueError(f"{source} has a missing query_id")
        if query_id in indexed:
            raise ValueError(f"{source} has duplicate query_id: {query_id!r}")
        indexed[query_id] = row
    return indexed


def ensure_safe_feature_columns(columns: list[str]) -> None:
    forbidden = [
        column for column in columns
        if column.lower() in {"query_id", "doc_id"}
        or "qrel" in column.lower()
        or "relevance" in column.lower()
    ]
    if forbidden:
        raise ValueError(
            "STOP: identifier or qrel/relevance columns would enter X: "
            + ", ".join(forbidden)
        )


def numeric_matrix(
    pool: dict[str, dict[str, str]], query: dict[str, dict[str, str]], ids: list[str]
) -> tuple[list[str], np.ndarray, dict[str, int], list[str], list[str]]:
    pool_columns = [name for name in next(iter(pool.values())) if name != ID_COLUMN]
    query_columns = [name for name in next(iter(query.values())) if name != ID_COLUMN]
    duplicate_features = sorted(set(pool_columns) & set(query_columns))
    if duplicate_features:
        raise ValueError("Duplicate pool/query feature names: " + ", ".join(duplicate_features))
    columns = pool_columns + query_columns
    ensure_safe_feature_columns(columns)

    values = np.full((len(ids), len(columns)), np.nan, dtype=float)
    for row_index, query_id in enumerate(ids):
        merged = {**pool[query_id], **query[query_id]}
        for column_index, column in enumerate(columns):
            raw = merged.get(column, "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values[row_index, column_index] = value

    missing = {column: int(np.isnan(values[:, i]).sum()) for i, column in enumerate(columns)}
    entirely_missing = [column for i, column in enumerate(columns) if np.isnan(values[:, i]).all()]
    available = [i for i, column in enumerate(columns) if column not in entirely_missing]
    constants = [
        columns[i] for i in available
        if np.nanmax(values[:, i]) == np.nanmin(values[:, i])
    ]
    selected = [i for i in available if columns[i] not in constants]
    return ([columns[i] for i in selected], values[:, selected], missing, entirely_missing, constants)


def pipeline(balanced: bool) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced" if balanced else None,
            random_state=RANDOM_STATE,
            max_iter=2000,
        )),
    ])


def splits_for(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(y)), y))


def metrics_for(X: np.ndarray, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]], balanced: bool) -> dict[str, list[float]]:
    roc_scores, pr_scores = [], []
    for train, test in splits:
        model = pipeline(balanced)
        model.fit(X[train], y[train])
        probabilities = model.predict_proba(X[test])[:, 1]
        roc_scores.append(float(roc_auc_score(y[test], probabilities)))
        pr_scores.append(float(average_precision_score(y[test], probabilities)))
    return {"roc_auc": roc_scores, "pr_auc": pr_scores}


def summary(scores: list[float]) -> dict[str, float]:
    data = np.asarray(scores, dtype=float)
    return {
        "mean": float(data.mean()), "std": float(data.std(ddof=0)),
        "median": float(np.median(data)), "min": float(data.min()), "max": float(data.max()),
    }


def evaluate(name: str, feature_names: list[str], all_names: list[str], X: np.ndarray, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]], balanced: bool) -> dict[str, object]:
    missing = [feature for feature in feature_names if feature not in all_names]
    if missing:
        raise ValueError(f"{name} requires unavailable feature(s): {', '.join(missing)}")
    indices = [all_names.index(feature) for feature in feature_names]
    scores = metrics_for(X[:, indices], y, splits, balanced)
    return {"features": feature_names, "roc_auc": summary(scores["roc_auc"]), "pr_auc": summary(scores["pr_auc"])}


def write_correlation(path: Path, names: list[str], X: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", *names])
        for i, name in enumerate(names):
            row = [name]
            for j in range(len(names)):
                usable = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
                correlation = np.nan if usable.sum() < 2 else np.corrcoef(X[usable, i], X[usable, j])[0, 1]
                row.append("" if not np.isfinite(correlation) else float(correlation))
            writer.writerow(row)


def permutation_test(X: np.ndarray, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]], observed: dict[str, object]) -> dict[str, object]:
    rng = np.random.RandomState(RANDOM_STATE)
    null_roc, null_pr = [], []
    for permutation in range(1, N_PERMUTATIONS + 1):
        shuffled_y = rng.permutation(y)
        scores = metrics_for(X, shuffled_y, splits_for(shuffled_y), balanced=True)
        null_roc.append(float(np.mean(scores["roc_auc"])))
        null_pr.append(float(np.mean(scores["pr_auc"])))
        if permutation % 10 == 0:
            print(f"Permutation {permutation}/{N_PERMUTATIONS}")

    def result(null: list[float], observed_score: float) -> dict[str, float | int]:
        values = np.asarray(null)
        return {
            "observed_score": observed_score,
            "null_mean": float(values.mean()), "null_std": float(values.std(ddof=0)),
            "empirical_p_value": float((1 + np.sum(values >= observed_score)) / (N_PERMUTATIONS + 1)),
            "n_permutations": N_PERMUTATIONS,
        }
    return {
        "model": "balanced logistic regression",
        "n_permutations": N_PERMUTATIONS, "random_state": RANDOM_STATE,
        "roc_auc": result(null_roc, float(observed["roc_auc"]["mean"])),
        "pr_auc": result(null_pr, float(observed["pr_auc"]["mean"])),
    }


def main() -> None:
    for path in (POOL_PATH, QUERY_PATH, MSTAR_PATH, DEVELOPMENT_IDS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
    pool = index_unique(read_rows(POOL_PATH), "pool_features.csv")
    query = index_unique(read_rows(QUERY_PATH), "query_features.csv")
    labels = index_unique(read_rows(MSTAR_PATH), "mstar_labels_tau099.csv")
    development_ids = {row[ID_COLUMN] for row in read_rows(DEVELOPMENT_IDS_PATH)}
    common = set(pool) & set(query) & set(labels)
    if len(common) != EXPECTED_N:
        missing = sorted(development_ids - common)
        extra = sorted(common - development_ids)
        raise ValueError(f"STOP: merged queries={len(common)}, expected={EXPECTED_N}; missing={missing}; extra={extra}")
    if common - development_ids:
        raise ValueError("STOP: test/non-development query IDs appear: " + ", ".join(sorted(common - development_ids)))
    if common != development_ids:
        raise ValueError("STOP: development query IDs do not exactly match merged IDs; missing=" + ", ".join(sorted(development_ids - common)))

    ids = sorted(common)
    raw_mstar = []
    for query_id in ids:
        try:
            value = float(labels[query_id][MSTAR_COLUMN])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Target label incomplete for query_id={query_id!r}") from error
        if not np.isfinite(value):
            raise ValueError(f"Target label incomplete for query_id={query_id!r}")
        raw_mstar.append(value)
    y = np.asarray([int(value > 10) for value in raw_mstar], dtype=int)
    n_hard, n_easy = int(y.sum()), int((y == 0).sum())
    if (n_easy, n_hard) != (EXPECTED_EASY, EXPECTED_HARD):
        raise ValueError(f"STOP: class counts are easy={n_easy}, hard={n_hard}; expected easy={EXPECTED_EASY}, hard={EXPECTED_HARD}")

    names, X, missing, entirely_missing, constants = numeric_matrix(pool, query, ids)
    if not names:
        raise ValueError("No eligible non-constant features remain")
    print(f"Excluded entirely missing features: {entirely_missing}")
    print(f"Excluded constant features: {constants}")
    print(f"Features used ({len(names)}): {names}")
    print(f"Missing-value counts: {missing}")

    splits = splits_for(y)
    top1 = evaluate("PREDEFINED_TOP1", PREDEFINED_TOP1, names, X, y, splits, balanced=True)
    top3 = evaluate("PREDEFINED_TOP3", PREDEFINED_TOP3, names, X, y, splits, balanced=True)
    balanced = evaluate("balanced multivariate", names, names, X, y, splits, balanced=True)
    unbalanced = evaluate("unbalanced multivariate", names, names, X, y, splits, balanced=False)
    permutation = permutation_test(X, y, splits, balanced)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_rows = []
    for label, result in (("predefined_top1", top1), ("predefined_top3", top3)):
        for metric in ("roc_auc", "pr_auc"):
            baseline_rows.append({"experiment": label, "features": ";".join(result["features"]), "metric": metric, **result[metric]})
    with (OUTPUT_DIR / "single_feature_baselines.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "features", "metric", "mean", "std", "median", "min", "max"])
        writer.writeheader(); writer.writerows(baseline_rows)
    write_correlation(OUTPUT_DIR / "feature_correlation.csv", names, X)
    fitted = pipeline(True).fit(X, y)
    coefficients = fitted.named_steps["model"].coef_[0]
    with (OUTPUT_DIR / "feature_coefficients.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "coefficient", "abs_coefficient"])
        writer.writeheader()
        writer.writerows({"feature": name, "coefficient": float(coefficient), "abs_coefficient": float(abs(coefficient))} for name, coefficient in zip(names, coefficients))

    deltas = {
        "vs_top1": {metric: float(balanced[metric]["mean"] - top1[metric]["mean"]) for metric in ("roc_auc", "pr_auc")},
        "vs_top3": {metric: float(balanced[metric]["mean"] - top3[metric]["mean"]) for metric in ("roc_auc", "pr_auc")},
    }
    results = {
        "scope": "feasibility experiment only, not a G2 substitute",
        "n_queries": len(ids), "n_easy": n_easy, "n_hard": n_hard, "positive_rate": float(y.mean()),
        "n_features": len(names), "feature_names": names, "missing_value_counts": missing,
        "entirely_missing_features_excluded": entirely_missing, "constant_features_excluded": constants,
        "sanity_checks": {"no_test_ids": True, "no_qrel_or_relevance_columns": True, "no_identifier_columns": True, "one_row_per_query": True},
        "cv": {"type": "RepeatedStratifiedKFold", "n_splits": 5, "n_repeats": 10, "n_folds_total": len(splits), "random_state": RANDOM_STATE},
        "predefined_top1": PREDEFINED_TOP1, "predefined_top3": PREDEFINED_TOP3,
        "models": {"balanced": balanced, "unbalanced": unbalanced},
        "single_feature_baselines": {"predefined_top1": top1, "predefined_top3": top3},
        "multivariate_deltas": deltas, "permutation_test": permutation,
        "coefficient_caveat": "Coefficients are model parameters, not causal effects. Correlated predictors can make them unstable, and abs(coefficient) is not a formal feature-importance ranking.",
        "correlation_caveat": "Pearson correlations are descriptive only; no features were removed because of correlation.",
    }
    with (OUTPUT_DIR / "multivariate_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2); handle.write("\n")

    print(f"\nQueries: {len(ids)}\nEasy: {n_easy}\nHard: {n_hard}\nHard prevalence: {y.mean():.4%}\n")
    print(f"Features used: {len(names)}")
    for label, result in (("Top-1: concentration", top1), ("Top-3", top3), ("Multivariate balanced", balanced), ("Multivariate unbalanced", unbalanced)):
        print(f"{label}\n  ROC-AUC: {result['roc_auc']['mean']:.4f} ± {result['roc_auc']['std']:.4f}\n  PR-AUC: {result['pr_auc']['mean']:.4f} ± {result['pr_auc']['std']:.4f}")
    print(f"\nMultivariate vs top-1: Δ ROC-AUC {deltas['vs_top1']['roc_auc']:.4f}; Δ PR-AUC {deltas['vs_top1']['pr_auc']:.4f}")
    print(f"Multivariate vs top-3: Δ ROC-AUC {deltas['vs_top3']['roc_auc']:.4f}; Δ PR-AUC {deltas['vs_top3']['pr_auc']:.4f}")
    print(f"Permutation test: ROC-AUC p-value {permutation['roc_auc']['empirical_p_value']:.4f}; PR-AUC p-value {permutation['pr_auc']['empirical_p_value']:.4f}")


if __name__ == "__main__":
    main()
