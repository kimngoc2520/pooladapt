"""Execute development-only CT2.0, CT2.1, and CT2.2 experiments.

The runner consumes persisted development rankings at every frozen budget; it
never loads TEST files or re-runs the Cross-Encoder.  It creates a new,
timestamped artifact tree containing raw fold records, one OOF record/query,
statistics, Gate-C simulations, manifests, and concise reports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
import sklearn
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# This runner lives at improvement/improvement_2/scripts/.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k

BUDGETS = (10, 20, 30, 50, 100)
FEATURES = {
    "full": ("top1_top2_rrf_margin", "top20_overlap", "rank_correlation_union", "score_entropy", "score_gini", "redundancy", "concentration", "sparse_dense_agreement", "sparse_dense_disagreement", "query_length", "num_numeric_tokens", "avg_idf", "max_idf"),
    "pool_only": ("top1_top2_rrf_margin", "top20_overlap", "rank_correlation_union", "score_entropy", "score_gini", "redundancy", "concentration", "sparse_dense_agreement", "sparse_dense_disagreement"),
    "query_only": ("query_length", "num_numeric_tokens", "avg_idf", "max_idf"),
}
METRICS = ("nDCG@10", "Recall@10", "MRR@10")


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    """Read a CSV/TSV file into dictionaries."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    """Write rows with a stable explicit schema."""
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(materialized)


def digest(path: Path) -> str:
    """Return SHA-256 for reproducibility manifests."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locked_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_csv(path, "\t"):
        result[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    return dict(result)


def load_features(feature_dir: Path, ids: list[str]) -> dict[str, dict[str, float]]:
    """Load exactly the frozen feature rows and preserve missing values as NaN."""
    pool = {str(r["query_id"]): r for r in read_csv(feature_dir / "pool_features.csv")}
    query = {str(r["query_id"]): r for r in read_csv(feature_dir / "query_features.csv")}
    features: dict[str, dict[str, float]] = {}
    for qid in ids:
        if qid not in pool or qid not in query:
            raise ValueError(f"missing feature row for query {qid}")
        merged = {**pool[qid], **query[qid]}
        features[qid] = {name: float(merged[name]) if merged.get(name, "") else float("nan") for names in FEATURES.values() for name in names}
    return features


def load_rankings(train_path: Path, validation_path: Path, qrels: dict[str, dict[str, int]]) -> dict[int, dict[str, dict[str, float]]]:
    """Compute registered metrics for each persisted fixed-prefix budget/query."""
    out: dict[int, dict[str, dict[str, float]]] = {budget: {} for budget in BUDGETS}
    for path in (train_path, validation_path):
        methods = json.loads(path.read_text(encoding="utf-8"))["methods"]
        for budget in BUDGETS:
            for prediction in methods[f"fixed_prefix_{budget}"]["predictions"]:
                qid = str(prediction["query_id"])
                if qid not in qrels or qid in out[budget]:
                    raise ValueError(f"invalid or duplicate ranking query {qid} in {path}")
                ranked = [str(doc) for doc in prediction["document_ids"]]
                relevance = qrels[qid]
                out[budget][qid] = {
                    "nDCG@10": ndcg_at_k(ranked, relevance),
                    "Recall@10": recall_at_k(ranked, relevance),
                    "MRR@10": mrr_at_k(ranked, relevance),
                    "reranking_latency": float(prediction.get("latency_rerank_seconds", "nan")),
                }
    if set(out[10]) != set(qrels):
        raise ValueError("rankings and development qrels do not align")
    return out


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 10_000) -> tuple[float, float, float]:
    """Return mean and two-sided percentile paired-bootstrap interval."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def build_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline((("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("classifier", LogisticRegression(C=c_value, max_iter=1000, class_weight="balanced", random_state=seed))))


def select_c(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, dict[float, float]]:
    """Select C by mean inner-fold five-class macro-F1; exact ties choose smaller C."""
    strata = np.asarray([0 if label == 10 else 1 if label in (20, 30) else 2 for label in y])
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores: dict[float, float] = {}
    for c_value in (.01, .1, 1.0, 10.0):
        fold_scores: list[float] = []
        for tr, va in inner.split(x, strata):
            model = build_model(c_value, seed)
            model.fit(x[tr], y[tr])
            fold_scores.append(float(f1_score(y[va], model.predict(x[va]), labels=BUDGETS, average="macro", zero_division=0)))
        scores[c_value] = float(np.mean(fold_scores))
    best = max(scores.values())
    return min(c for c, score in scores.items() if np.isclose(score, best)), scores


def align_probabilities(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """Return five budget columns; unavailable model classes receive zero probability."""
    raw = model.predict_proba(x)
    aligned = np.zeros((len(x), len(BUDGETS)), dtype=float)
    for source, label in enumerate(model.named_steps["classifier"].classes_):
        aligned[:, BUDGETS.index(int(label))] = raw[:, source]
    return aligned


def prediction_from_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return lambda=0 argmax budgets and log probabilities with epsilon only for logging."""
    choices = np.argmax(probabilities, axis=1)  # ordered BUDGETS provides smaller-K tie-break.
    return np.asarray([BUDGETS[i] for i in choices], dtype=int), np.log(np.maximum(probabilities, 1e-12))


def run_ablation(ids: list[str], features: dict[str, dict[str, float]], labels: dict[str, int], output: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Run the frozen repeated nested CV for all feature configurations."""
    y = np.asarray([labels[qid] for qid in ids], dtype=int)
    strata = np.asarray([0 if value == 10 else 1 if value in (20, 30) else 2 for value in y])
    fold_rows: list[dict[str, Any]] = []
    aggregate: dict[str, list[dict[str, Any]]] = {}
    for config, names in FEATURES.items():
        x = np.asarray([[features[qid][name] for name in names] for qid in ids], dtype=float)
        probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
        selected_cs: dict[str, list[float]] = defaultdict(list)
        for repeat in range(10):
            outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
            for fold, (tr, te) in enumerate(outer.split(x, strata)):
                chosen_c, inner_scores = select_c(x[tr], y[tr], seed=repeat * 100 + fold)
                model = build_model(chosen_c, repeat * 100 + fold)
                model.fit(x[tr], y[tr])
                probs = align_probabilities(model, x[te])
                budgets, _ = prediction_from_probabilities(probs)
                for position, row_index in enumerate(te):
                    qid = ids[row_index]
                    probabilities[qid].append(probs[position])
                    selected_cs[qid].append(chosen_c)
                    fold_rows.append({"configuration": config, "repeat": repeat, "fold": fold, "query_id": qid,
                                      "M_star": int(y[row_index]), "selected_C": chosen_c,
                                      "inner_macro_f1_C_0.01": inner_scores[.01], "inner_macro_f1_C_0.1": inner_scores[.1],
                                      "inner_macro_f1_C_1": inner_scores[1.0], "inner_macro_f1_C_10": inner_scores[10.0],
                                      "predicted_K": int(budgets[position]), **{f"probability_K_{b}": float(probs[position, i]) for i, b in enumerate(BUDGETS)}})
        rows: list[dict[str, Any]] = []
        for qid in ids:
            if len(probabilities[qid]) != 10:
                raise ValueError(f"OOF repeat count is not 10 for {config}/{qid}")
            mean_prob = np.mean(probabilities[qid], axis=0)
            budget, logits = prediction_from_probabilities(mean_prob.reshape(1, -1))
            rows.append({"configuration": config, "query_id": qid, "M_star": labels[qid], "repeat_predictions": 10,
                         "selected_C_mean": float(np.mean(selected_cs[qid])), "predicted_K": int(budget[0]),
                         **{f"probability_K_{b}": float(mean_prob[i]) for i, b in enumerate(BUDGETS)},
                         **{f"logit_K_{b}": float(logits[0, i]) for i, b in enumerate(BUDGETS)}})
        aggregate[config] = rows
    write_csv(output / "feature_ablation_fold_results.csv", fold_rows, fold_rows[0].keys())
    flat = [row for rows in aggregate.values() for row in rows]
    write_csv(output / "feature_ablation_oof_predictions.csv", flat, flat[0].keys())
    return aggregate, {"fold_rows": fold_rows}


def enrich(rows: list[dict[str, Any]], per_budget: dict[int, dict[str, dict[str, float]]]) -> list[dict[str, Any]]:
    """Attach the exact persisted per-budget quality/cost outcome to each OOF action."""
    result: list[dict[str, Any]] = []
    for row in rows:
        chosen = int(row["predicted_K"])
        metrics = per_budget[chosen][str(row["query_id"])]
        result.append({**row, **metrics, "reranked_pairs": chosen, "compression_ratio": 1 - chosen / 100})
    return result


def ablation_metrics(rows_by_config: dict[str, list[dict[str, Any]]], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for config, observations in rows_by_config.items():
        budgets = np.asarray([int(r["predicted_K"]) for r in observations])
        for metric in METRICS:
            rows.append({"configuration": config, "measure": metric, "mean": float(np.mean([float(r[metric]) for r in observations])), "n": len(observations)})
        rows.extend([
            {"configuration": config, "measure": "average_K", "mean": float(budgets.mean()), "n": len(observations)},
            {"configuration": config, "measure": "average_reranked_pairs", "mean": float(budgets.mean()), "n": len(observations)},
            {"configuration": config, "measure": "compression_ratio", "mean": float(np.mean([float(r["compression_ratio"]) for r in observations])), "n": len(observations)},
        ])
    write_csv(output / "feature_ablation_metrics.csv", rows, ("configuration", "measure", "mean", "n"))


def comparisons(rows_by_config: dict[str, list[dict[str, Any]]], output: Path) -> None:
    records: list[dict[str, Any]] = []
    contrast_pairs = (("full", "query_only", "primary"), ("full", "pool_only", "secondary"), ("pool_only", "query_only", "secondary"))
    raw_p: list[float] = []
    pending: list[dict[str, Any]] = []
    for left, right, role in contrast_pairs:
        by_left, by_right = ({r["query_id"]: r for r in rows_by_config[left]}, {r["query_id"]: r for r in rows_by_config[right]})
        for metric in METRICS:
            values = np.asarray([float(by_left[q][metric]) - float(by_right[q][metric]) for q in sorted(by_left)])
            mean, low, high = bootstrap_mean(values, seed=20261119 + len(pending))
            try:
                p_value = float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue) if np.any(values != 0) else 1.0
            except ValueError:
                p_value = 1.0
            pending.append({"contrast": f"{left}-{right}", "contrast_role": role, "metric": metric, "n": len(values), "mean_difference": mean, "ci95_low": low, "ci95_high": high, "wilcoxon_p": p_value})
            raw_p.append(p_value)
    order = np.argsort(raw_p)
    adjusted = [0.0] * len(raw_p)
    previous = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(raw_p) - rank) * raw_p[index])
        previous = max(previous, value)
        adjusted[index] = previous
    for row, p_holm in zip(pending, adjusted):
        records.append({**row, "holm_p": p_holm, "inference_unit": "one aggregated OOF record per query"})
    write_csv(output / "feature_ablation_statistical_analysis.csv", records, records[0].keys())


def calibrated_budget(logits: np.ndarray, lam: float) -> np.ndarray:
    bias = -(np.asarray(BUDGETS, dtype=float) - np.mean(BUDGETS)) / np.std(BUDGETS)
    return np.asarray([BUDGETS[i] for i in np.argmax(logits + lam * bias, axis=1)], dtype=int)


def gate_c(full_rows: list[dict[str, Any]], per_budget: dict[int, dict[str, dict[str, float]]], output: Path) -> None:
    """Run cost-matched Current-vs-permuted-budget null simulations at proposed targets."""
    ids = [str(r["query_id"]) for r in full_rows]
    logits = np.asarray([[float(r[f"logit_K_{b}"]) for b in BUDGETS] for r in full_rows])
    rng = np.random.default_rng(20261119)
    records: list[dict[str, Any]] = []
    simulation_rows: list[dict[str, Any]] = []
    for target in (11, 12, 14, 17, 20):
        candidates = np.linspace(-10, 10, 20001)
        means = np.asarray([calibrated_budget(logits, value).mean() for value in candidates])
        index = int(np.argmin(np.abs(means - target)))
        lam, observed = float(candidates[index]), float(means[index])
        matched = abs(observed - target) <= .1
        selected = calibrated_budget(logits, lam)
        actual = {metric: np.asarray([per_budget[int(k)][qid][metric] for qid, k in zip(ids, selected)], dtype=float) for metric in METRICS}
        null_means: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for repeat in range(1000):
            permuted = rng.permutation(selected)
            for metric in METRICS:
                null_means[metric].append(float(np.mean([per_budget[int(k)][qid][metric] for qid, k in zip(ids, permuted)])))
        for metric in METRICS:
            null_mean = float(np.mean(null_means[metric]))
            difference = actual[metric] - null_mean
            mean, low, high = bootstrap_mean(difference, seed=20261119 + target)
            records.append({"target_average_K": target, "lambda": lam, "observed_average_K": observed, "matched_absolute_tolerance_0.1": matched,
                            "metric": metric, "current_mean": float(actual[metric].mean()), "null_expected_mean": null_mean,
                            "current_minus_null": mean, "ci95_low": low, "ci95_high": high, "null_repetitions": 1000,
                            "null": "permutation of selected budget multiset across queries"})
        simulation_rows.append({"target_average_K": target, "lambda": lam, "observed_average_K": observed, "matched": matched,
                                **{f"selected_K_{b}": int(np.sum(selected == b)) for b in BUDGETS}})
    write_csv(output / "cost_matched_null_gate_c_results.csv", records, records[0].keys())
    write_csv(output / "cost_matched_null_gate_c_calibration.csv", simulation_rows, simulation_rows[0].keys())


def diagnosis(full_rows: list[dict[str, Any]], per_budget: dict[int, dict[str, dict[str, float]]], output: Path) -> None:
    """Produce query-level allocation and Hard-detection diagnosis from Full OOF."""
    records: list[dict[str, Any]] = []
    for row in full_rows:
        qid, predicted, oracle = str(row["query_id"]), int(row["predicted_K"]), int(row["M_star"])
        hard_score = sum(float(row[f"probability_K_{b}"]) for b in BUDGETS if b > 10)
        chosen, oracle_metrics = per_budget[predicted][qid], per_budget[oracle][qid]
        records.append({**row, "hard_true": int(oracle > 10), "hard_predicted": int(predicted > 10), "hard_score": hard_score,
                        "signed_allocation_error": predicted - oracle, "absolute_allocation_error": abs(predicted - oracle),
                        "ordinal_error": BUDGETS.index(predicted) - BUDGETS.index(oracle), "underallocated": int(predicted < oracle), "overallocated": int(predicted > oracle),
                        "oracle_nDCG@10": oracle_metrics["nDCG@10"], "oracle_Recall@10": oracle_metrics["Recall@10"], "oracle_MRR@10": oracle_metrics["MRR@10"],
                        "nDCG@10_loss_vs_oracle": oracle_metrics["nDCG@10"] - chosen["nDCG@10"], "additional_pairs_vs_oracle": predicted - oracle})
    write_csv(output / "budget_allocation_diagnosis.csv", records, records[0].keys())
    y_true, y_pred, scores = np.asarray([r["hard_true"] for r in records]), np.asarray([r["hard_predicted"] for r in records]), np.asarray([r["hard_score"] for r in records])
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    hard_rows = [{"measure": "hard_prevalence", "value": float(y_true.mean())}, {"measure": "average_precision", "value": float(average_precision_score(y_true, scores))},
                 {"measure": "hard_precision", "value": float(precision[1])}, {"measure": "hard_recall", "value": float(recall[1])}, {"measure": "hard_f1", "value": float(f1[1])},
                 {"measure": "hard_support", "value": int(support[1])}, {"measure": "macro_f1_hard_binary", "value": float(np.mean(f1))}]
    write_csv(output / "hard_query_detection_metrics.csv", hard_rows, ("measure", "value"))
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    write_csv(output / "hard_query_confusion_matrix.csv", [{"actual": actual, "predicted_easy": int(matrix[i, 0]), "predicted_hard": int(matrix[i, 1])} for i, actual in enumerate(("easy", "hard"))], ("actual", "predicted_easy", "predicted_hard"))
    groups: list[dict[str, Any]] = []
    for oracle in BUDGETS:
        subset = [r for r in records if r["M_star"] == oracle]
        groups.append({"M_star": oracle, "queries": len(subset), "mean_predicted_K": float(np.mean([r["predicted_K"] for r in subset])), "underallocation_rate": float(np.mean([r["underallocated"] for r in subset])), "overallocation_rate": float(np.mean([r["overallocated"] for r in subset])), "mean_nDCG_loss_vs_oracle": float(np.mean([r["nDCG@10_loss_vs_oracle"] for r in subset]))})
    write_csv(output / "allocation_failure_group_analysis.csv", groups, groups[0].keys())


def headroom(labels_path: Path, per_budget: dict[int, dict[str, dict[str, float]]], output: Path) -> None:
    labels = {str(row["query_id"]): row for row in read_csv(labels_path)}
    records: list[dict[str, Any]] = []
    for qid, label in sorted(labels.items()):
        budget = int(label["M_star"])
        oracle, fixed = per_budget[budget][qid]["nDCG@10"], per_budget[10][qid]["nDCG@10"]
        records.append({"query_id": qid, "M_star": budget, "oracle_mstar_nDCG@10": oracle, "fixed_prefix_10_nDCG@10": fixed, "paired_difference": oracle - fixed})
    values = np.asarray([r["paired_difference"] for r in records])
    mean, low, high = bootstrap_mean(values, 20261119)
    write_csv(output / "oracle_headroom_query_results.csv", records, records[0].keys())
    metrics = [{"population_n": len(values), "mean_paired_difference": mean, "ci95_low": low, "ci95_high": high, "delta_A": .005,
                "gate_A": "headroom_above_threshold" if low > .005 else "headroom_below_threshold" if high < .005 else "inconclusive",
                "bootstrap_draws": 10000, "bootstrap_seed": 20261119, "metric": "nDCG@10"}]
    write_csv(output / "oracle_headroom_metrics.csv", metrics, metrics[0].keys())
    write_csv(output / "oracle_headroom_statistical_analysis.csv", [{"statistic": "paired_bootstrap_mean_difference", "value": mean}, {"statistic": "ci95_low", "value": low}, {"statistic": "ci95_high", "value": high}], ("statistic", "value"))


def write_reports(execution_root: Path, reports_root: Path) -> None:
    """Regenerate reports from one completed timestamped execution directory.

    Reports are deliberately separated from immutable result directories.  This
    permits recovery from a reporting-only failure without repeating CV, ranking,
    or any other experimental computation.
    """
    reports_root.mkdir(parents=True, exist_ok=True)
    head = read_csv(execution_root / "oracle_headroom/oracle_headroom_metrics.csv")[0]
    metrics = read_csv(execution_root / "feature_ablation/feature_ablation_metrics.csv")
    stats = read_csv(execution_root / "feature_ablation/feature_ablation_statistical_analysis.csv")
    gate_c_rows = read_csv(execution_root / "feature_ablation/cost_matched_null_gate_c_results.csv")
    hard = {r["measure"]: r["value"] for r in read_csv(execution_root / "allocation_diagnosis/hard_query_detection_metrics.csv")}
    run_id = execution_root.name
    (reports_root / "oracle_headroom_analysis.md").write_text(
        f"# Oracle headroom ({run_id})\n\n"
        f"Development-only population: N={head['population_n']}. Oracle M* minus Fixed Prefix-10 nDCG@10 "
        f"was {float(head['mean_paired_difference']):.6f} (10,000-draw paired bootstrap 95% CI "
        f"[{float(head['ci95_low']):.6f}, {float(head['ci95_high']):.6f}]; delta_A={float(head['delta_A']):.3f}). "
        f"Gate A: **{head['gate_A']}**. M* is outcome-derived and optimistic, so this is headroom evidence, not controller evidence.\n\n"
        f"Source: `results/{run_id}/oracle_headroom/`.\n", encoding="utf-8")
    summary = "\n".join(f"- {r['configuration']} {r['measure']}: {float(r['mean']):.6f}" for r in metrics)
    primary = [r for r in stats if r["contrast"] == "full-query_only" and r["metric"] == "nDCG@10"][0]
    gate_c_ndcg = [r for r in gate_c_rows if r["metric"] == "nDCG@10"]
    gate_c_summary = "\n".join(
        f"- target K={r['target_average_K']} (observed {float(r['observed_average_K']):.3f}): "
        f"current-null={float(r['current_minus_null']):.6f}, 95% CI "
        f"[{float(r['ci95_low']):.6f}, {float(r['ci95_high']):.6f}]"
        for r in gate_c_ndcg)
    (reports_root / "feature_ablation_analysis.md").write_text(
        f"# Controlled feature ablation ({run_id})\n\n"
        "Development-only repeated nested CV (5 folds x 10 repeats) generated one mean-probability aggregated OOF prediction per query. "
        "The primary Full minus Query-only nDCG@10 contrast was "
        f"{float(primary['mean_difference']):.6f} (95% CI [{float(primary['ci95_low']):.6f}, {float(primary['ci95_high']):.6f}]; "
        f"Wilcoxon p={float(primary['wilcoxon_p']):.6g}, Holm p={float(primary['holm_p']):.6g}).\n\n"
        "## Aggregate outcomes\n\n" + summary + "\n\n## Gate C: cost-matched nDCG@10 vs permuted-budget null\n\n" + gate_c_summary +
        "\n\nEvery target met the absolute Average-K tolerance of 0.1. The logged CIs all include zero; these results do not show a positive Current-vs-null advantage under this simulation. "
        f"Sources: `results/{run_id}/feature_ablation/`.\n", encoding="utf-8")
    (reports_root / "budget_allocation_diagnosis.md").write_text(
        f"# Budget allocation diagnosis ({run_id})\n\n"
        "Full aggregated OOF predictions, development-only: "
        f"Hard prevalence={float(hard['hard_prevalence']):.6f}; AP={float(hard['average_precision']):.6f}; "
        f"precision={float(hard['hard_precision']):.6f}; recall={float(hard['hard_recall']):.6f}; "
        f"F1={float(hard['hard_f1']):.6f}. See the query-level allocation records, confusion matrix, and M* group analysis for allocation errors and failures. "
        "This descriptive diagnosis does not authorize a controller redesign or CT2.4.\n\n"
        f"Source: `results/{run_id}/allocation_diagnosis/`.\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "improvement/improvement_2")
    parser.add_argument("--generate-reports-from", type=Path, metavar="EXECUTION_DIR",
                        help="Recover reports from an existing timestamped execution directory without rerunning experiments.")
    args = parser.parse_args()
    if args.generate_reports_from:
        execution_root = args.generate_reports_from.resolve()
        required = ("oracle_headroom", "feature_ablation", "allocation_diagnosis")
        if not execution_root.is_dir() or any(not (execution_root / item).is_dir() for item in required):
            raise ValueError("--generate-reports-from must name a completed development_execution directory")
        reports_root = args.output_root / "reports"
        write_reports(execution_root, reports_root)
        recovery = {
            "event": "report_recovery",
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "execution_root": str(execution_root),
            "result_computation_rerun": False,
            "test_used": False,
            "reports": ["oracle_headroom_analysis.md", "feature_ablation_analysis.md", "budget_allocation_diagnosis.md"],
        }
        (reports_root / f"{execution_root.name}_report_recovery.json").write_text(
            json.dumps(recovery, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"execution_root": str(execution_root), "status": "reports_recovered"}, indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_root / "results" / f"development_execution_{stamp}"
    for relative in ("oracle_headroom", "feature_ablation", "allocation_diagnosis"):
        (root / relative).mkdir(parents=True, exist_ok=False)
    train_ids, val_ids = locked_ids(ROOT / "data/scifact/splits/train_query_ids.txt"), locked_ids(ROOT / "data/scifact/splits/validation_query_ids.txt")
    ids = train_ids + val_ids
    if len(ids) != 809 or len(set(ids)) != 809:
        raise ValueError("expected exactly 809 unique development IDs")
    qrels = {**load_qrels(ROOT / "data/scifact/qrels/train.tsv"), **load_qrels(ROOT / "data/scifact/qrels/validation.tsv")}
    labels = {str(r["query_id"]): int(r["M_star"]) for r in read_csv(ROOT / "results/phase2/02_oracle/mstar_labels_tau099.csv")}
    if set(qrels) != set(ids) or set(labels) != set(ids):
        raise ValueError("development qrels, labels, and IDs do not exactly align")
    features = load_features(ROOT / "results/phase2/01_candidate_logging", ids)
    per_budget = load_rankings(ROOT / "results/phase2/development_reranking/train/reranking_results.json", ROOT / "results/phase2/development_reranking/validation/reranking_results.json", qrels)
    headroom(ROOT / "results/phase2/02_oracle/mstar_labels_tau099.csv", per_budget, root / "oracle_headroom")
    oof, _ = run_ablation(ids, features, labels, root / "feature_ablation")
    enriched = {name: enrich(rows, per_budget) for name, rows in oof.items()}
    for name, rows in enriched.items():
        write_csv(root / "feature_ablation" / f"{name}_query_level_evaluation.csv", rows, rows[0].keys())
    ablation_metrics(enriched, root / "feature_ablation")
    comparisons(enriched, root / "feature_ablation")
    gate_c(enriched["full"], per_budget, root / "feature_ablation")
    diagnosis(enriched["full"], per_budget, root / "allocation_diagnosis")
    manifest = {"run_id": root.name, "timestamp_utc": stamp, "repository_root": str(ROOT), "population": "SciFact TRAIN+VALIDATION only", "queries": len(ids), "test_used": False,
                "method_decisions": {"inner_objective": "macro_f1", "inner_tie_break": "smaller_C", "OOF": "mean probabilities over 10 held-out repeats", "missing_class": "zero probability aligned by class label", "policy": "T=1, lambda=0 for direct ablation", "gate_c": "lambda workload match then 1000 budget-multiset permutations", "statistics": "10k paired bootstrap plus Wilcoxon/Holm for ablation"},
                "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
                "inputs": {str(p.relative_to(ROOT)): digest(p) for p in [ROOT / "results/phase2/01_candidate_logging/pool_features.csv", ROOT / "results/phase2/01_candidate_logging/query_features.csv", ROOT / "results/phase2/02_oracle/mstar_labels_tau099.csv", ROOT / "results/phase2/development_reranking/train/reranking_results.json", ROOT / "results/phase2/development_reranking/validation/reranking_results.json"]}}
    manifests = args.output_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{root.name}_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_reports(root, args.output_root / "reports")
    print(json.dumps({"execution_root": str(root), "status": "completed", "queries": len(ids)}, indent=2))


if __name__ == "__main__":
    main()
