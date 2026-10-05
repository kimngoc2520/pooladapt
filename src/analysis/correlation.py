"""TRAIN+VALIDATION feature--M* association analysis (associations, not causes)."""
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TAUS = ("095", "098", "099")
CANDIDATE_MEAN_FEATURES = ("bm25_rank", "dense_rank", "rank_displacement", "bm25_score", "dense_score", "local_rrf_margin")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    order, adjusted, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _unique_rows(rows: Sequence[Mapping[str, str]], source: str, exclude: Sequence[str] = ()) -> dict[str, dict[str, float | str]]:
    by_query: dict[str, dict[str, float | str]] = {}
    for raw in rows:
        if "query_id" not in raw:
            raise ValueError(f"{source} is missing query_id")
        query_id = str(raw["query_id"])
        if query_id in by_query:
            raise ValueError(f"{source} contains duplicate query ID {query_id!r}")
        row: dict[str, float | str] = {"query_id": query_id}
        for key, value in raw.items():
            if key in ("query_id", *exclude) or value in ("", None):
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                continue
        by_query[query_id] = row
    return by_query


def load_feature_sets(candidate_path: str | Path, pool_path: str | Path, query_path: str | Path) -> dict[str, dict[str, dict[str, float | str]]]:
    """Load query values, aggregating selected candidate features by mean over N=100."""
    grouped: dict[str, list[dict[str, float]]] = {}
    candidate_ids: set[tuple[str, str]] = set()
    for raw in read_rows(candidate_path):
        query_id, doc_id = str(raw.get("query_id", "")), str(raw.get("doc_id", ""))
        if not query_id or not doc_id:
            raise ValueError("candidate_features.csv requires query_id and doc_id")
        pair = (query_id, doc_id)
        if pair in candidate_ids:
            raise ValueError(f"candidate_features.csv contains duplicate candidate pair {pair!r}")
        candidate_ids.add(pair)
        numeric: dict[str, float] = {}
        for name in CANDIDATE_MEAN_FEATURES:
            value = raw.get(name)
            if value not in ("", None):
                try:
                    numeric[name] = float(value)
                except (TypeError, ValueError):
                    pass
        grouped.setdefault(query_id, []).append(numeric)
    candidate: dict[str, dict[str, float | str]] = {}
    for query_id, rows in grouped.items():
        if len(rows) != 100:
            raise ValueError(f"candidate_features.csv query {query_id!r} has {len(rows)} rows; expected N=100")
        summary: dict[str, float | str] = {"query_id": query_id}
        for name in CANDIDATE_MEAN_FEATURES:
            values = [row[name] for row in rows if name in row and np.isfinite(row[name])]
            if values:
                summary[name] = float(np.mean(values))
        candidate[query_id] = summary
    return {
        "candidate": candidate,
        "pool": _unique_rows(read_rows(pool_path), "pool_features.csv"),
        "query": _unique_rows(read_rows(query_path), "query_features.csv", exclude=("entity_count",)),
    }


def _labels_for_tau(labels: Sequence[Mapping[str, str]], tau: str) -> dict[str, float]:
    values: dict[str, float] = {}
    expected_tau = int(tau) / 100
    for row in labels:
        if not {"query_id", "M_star", "tau"} <= row.keys():
            raise ValueError("official M* labels require query_id, M_star, and tau columns")
        query_id = str(row["query_id"])
        if query_id in values:
            raise ValueError(f"M* labels contain duplicate query ID {query_id!r}")
        if not np.isclose(float(row["tau"]), expected_tau):
            raise ValueError(f"M* label tau does not match requested tau={expected_tau:.2f}")
        values[query_id] = float(row["M_star"])
    if not values:
        raise ValueError("M* labels must not be empty")
    return values


def analyze(features: Mapping[str, Mapping[str, Mapping[str, float | str]]], labels: Sequence[Mapping[str, str]], tau: str) -> list[dict[str, float | str | bool | int]]:
    """Compute Spearman + Holm with a hard one-row-per-query assertion."""
    if tau not in TAUS:
        raise ValueError(f"unsupported tau={tau}")
    y, results = _labels_for_tau(labels, tau), []
    for level, by_query in features.items():
        if set(by_query) != set(y):
            raise ValueError(f"{level} feature query IDs must exactly match official M* labels")
        aggregation = "mean_N100" if level == "candidate" else "none"
        names = sorted({name for row in by_query.values() for name in row if name != "query_id"})
        for name in names:
            usable = [(query_id, float(row[name]), y[query_id]) for query_id, row in by_query.items() if query_id in y and name in row and np.isfinite(float(row[name])) and np.isfinite(y[query_id])]
            row_count, unique_query_count = len(usable), len({pair[0] for pair in usable})
            if row_count != unique_query_count:
                raise ValueError(f"{level}.{name} is not one row per query ({row_count} rows, {unique_query_count} queries)")
            if row_count < 3:
                rho, p_value = 0.0, 1.0
            else:
                x_values, y_values = [pair[1] for pair in usable], [pair[2] for pair in usable]
                if np.std(x_values) == 0 or np.std(y_values) == 0:
                    rho, p_value = 0.0, 1.0
                else:
                    rho, p_value = spearmanr(x_values, y_values)
                    if not np.isfinite(rho) or not np.isfinite(p_value):
                        rho, p_value = 0.0, 1.0
            results.append({"feature_name": name, "feature_level": level, "aggregation_method": aggregation, "tau": f"{int(tau) / 100:.2f}", "spearman_rho": float(rho), "p_value": float(p_value), "abs_rho": abs(float(rho)), "number_of_rows_used": row_count, "number_of_unique_queries": unique_query_count})
    for row, p_holm in zip(results, holm_adjust([float(row["p_value"]) for row in results])):
        row["p_holm"] = p_holm
        row["selected"] = bool(float(row["abs_rho"]) > 0.2 and p_holm < 0.05)
    return results


def multicollinearity(features: Mapping[str, Mapping[str, Mapping[str, float | str]]], threshold: float = .8) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for level, data in features.items():
        names = sorted({key for row in data.values() for key in row if key != "query_id"})
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                pairs = [(float(row[first]), float(row[second])) for row in data.values() if first in row and second in row]
                if len(pairs) > 2:
                    correlation = np.corrcoef(np.asarray(pairs).T)[0, 1]
                    if np.isfinite(correlation) and abs(correlation) >= threshold:
                        rows.append({"feature_level": level, "feature_a": first, "feature_b": second, "pearson_r": float(correlation)})
    return rows
