"""Generate official Phase 2.2 M* labels from TRAIN + VALIDATION only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.oracle_mstar import TAUS, analyse_mstar, compute_mstar_labels, extract_rankings, write_analysis, write_labels


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            query_id, document_id, score = line.rstrip("\n").split("\t")
            qrels.setdefault(str(query_id), {})[str(document_id)] = int(score)
    if not qrels:
        raise ValueError(f"No qrels found in {path}")
    return qrels


def load_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    split = str(result.get("split", result.get("qrels_split", ""))).lower()
    if split == "test":
        raise ValueError(f"Official M* generation rejects TEST artifact: {path}")
    return result


def merge_qrels(train: dict[str, dict[str, int]], validation: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError(f"Train and validation qrels overlap: {sorted(overlap)[:5]}")
    return {**train, **validation}


def merge_rankings(*parts: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for part in parts:
        overlap = set(merged) & set(part)
        if overlap:
            raise ValueError(f"Train and validation results overlap: {sorted(overlap)[:5]}")
        merged.update(part)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-qrels", required=True, type=Path)
    parser.add_argument("--validation-qrels", required=True, type=Path)
    parser.add_argument("--train-results", required=True, type=Path)
    parser.add_argument("--validation-results", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2/02_oracle"))
    args = parser.parse_args()

    qrels = merge_qrels(load_qrels(args.train_qrels), load_qrels(args.validation_qrels))
    train_budgets, train_full = extract_rankings(load_result(args.train_results))
    validation_budgets, validation_full = extract_rankings(load_result(args.validation_results))
    budgets = {budget: merge_rankings(train_budgets[budget], validation_budgets[budget]) for budget in train_budgets}
    full = merge_rankings(train_full, validation_full)
    for tau in TAUS:
        suffix = f"{tau:.2f}".replace(".", "")
        rows = compute_mstar_labels(qrels, budgets, full, tau)
        analysis = analyse_mstar(rows)
        analysis.update({"tau": tau, "source_splits": ["train", "validation"]})
        write_labels(rows, args.output_dir / f"mstar_labels_tau{suffix}.csv")
        write_analysis(analysis, args.output_dir / f"mstar_analysis_tau{suffix}.json")
        print(f"tau={tau:.2f}: G1_passes={analysis['G1_passes']}; wrote {len(rows)} labels")


if __name__ == "__main__":
    main()
