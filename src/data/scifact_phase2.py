"""Locked SciFact Phase 2 development-split utilities."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.data.split import split_items


SPLIT_SEED = 42
TRAIN_RATIO = 0.8


def read_qrel_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or set(rows[0]) != {"query-id", "corpus-id", "score"}:
        raise ValueError(f"{path} must use the BEIR qrels schema")
    return rows


def query_ids(rows: list[dict[str, str]]) -> set[str]:
    return {str(row["query-id"]) for row in rows}


def split_original_train(original_train_ids: set[str]) -> tuple[list[str], list[str]]:
    """Create the fixed 80/20 split from sorted unique original-train IDs."""
    train, validation = split_items(sorted(original_train_ids), validation_fraction=.2, seed=SPLIT_SEED)
    return sorted(train), sorted(validation)


def validate_split(train_ids: set[str], validation_ids: set[str], original_train_ids: set[str], test_ids: set[str]) -> None:
    if train_ids & validation_ids:
        raise ValueError("TRAIN and VALIDATION query IDs overlap")
    if train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("development split contains locked TEST query IDs")
    if train_ids | validation_ids != original_train_ids:
        raise ValueError("TRAIN and VALIDATION do not completely assign original TRAIN query IDs")


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_split(dataset_dir: str | Path) -> dict[str, object]:
    """Create or verify locked IDs and filtered qrels without touching originals."""
    root = Path(dataset_dir)
    qrels_dir, split_dir = root / "qrels", root / "splits"
    original_train_path, test_path = qrels_dir / "train.tsv", qrels_dir / "test.tsv"
    source_rows, test_rows = read_qrel_rows(original_train_path), read_qrel_rows(test_path)
    source_ids, test_ids = query_ids(source_rows), query_ids(test_rows)
    expected_train, expected_validation = split_original_train(source_ids)
    train_ids_path, validation_ids_path, metadata_path = split_dir / "train_query_ids.txt", split_dir / "validation_query_ids.txt", split_dir / "split_metadata.json"
    expected_metadata = {
        "dataset": "scifact", "source_split": "original_train", "split_ratio": "80/20",
        "train_ratio": TRAIN_RATIO, "validation_ratio": .2, "random_seed": SPLIT_SEED,
        "split_level": "query", "train_query_count": len(expected_train),
        "validation_query_count": len(expected_validation), "overlap_count": 0,
        "test_overlap_count": 0, "phase1_random_baseline_seed": 0,
        "creation_method": "sorted unique original-train query IDs; random.Random(42).shuffle; split_items(validation_fraction=0.2)",
    }
    existing = [train_ids_path.exists(), validation_ids_path.exists(), metadata_path.exists()]
    if any(existing) and not all(existing):
        raise ValueError("SciFact split lock is incomplete; refusing to regenerate it")
    if all(existing):
        train_ids, validation_ids = _read_ids(train_ids_path), _read_ids(validation_ids_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if train_ids != expected_train or validation_ids != expected_validation or metadata != expected_metadata:
            raise ValueError("existing SciFact split lock does not match the fixed Phase 2 protocol")
    else:
        split_dir.mkdir(parents=True, exist_ok=True)
        train_ids_path.write_text("\n".join(expected_train) + "\n", encoding="utf-8")
        validation_ids_path.write_text("\n".join(expected_validation) + "\n", encoding="utf-8")
        metadata_path.write_text(json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        train_ids, validation_ids = expected_train, expected_validation
    validate_split(set(train_ids), set(validation_ids), source_ids, test_ids)
    _write_filtered_qrels(qrels_dir / "train_split.tsv", source_rows, set(train_ids))
    _write_filtered_qrels(qrels_dir / "validation.tsv", source_rows, set(validation_ids))
    return {**expected_metadata, "train_ids": train_ids, "validation_ids": validation_ids,
            "source_qrels_sha256": hashlib.sha256(original_train_path.read_bytes()).hexdigest()}


def _write_filtered_qrels(path: Path, source_rows: list[dict[str, str]], allowed_ids: set[str]) -> None:
    filtered = [row for row in source_rows if row["query-id"] in allowed_ids]
    if query_ids(filtered) != allowed_ids:
        raise ValueError(f"cannot produce complete filtered qrels at {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query-id", "corpus-id", "score"], delimiter="\t")
        writer.writeheader()
        writer.writerows(filtered)
