"""Small, format-neutral dataset loading helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL dataset without imposing a dataset-specific schema."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_beir_dataset(dataset_dir: str | Path, qrels_split: str | None = None) -> tuple[
    dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, int]]
]:
    """Load an on-disk BEIR dataset without downloading or altering it.

    The returned corpus is keyed by document ID and qrels are optional so the
    retrieval backbone can be used for either development or evaluation.
    """
    root = Path(dataset_dir)
    corpus = {str(row["_id"]): row for row in load_jsonl(root / "corpus.jsonl")}
    queries = {str(row["_id"]): str(row["text"]) for row in load_jsonl(root / "queries.jsonl")}
    if qrels_split is None:
        return corpus, queries, {}

    qrels_path = root / "qrels" / f"{qrels_split}.tsv"
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    return corpus, queries, qrels
