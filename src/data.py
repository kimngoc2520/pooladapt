"""Load local BEIR-format datasets without downloading data."""
from __future__ import annotations

import json
from pathlib import Path


def load_beir_inputs(dataset_dir: str | Path):
    root = Path(dataset_dir)
    with (root / "corpus.jsonl").open(encoding="utf-8") as handle:
        corpus = {str(row["_id"]): {"title": row.get("title", ""), "text": row.get("text", "")} for row in map(json.loads, handle)}
    with (root / "queries.jsonl").open(encoding="utf-8") as handle:
        queries = {str(row["_id"]): row.get("text", "") for row in map(json.loads, handle)}
    return corpus, queries


def load_beir_dataset(dataset_dir: str | Path, qrels_split: str = "test"):
    root = Path(dataset_dir)
    corpus, queries = load_beir_inputs(root)
    qrels = {}
    with (root / "qrels" / f"{qrels_split}.tsv").open(encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            query_id, document_id, score = line.rstrip("\n").split("\t")
            qrels.setdefault(query_id, {})[document_id] = int(score)
    return corpus, queries, qrels
