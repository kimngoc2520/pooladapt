"""Shared, provenance-aware helpers for frozen TEST query-group analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "improvement" / "improvement_1" / "query_group_analysis"
LOCKED = ROOT / "results" / "phase2" / "evaluation" / "locked_configurations.json"
PREDICTIONS = ROOT / "results" / "phase2" / "evaluation" / "final_gpu" / "test_predictions.jsonl"
METRICS = ROOT / "results" / "phase2" / "evaluation" / "final_gpu" / "test_query_metrics.csv"
SNAPSHOT = Path(r"C:\Users\PC\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
RECON = OUT / "reconstruction"
FEATURES = OUT / "features"
GROUPS = OUT / "groups"
EVALUATION = OUT / "evaluation"
AUDIT = OUT / "audit"
SCRIPTS = OUT / "scripts"
FEATURE_SCHEMA_VERSION = "query-group-features-v1"

# This checkout's original venv points to an unavailable Windows Store alias.
# The installed packages are reused via py -3.11; no package is installed here.
VENV_SITE = ROOT / ".venv" / "Lib" / "site-packages"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if VENV_SITE.exists() and str(VENV_SITE) not in sys.path:
    sys.path.insert(0, str(VENV_SITE))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using stable compact serialization."""
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_bytes_new(path: Path, payload: bytes) -> None:
    """Atomically create a file, refusing to overwrite any existing output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_json_new(path: Path, value: Any) -> None:
    """Write formatted UTF-8 JSON atomically without overwriting."""
    write_bytes_new(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def write_csv_new(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV atomically with a stable header and no overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV table as string-valued records."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_frozen_inputs() -> tuple[dict[str, dict[str, str]], dict[str, str], list[str], dict[str, list[str]]]:
    """Load the verified corpus, queries, ordered TEST IDs, and frozen pools."""
    from src.data.loader import load_beir_dataset

    lock = json.loads(LOCKED.read_text(encoding="utf-8"))
    query_ids = [str(qid) for qid in lock["test_ids"]]
    if len(query_ids) != 300 or len(set(query_ids)) != 300:
        raise ValueError("locked_configurations.json must contain exactly 300 unique TEST IDs")
    corpus, queries, _ = load_beir_dataset(ROOT / "data" / "scifact")
    absent = set(query_ids) - set(queries)
    if absent:
        raise ValueError(f"Frozen TEST IDs are absent from queries: {sorted(absent)[:10]}")
    pools: dict[str, list[str]] = {}
    with PREDICTIONS.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["query_id"])
            if qid not in set(query_ids):
                raise ValueError(f"Frozen predictions contain unexpected TEST query ID {qid}")
            pool = [str(doc_id) for doc_id in row["pool_ids"]]
            if len(pool) != 100 or len(set(pool)) != 100:
                raise ValueError(f"Frozen pool at predictions line {line_number} is not 100 unique IDs")
            if qid in pools and pools[qid] != pool:
                raise ValueError(f"Frozen Phase 2 pool varies across methods/seeds for query {qid}")
            pools[qid] = pool
    if set(pools) != set(query_ids):
        raise ValueError(f"Frozen pools cover {len(pools)} of 300 locked TEST queries")
    missing_docs = {doc_id for pool in pools.values() for doc_id in pool} - set(corpus)
    if missing_docs:
        raise ValueError(f"Frozen pools reference missing corpus documents: {sorted(missing_docs)[:10]}")
    return corpus, queries, query_ids, pools


def get_run_id() -> str:
    """Create a human-readable UTC reconstruction identifier."""
    return datetime.now(timezone.utc).strftime("qga-%Y%m%dT%H%M%SZ")


def rbo(left: list[str], right: list[str], persistence: float = 0.9) -> float:
    """Compute extrapolated Rank-Biased Overlap for finite rankings."""
    if not 0 <= persistence < 1:
        raise ValueError("RBO persistence must be in [0, 1)")
    depth = max(len(left), len(right))
    if depth == 0:
        return 1.0
    seen_left: set[str] = set()
    seen_right: set[str] = set()
    weighted = 0.0
    overlap = 0
    for rank in range(1, depth + 1):
        if rank <= len(left):
            seen_left.add(left[rank - 1])
        if rank <= len(right):
            seen_right.add(right[rank - 1])
        overlap = len(seen_left & seen_right)
        agreement = overlap / rank
        weighted += (1 - persistence) * (persistence ** (rank - 1)) * agreement
    terminal_agreement = overlap / depth
    return weighted + (persistence**depth) * terminal_agreement

