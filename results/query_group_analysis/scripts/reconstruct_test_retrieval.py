"""Reconstruct only frozen SciFact TEST BM25, Dense, and RRF Top-100 lists."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .common import (LOCKED, METRICS, OUT, PREDICTIONS, RECON, SNAPSHOT,
                         get_run_id, load_frozen_inputs, rbo, sha256_file, sha256_json, write_bytes_new, write_json_new)
except ImportError:  # Direct script execution from this isolated scripts directory.
    from common import (LOCKED, METRICS, OUT, PREDICTIONS, RECON, SNAPSHOT,
                        get_run_id, load_frozen_inputs, rbo, sha256_file, sha256_json, write_bytes_new, write_json_new)
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXPECTED_SNAPSHOT = "RECONSTRUCTED_CANDIDATE_REVISION_NOT_VERIFIED_HISTORICAL"
EXPECTED_MODEL_FILES = {
    "config.json", "config_sentence_transformers.json", "model.safetensors", "modules.json",
    "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.txt", "1_Pooling/config.json",
}


def frame_to_new_files(frame: pd.DataFrame, stem: str) -> None:
    """Atomically write one table to CSV and Parquet without overwriting."""
    csv_path, parquet_path = RECON / f"{stem}.csv", RECON / f"{stem}.parquet"
    for path in (csv_path, parquet_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    RECON.mkdir(parents=True, exist_ok=True)
    temp_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    temp_parquet = parquet_path.with_name(f".{parquet_path.name}.tmp")
    try:
        frame.to_csv(temp_csv, index=False, encoding="utf-8-sig")
        frame.to_parquet(temp_parquet, index=False)
        temp_csv.replace(csv_path)
        temp_parquet.replace(parquet_path)
    finally:
        temp_csv.unlink(missing_ok=True)
        temp_parquet.unlink(missing_ok=True)


def percentile_overlap(left: list[str], right: list[str], depth: int) -> tuple[int, float]:
    """Return intersection count and Jaccard for top-depth prefixes."""
    a, b = set(left[:depth]), set(right[:depth])
    return len(a & b), (len(a & b) / len(a | b)) if a | b else 1.0


def validate_rows(rows: list[dict[str, Any]], query_ids: list[str], corpus_ids: set[str], rank_name: str) -> None:
    """Enforce complete and finite per-query Top-100 ranked retrieval output."""
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_id"]), []).append(row)
        if row["doc_id"] not in corpus_ids:
            raise ValueError(f"Retrieval output contains unknown corpus document {row['doc_id']}")
        if not pd.notna(row[f"{rank_name}_score"]) or not float("-inf") < float(row[f"{rank_name}_score"]) < float("inf"):
            raise ValueError(f"Retrieval output contains non-finite score for {row['query_id']}")
    if set(by_query) != set(query_ids):
        raise ValueError(f"{rank_name} query coverage is {len(by_query)}/300")
    for qid in query_ids:
        query_rows = by_query[qid]
        ranks = sorted(int(row[f"{rank_name}_rank"]) for row in query_rows)
        docs = [str(row["doc_id"]) for row in query_rows]
        if len(query_rows) != 100 or ranks != list(range(1, 101)) or len(set(docs)) != 100:
            raise ValueError(f"{rank_name} query {qid} failed Top-100 uniqueness/rank validation")


def main() -> None:
    """Run integrity gates, reconstruct rankings, and validate frozen RRF pools."""
    import hashlib
    import importlib.metadata as metadata
    import numpy as np
    import psutil
    import scipy
    import torch
    from sentence_transformers import SentenceTransformer

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run_id = get_run_id()
    output_paths = [
        RECON / name for name in (
            "bm25_test_top100.csv", "bm25_test_top100.parquet", "dense_test_top100.csv",
            "dense_test_top100.parquet", "rrf_test_top100.csv", "rrf_test_top100.parquet",
            "rrf_pool_validation.csv", "rrf_pool_validation_summary.md", "reconstruction_manifest.json",
        )
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing reconstruction outputs: " + ", ".join(existing))

    corpus_path = ROOT / "data" / "scifact" / "corpus.jsonl"
    query_path = ROOT / "data" / "scifact" / "queries.jsonl"
    qrels_path = ROOT / "data" / "scifact" / "qrels" / "test.tsv"
    locked = json.loads(LOCKED.read_text(encoding="utf-8"))
    execution = json.loads((ROOT / "results" / "phase2" / "evaluation" / "final_gpu" / "execution_manifest.json").read_text(encoding="utf-8"))
    expected_hashes = execution["input_sha256"]
    expected = {
        "data/scifact/corpus.jsonl": sha256_file(corpus_path),
        "data/scifact/queries.jsonl": sha256_file(query_path),
        "data/scifact/qrels/test.tsv": sha256_file(qrels_path),
    }
    for name, digest in expected.items():
        if expected_hashes.get(name) != digest:
            raise ValueError(f"Frozen input hash mismatch for {name}; stopping before retrieval")
    if locked.get("test_query_count") != 300 or len(locked.get("test_ids", [])) != 300:
        raise ValueError("Frozen lock does not specify exactly 300 TEST queries")

    snapshot_files = sorted(path for path in SNAPSHOT.rglob("*") if path.is_file())
    relative_files = {path.relative_to(SNAPSHOT).as_posix() for path in snapshot_files}
    missing_model_files = sorted(EXPECTED_MODEL_FILES - relative_files)
    if missing_model_files:
        raise FileNotFoundError(f"Dense model snapshot is incomplete: {missing_model_files}")
    snapshot_manifest = [
        {"path": path.relative_to(SNAPSHOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in snapshot_files
    ]
    snapshot_hash = sha256_json(snapshot_manifest)

    # Hash the precise retrieval implementation and reference inputs before any run.
    source_paths = [ROOT / "src" / "retrieval" / name for name in ("bm25.py", "dense.py", "rrf.py")]
    source_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in source_paths}
    reference_hashes = {
        "locked_configurations.json": sha256_file(LOCKED),
        "test_predictions.jsonl": sha256_file(PREDICTIONS),
        "test_query_metrics.csv": sha256_file(METRICS),
    }

    corpus, queries, query_ids, frozen_pools = load_frozen_inputs()
    corpus_bytes = len(corpus) * 384 * 4
    resource = psutil.virtual_memory()
    if resource.available < 512 * 1024 * 1024:
        raise MemoryError(f"Only {resource.available / (1024**2):.0f} MiB memory is available; refusing to start Dense encoding")
    disk_free = __import__("shutil").disk_usage(OUT.anchor or ROOT).free
    estimated_outputs = (len(query_ids) * 100 * 250)  # rows plus scores/metadata, conservative order-of-magnitude
    if disk_free < max(1024**3, estimated_outputs * 4):
        raise OSError(f"Insufficient free disk for isolated retrieval artifacts: {disk_free} bytes")

    # The original document batch is 32. CPU-only runtime plus <1 GiB free RAM
    # warrants the permitted controlled fallback to batch 8; scoring remains exact.
    batch_size = 8
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model = SentenceTransformer(str(SNAPSHOT), device="cpu", model_kwargs={"local_files_only": True})
    model.max_seq_length = 256
    model.eval()
    model_revision = EXPECTED_SNAPSHOT
    if int(model.get_sentence_embedding_dimension()) != 384:
        raise ValueError("Local model embedding dimension is not 384")
    if int(model.max_seq_length) != 256:
        raise ValueError("Local model maximum sequence length is not 256")

    from src.retrieval import BM25Retriever, DenseRetriever, fuse_ranked_lists

    bm25_started = time.perf_counter()
    bm25 = BM25Retriever(corpus)
    bm25_rows: list[dict[str, Any]] = []
    dense_rows: list[dict[str, Any]] = []
    rrf_rows: list[dict[str, Any]] = []
    reconstructed_pools: dict[str, list[str]] = {}
    dense = DenseRetriever(corpus, model_name=str(SNAPSHOT), model=model, batch_size=batch_size)
    encode_completed_at = time.perf_counter()
    for qid in query_ids:
        query = queries[qid]
        sparse = bm25.retrieve(query, top_n=100)
        dense_ranked = dense.retrieve(query, top_n=100)
        for row in sparse:
            bm25_rows.append({"query_id": qid, "doc_id": str(row["id"]), "bm25_rank": int(row["rank"]),
                              "bm25_score": float(row["score"]), "retrieval_method": "BM25Okapi",
                              "reconstruction_run_id": run_id})
        for row in dense_ranked:
            dense_rows.append({"query_id": qid, "doc_id": str(row["id"]), "dense_rank": int(row["rank"]),
                               "dense_score": float(row["score"]), "retrieval_method": "DenseDotProductFloat32",
                               "reconstruction_run_id": run_id})
        fused = fuse_ranked_lists({"bm25": sparse, "dense": dense_ranked}, top_n=100, k=60)
        reconstructed_pools[qid] = [str(row["id"]) for row in fused]
        sparse_map = {str(row["id"]): row for row in sparse}
        dense_map = {str(row["id"]): row for row in dense_ranked}
        for row in fused:
            doc_id = str(row["id"])
            b, d = sparse_map.get(doc_id), dense_map.get(doc_id)
            bm_contrib = 1 / (60 + int(b["rank"])) if b else 0.0
            dense_contrib = 1 / (60 + int(d["rank"])) if d else 0.0
            rrf_rows.append({
                "query_id": qid, "doc_id": doc_id, "rrf_rank": int(row["rank"]), "rrf_score": float(row["rrf_score"]),
                "bm25_rank": int(b["rank"]) if b else None, "bm25_score": float(b["score"]) if b else None,
                "dense_rank": int(d["rank"]) if d else None, "dense_score": float(d["score"]) if d else None,
                "rrf_contribution_bm25": bm_contrib, "rrf_contribution_dense": dense_contrib,
                "in_bm25_top100": bool(b), "in_dense_top100": bool(d), "reconstruction_run_id": run_id,
            })
    if set(reconstructed_pools) != set(query_ids):
        raise ValueError("RRF reconstruction did not cover the frozen TEST IDs exactly")
    validate_rows(bm25_rows, query_ids, set(corpus), "bm25")
    validate_rows(dense_rows, query_ids, set(corpus), "dense")
    if any(len(reconstructed_pools[qid]) != 100 or len(set(reconstructed_pools[qid])) != 100 for qid in query_ids):
        raise ValueError("Reconstructed RRF output failed unique Top-100 integrity")

    from scipy.stats import kendalltau, spearmanr

    validation_rows: list[dict[str, Any]] = []
    for qid in query_ids:
        rebuilt, frozen = reconstructed_pools[qid], frozen_pools[qid]
        shared = set(rebuilt) & set(frozen)
        rb_rank = {doc: index + 1 for index, doc in enumerate(rebuilt)}
        fr_rank = {doc: index + 1 for index, doc in enumerate(frozen)}
        r10, j10 = percentile_overlap(rebuilt, frozen, 10)
        r20, j20 = percentile_overlap(rebuilt, frozen, 20)
        rbo90 = rbo(rebuilt, frozen, 0.9)
        rbo95 = rbo(rebuilt, frozen, 0.95)
        if len(shared) > 1:
            rank_a = [rb_rank[d] for d in sorted(shared)]
            rank_b = [fr_rank[d] for d in sorted(shared)]
            spear = float(spearmanr(rank_a, rank_b).statistic)
            kendall = float(kendalltau(rank_a, rank_b).statistic)
        else:
            spear, kendall = float("nan"), float("nan")
        validation_rows.append({
            "query_id": qid,
            "exact_ordered_top100_match": rebuilt == frozen,
            "exact_top100_set_match": set(rebuilt) == set(frozen),
            "jaccard_at_100": len(shared) / len(set(rebuilt) | set(frozen)),
            "top10_overlap_count": r10, "top10_overlap_fraction": r10 / 10,
            "top10_jaccard": j10,
            "top20_overlap_count": r20, "top20_overlap_fraction": r20 / 20,
            "top20_jaccard": j20,
            "shared_document_count": len(shared),
            "shared_rank_spearman": spear, "shared_rank_kendall": kendall,
            "shared_mean_absolute_rank_displacement": sum(abs(rb_rank[d] - fr_rank[d]) for d in shared) / len(shared) if shared else None,
            "rbo_p_0_90": rbo90, "rbo_p_0_95": rbo95,
            "missing_reconstructed_doc_ids": ";".join(sorted(set(frozen) - set(rebuilt))),
            "extra_reconstructed_doc_ids": ";".join(sorted(set(rebuilt) - set(frozen))),
            "first_order_mismatch_rank": next((i + 1 for i, (a, b) in enumerate(zip(rebuilt, frozen)) if a != b), None),
            "reconstruction_run_id": run_id,
        })
    exact_count = sum(row["exact_ordered_top100_match"] for row in validation_rows)
    exact_set_count = sum(row["exact_top100_set_match"] for row in validation_rows)
    status = "GATE_A_EXACT_MATCH" if exact_count == 300 else ("GATE_B_PARTIAL_MATCH" if exact_count > 0 else "GATE_C_FAILED_VALIDATION")

    # Output schema validated in memory first; only now atomically persist artifacts.
    bm25_frame = pd.DataFrame(bm25_rows)
    dense_frame = pd.DataFrame(dense_rows)
    rrf_frame = pd.DataFrame(rrf_rows)
    validation_frame = pd.DataFrame(validation_rows)
    for frame, rank_col, score_col in ((bm25_frame, "bm25_rank", "bm25_score"), (dense_frame, "dense_rank", "dense_score"), (rrf_frame, "rrf_rank", "rrf_score")):
        if not frame[score_col].map(lambda v: pd.notna(v) and float("-inf") < float(v) < float("inf")).all():
            raise ValueError(f"Non-finite values found in {score_col}")
        if frame.duplicated(["query_id", "doc_id"]).any():
            raise ValueError(f"Duplicate query/document pair in {rank_col} output")

    def save_table(frame: pd.DataFrame, stem: str) -> None:
        csv_path, parquet_path = RECON / f"{stem}.csv", RECON / f"{stem}.parquet"
        for path in (csv_path, parquet_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        RECON.mkdir(parents=True, exist_ok=True)
        temp_csv, temp_parquet = csv_path.with_name(f".{csv_path.name}.tmp"), parquet_path.with_name(f".{parquet_path.name}.tmp")
        try:
            frame.to_csv(temp_csv, index=False, encoding="utf-8-sig")
            frame.to_parquet(temp_parquet, index=False)
            temp_csv.replace(csv_path)
            temp_parquet.replace(parquet_path)
        finally:
            temp_csv.unlink(missing_ok=True)
            temp_parquet.unlink(missing_ok=True)

    save_table(bm25_frame, "bm25_test_top100")
    save_table(dense_frame, "dense_test_top100")
    save_table(rrf_frame, "rrf_test_top100")
    save_table(validation_frame, "rrf_pool_validation")
    validation = {
        "classification": status,
        "queries": 300,
        "exact_ordered_match_count": exact_count,
        "exact_ordered_match_rate": exact_count / 300,
        "exact_set_match_count": exact_set_count,
        "exact_set_match_rate": exact_set_count / 300,
        "mean_jaccard_at_100": float(validation_frame["jaccard_at_100"].mean()),
        "median_jaccard_at_100": float(validation_frame["jaccard_at_100"].median()),
        "mean_top10_overlap_fraction": float(validation_frame["top10_overlap_fraction"].mean()),
        "mean_top20_overlap_fraction": float(validation_frame["top20_overlap_fraction"].mean()),
        "mismatch_count": 300 - exact_count,
        "mismatch_by_first_mismatch_rank": validation_frame.loc[~validation_frame["exact_ordered_top100_match"], "first_order_mismatch_rank"].value_counts(dropna=False).sort_index().to_dict(),
        "limitations": [
            "A matching reconstructed RRF list does not prove historical BM25/Dense scores or the historical Dense model revision.",
            "This local Dense snapshot is a reconstructed candidate revision, not a verified historical revision.",
        ],
    }
    report = [
        "# Reconstructed RRF Pool Validation", "",
        f"**Validation gate:** `{status}`", "",
        f"- Exact ordered Top-100 match: {exact_count}/300 ({exact_count / 300:.1%}).",
        f"- Exact Top-100 set match: {exact_set_count}/300 ({exact_set_count / 300:.1%}).",
        f"- Jaccard@100: mean {validation['mean_jaccard_at_100']:.4f}; median {validation['median_jaccard_at_100']:.4f}.",
        f"- Mean Top-10 overlap: {validation['mean_top10_overlap_fraction']:.1%}; mean Top-20 overlap: {validation['mean_top20_overlap_fraction']:.1%}.",
        f"- Queries with any ordered mismatch: {300 - exact_count}.",
        "- Per-query details, rank correlations, missing/extra IDs, and RBO at p=0.90/0.95 are in `rrf_pool_validation.csv`.",
        "", "## Interpretation", "",
        "The reconstructed ordering is classified by the gate above. Even Gate A establishes only candidate ordering agreement with frozen Phase 2 pools; it does not establish historical raw component scores or Dense model revision identity.",
        "The Dense model is explicitly labeled `RECONSTRUCTED_CANDIDATE_REVISION_NOT_VERIFIED_HISTORICAL`. Any component score or component feature is reconstructed.",
    ]
    write_bytes_new(RECON / "rrf_pool_validation_summary.md", ("\n".join(report) + "\n").encode("utf-8"))

    snapshot_config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    manifest = {
        "retrieval_run_id": run_id,
        "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_status": __import__("subprocess").check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "scope": "Only 300 frozen SciFact TEST queries; no TRAIN/VALIDATION query retrieval.",
        "query_ids": {"count": len(query_ids), "sha256_ordered": sha256_json(query_ids), "sha256_sorted": sha256_json(sorted(query_ids))},
        "corpus_sha256": expected["data/scifact/corpus.jsonl"],
        "queries_sha256": expected["data/scifact/queries.jsonl"],
        "test_qrels_sha256_integrity_only": expected["data/scifact/qrels/test.tsv"],
        "frozen_reference_hashes": reference_hashes,
        "source_sha256": source_hashes,
        "dense_snapshot": {
            "path": str(SNAPSHOT), "label": model_revision, "manifest_sha256": snapshot_hash,
            "files": snapshot_manifest, "embedding_dimension": 384, "max_sequence_length": 256,
            "normalized_embeddings": True, "similarity": "float32 dot product over normalized embeddings",
            "config": snapshot_config,
        },
        "bm25": {"implementation": "src/retrieval/bm25.py::BM25Retriever", "package": "rank-bm25", "version": metadata.version("rank-bm25"),
                 "k1": 1.5, "b": 0.75, "epsilon": 0.25, "tokenization": "lowercase then Python re \\w+", "document_text": "title + single space + text", "tie_break": "descending score, ascending document ID"},
        "dense": {"implementation": "src/retrieval/dense.py::DenseRetriever", "package_versions": {name: metadata.version(name) for name in ("sentence-transformers", "transformers", "tokenizers", "torch", "numpy")},
                  "model_name": "sentence-transformers/all-MiniLM-L6-v2", "revision_status": model_revision,
                  "device": "cpu", "precision": "float32", "requested_document_batch_size": 32, "actual_document_batch_size": batch_size,
                  "query_batch_behavior": "existing implementation default: single query, no explicit batch_size", "thread_count": torch.get_num_threads(),
                  "hardware": {"cpu_count": os.cpu_count(), "cuda_available": torch.cuda.is_available(), "available_memory_before_encoding_bytes": resource.available,
                               "estimated_float32_corpus_embeddings_bytes": corpus_bytes, "free_disk_bytes_before_run": disk_free}},
        "rrf": {"implementation": "src/retrieval/rrf.py::fuse_ranked_lists", "k": 60, "source_top_n": 100, "output_top_n": 100,
                "tie_break": "descending RRF score, ascending document ID"},
        "integrity": {"bm25_valid": True, "dense_valid": True, "rrf_valid": True, "pool_validation": validation,
                      "bm25_rows": len(bm25_rows), "dense_rows": len(dense_rows), "rrf_rows": len(rrf_rows),
                      "elapsed_seconds": time.perf_counter() - bm25_started,
                      "dense_document_encoding_elapsed_seconds": encode_completed_at - bm25_started},
        "frozen_outputs_modified": False,
    }
    write_json_new(RECON / "reconstruction_manifest.json", manifest)
    print(json.dumps({"run_id": run_id, "status": status, "dense_revision_label": model_revision,
                      "exact_ordered_match": f"{exact_count}/300", "bm25_rows": len(bm25_rows), "dense_rows": len(dense_rows),
                      "rrf_rows": len(rrf_rows), "elapsed_seconds": manifest["integrity"]["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
