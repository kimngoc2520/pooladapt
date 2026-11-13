"""Extract independent frozen-pool and reconstructed-component TEST features."""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .common import (OUT, RECON, ROOT, FEATURE_SCHEMA_VERSION, load_frozen_inputs, read_csv, rbo, sha256_file, write_csv_new)
except ImportError:
    from common import (OUT, RECON, ROOT, FEATURE_SCHEMA_VERSION, load_frozen_inputs, read_csv, rbo, sha256_file, write_csv_new)
import numpy as np
import pandas as pd

OUT_FEATURES = OUT / "features"
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
FORBIDDEN = ("ndcg", "recall", "mrr", "qrel", "relevance", "label", "pooladapt", "fixed_prefix", "latency", "cost", "rerank")


def token_set(text: str) -> set[str]:
    """Normalize document text with lowercase Unicode word tokenization."""
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    """Return lexical token-set Jaccard, defining two empty sets as zero."""
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def overlap_metrics(left: list[str], right: list[str], depth: int) -> tuple[int, float]:
    """Return top-depth intersection and Jaccard without imputing absent ranks."""
    if depth < 1:
        raise ValueError("top-k depth must be positive")
    a, b = set(left[:depth]), set(right[:depth])
    union = a | b
    return len(a & b), (len(a & b) / len(union) if union else 1.0)


def shared_rank_displacement(left_rank: dict[str, int], right_rank: dict[str, int]) -> list[int]:
    """Return absolute rank changes for shared IDs only; missing ranks stay missing."""
    return [abs(left_rank[doc_id] - right_rank[doc_id]) for doc_id in left_rank.keys() & right_rank.keys()]


def pairwise_summary(docs: list[set[str]]) -> tuple[float | None, float | None, float]:
    """Return mean pairwise Jaccard, mean nearest-neighbor Jaccard, empty ratio."""
    if not docs:
        return None, None, 1.0
    total = 0.0
    count = 0
    nearest = [0.0] * len(docs)
    for i, left in enumerate(docs):
        for j in range(i):
            similarity = jaccard(left, docs[j])
            total += similarity
            count += 1
            nearest[i] = max(nearest[i], similarity)
            nearest[j] = max(nearest[j], similarity)
    return (total / count if count else None), (float(np.mean(nearest)) if len(docs) > 1 else None), sum(not x for x in docs) / len(docs)


def add_score_stats(row: dict[str, Any], name: str, values: list[float]) -> None:
    """Add retriever-specific score distribution statistics to one feature row."""
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return
    for statistic, value in (
        ("mean", np.mean(x)), ("std", np.std(x, ddof=0)), ("median", np.median(x)),
        ("min", np.min(x)), ("max", np.max(x)), ("p25", np.quantile(x, .25)),
        ("p75", np.quantile(x, .75)), ("p90", np.quantile(x, .90)), ("p95", np.quantile(x, .95)),
    ):
        row[f"{name}_score_{statistic}"] = float(value)
    row[f"{name}_score_iqr"] = float(np.quantile(x, .75) - np.quantile(x, .25))
    for depth in (10, 20, 100):
        row[f"{name}_score_top{depth}_mean"] = float(np.mean(x[:depth]))
    row[f"{name}_score_top10_minus_top100_mean"] = row[f"{name}_score_top10_mean"] - row[f"{name}_score_top100_mean"]
    row[f"{name}_score_top1_minus_top10_mean"] = float(x[0] - np.mean(x[:10]))
    row[f"{name}_score_top1_minus_top2"] = float(x[0] - x[1]) if len(x) > 1 else None


def to_frame_new(frame: pd.DataFrame, path: Path, parquet: bool = False) -> None:
    """Write a new feature table atomically without replacing prior work."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    try:
        if parquet:
            frame.to_parquet(temp, index=False)
        else:
            frame.to_csv(temp, index=False, encoding="utf-8-sig")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    """Compute features without ever opening qrels or evaluation outcomes."""
    required = [RECON / "reconstruction_manifest.json", RECON / "bm25_test_top100.csv",
                RECON / "dense_test_top100.csv", RECON / "rrf_test_top100.csv",
                RECON / "rrf_pool_validation.csv"]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("Complete retrieval reconstruction artifacts are required before feature extraction")
    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    retrieval_gate = manifest["integrity"]["pool_validation"]["classification"]
    corpus, queries, query_ids, frozen_pools = load_frozen_inputs()
    bm_rows, dense_rows, rrf_rows = (read_csv(path) for path in required[1:4])
    by_query: dict[str, dict[str, list[dict[str, str]]]] = {}
    for source, rows in (("bm25", bm_rows), ("dense", dense_rows), ("rrf", rrf_rows)):
        for row in rows:
            by_query.setdefault(row["query_id"], {})[source] = [] if source not in by_query.get(row["query_id"], {}) else by_query[row["query_id"]][source]
            by_query[row["query_id"]][source].append(row)
    if set(by_query) != set(query_ids):
        raise ValueError("Reconstruction component files do not have exact TEST query coverage")

    from src.retrieval.bm25 import BM25Retriever, tokenize
    bm25 = BM25Retriever(corpus)
    original_feature_names = list(__import__("src.pooladapt.selector", fromlist=["FROZEN_FEATURES"]).FROZEN_FEATURES)
    if len(original_feature_names) != 13:
        raise ValueError("Project frozen feature schema no longer contains the expected 13 features")
    source_metric = read_csv(ROOT / "results" / "phase2" / "01_candidate_logging" / "pool_features.csv")
    source_feature_impl_hash = sha256_file(ROOT / "src" / "evaluation" / "phase2_inputs.py")

    output: list[dict[str, Any]] = []
    original_rows: list[dict[str, Any]] = []
    for qid in query_ids:
        pool = frozen_pools[qid]
        tokenized: dict[str, set[str]] = {}
        lengths: list[int] = []
        empty_flags: list[int] = []
        for doc_id in pool:
            doc = corpus[doc_id]
            normalized = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            tokens = token_set(normalized)
            tokenized[doc_id] = tokens
            lengths.append(len(TOKEN_RE.findall(normalized)))
            empty_flags.append(not bool(normalized))
        direct: dict[str, Any] = {
            "query_id": qid,
            "feature_source": "frozen_phase2_pool_direct",
            "component_feature_source": "reconstructed_bm25_dense_rrf",
            "retrieval_reconstruction_id": manifest["retrieval_run_id"],
            "rrf_alignment_gate": retrieval_gate,
            "dense_snapshot_hash": manifest["dense_snapshot"]["manifest_sha256"],
            "corpus_hash": manifest["corpus_sha256"],
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "frozen_pool_size": len(pool),
            "pool_top10_position_mean": 5.5,
            "pool_top20_position_mean": 10.5,
            "pool_top100_position_mean": 50.5,
            "pool_top10_position_variance": float(np.var(np.arange(1, 11), ddof=0)),
            "pool_top20_position_variance": float(np.var(np.arange(1, 21), ddof=0)),
            "pool_top10_position_share_of_rank_sum": sum(range(1, 11)) / sum(range(1, 101)),
            "pool_rank_position_gap_mean_top10_top100": 50.5 - 5.5,
            "pool_top20_mean_pairwise_lexical_jaccard": None,
            "pool_top100_mean_pairwise_lexical_jaccard": None,
            "pool_top20_mean_max_lexical_jaccard": None,
            "pool_top100_mean_max_lexical_jaccard": None,
            "pool_top20_empty_document_ratio": None,
            "pool_top100_empty_document_ratio": None,
            "pool_top20_exact_duplicate_content_ratio": None,
            "pool_top100_exact_duplicate_content_ratio": None,
            "pool_top20_mean_token_length": float(np.mean(lengths[:20])),
            "pool_top100_mean_token_length": float(np.mean(lengths)),
            "pool_top20_median_token_length": float(np.median(lengths[:20])),
            "pool_top100_median_token_length": float(np.median(lengths)),
            "pool_top20_unique_token_ratio": len(set().union(*(tokenized[d] for d in pool[:20]))) / max(1, sum(len(tokenized[d]) for d in pool[:20])),
            "pool_top100_unique_token_ratio": len(set().union(*(tokenized[d] for d in pool))) / max(1, sum(len(tokenized[d]) for d in pool)),
        }
        for depth in (20, 100):
            selected = [tokenized[doc_id] for doc_id in pool[:depth]]
            mean_pair, mean_max, empty_ratio = pairwise_summary(selected)
            direct[f"pool_top{depth}_mean_pairwise_lexical_jaccard"] = mean_pair
            direct[f"pool_top{depth}_mean_max_lexical_jaccard"] = mean_max
            direct[f"pool_top{depth}_empty_document_ratio"] = empty_ratio
            joined = [" ".join(sorted(tokenized[doc_id])) for doc_id in pool[:depth]]
            direct[f"pool_top{depth}_exact_duplicate_content_ratio"] = 1 - len(set(joined)) / depth

        b = sorted(by_query[qid]["bm25"], key=lambda x: int(x["bm25_rank"]))
        d = sorted(by_query[qid]["dense"], key=lambda x: int(x["dense_rank"]))
        rr = sorted(by_query[qid]["rrf"], key=lambda x: int(x["rrf_rank"]))
        b_ids, d_ids, r_ids = ([str(x["doc_id"]) for x in rows] for rows in (b, d, rr))
        b_rank = {x["doc_id"]: int(x["bm25_rank"]) for x in b}
        d_rank = {x["doc_id"]: int(x["dense_rank"]) for x in d}
        shared = set(b_rank) & set(d_rank)
        common_ranks = sorted(shared)
        rank_b = [b_rank[doc] for doc in common_ranks]
        rank_d = [d_rank[doc] for doc in common_ranks]
        rho = float(np.corrcoef(rank_b, rank_d)[0, 1]) if len(shared) > 1 and np.std(rank_b) and np.std(rank_d) else None
        displacement = shared_rank_displacement(b_rank, d_rank)
        for depth in (10, 20, 50, 100):
            intersection, overlap_jaccard = overlap_metrics(b_ids, d_ids, depth)
            direct[f"component_top{depth}_intersection"] = intersection
            direct[f"component_top{depth}_jaccard"] = overlap_jaccard
        direct.update({
            "component_rbo_p_0_90": rbo(b_ids, d_ids, .90), "component_rbo_p_0_95": rbo(b_ids, d_ids, .95),
            "component_shared_top100_count": len(shared), "component_bm25_only_top100_count": len(set(b_ids) - set(d_ids)),
            "component_dense_only_top100_count": len(set(d_ids) - set(b_ids)),
            "component_shared_rank_spearman": rho,
            "component_shared_mean_absolute_rank_displacement": float(np.mean(displacement)) if displacement else None,
            "component_shared_median_absolute_rank_displacement": float(np.median(displacement)) if displacement else None,
            "component_overlap_top10_within_top100_shared": len(set(b_ids[:10]) & set(d_ids[:10])) / max(1, len(shared)),
            "component_missing_rank_convention": "Metrics use only documents present in both Top-100 lists; absent ranks are not imputed.",
        })
        add_score_stats(direct, "bm25", [float(x["bm25_score"]) for x in b])
        add_score_stats(direct, "dense", [float(x["dense_score"]) for x in d])
        add_score_stats(direct, "rrf_reconstructed", [float(x["rrf_score"]) for x in rr])

        # Reproduce the frozen 13 formulas exactly, but only when reconstructed
        # Top-100 order aligns to the frozen reference for all TEST queries.
        original_values: dict[str, float | None] = {name: None for name in original_feature_names}
        if retrieval_gate == "GATE_A_EXACT_MATCH":
            scores = np.asarray([float(x["rrf_score"]) for x in rr], dtype=np.float64)
            probs = scores / scores.sum()
            gaps = [abs(b_rank[doc] - d_rank[doc]) for doc in r_ids if doc in shared]
            disagreement = float(np.mean(gaps) / 99) if gaps else float("nan")
            frozen_top20 = [tokenized[doc] for doc in pool[:20]]
            lexical = [jaccard(left, right) for i, left in enumerate(frozen_top20) for right in frozen_top20[:i] if left | right]
            sorted_scores = np.sort(scores)
            score_gini = float(2 * np.dot(np.arange(1, 101), sorted_scores) / (100 * sorted_scores.sum()) - 101 / 100)
            query_tokens = TOKEN_RE.findall(queries[qid].lower())
            idfs = [bm25._index.idf[token] for token in query_tokens if token in bm25._index.idf]
            original_values = dict(zip(original_feature_names, (
                float(scores[0] - scores[1]),
                len(set(b_ids[:20]) & set(d_ids[:20])) / 20,
                rho,
                float(-(probs * np.log(probs)).sum()),
                score_gini,
                float(np.mean(lexical)) if lexical else None,
                float(scores[:10].sum() / scores.sum()),
                1 - disagreement,
                disagreement,
                len(query_tokens),
                sum(token.isnumeric() for token in query_tokens),
                float(np.mean(idfs)) if idfs else None,
                max(idfs) if idfs else None,
            )))
        direct.update({f"original13_{name}": original_values[name] for name in original_feature_names})
        original_rows.append({"query_id": qid, "retrieval_reconstruction_id": manifest["retrieval_run_id"], **original_values})
        output.append(direct)

    if len(output) != 300 or len({row["query_id"] for row in output}) != 300:
        raise ValueError("Feature extraction must produce exactly one row for each frozen TEST query")
    feature_frame = pd.DataFrame(output)
    forbidden_columns = [name for name in feature_frame.columns if any(word in name.lower() for word in FORBIDDEN)]
    if forbidden_columns:
        raise ValueError(f"Potential outcome leakage columns in feature matrix: {forbidden_columns}")
    feature_frame["query_id"] = [str(qid) for qid in query_ids]
    to_frame_new(feature_frame, OUT_FEATURES / "test_query_features.csv")
    to_frame_new(feature_frame, OUT_FEATURES / "test_query_features.parquet", parquet=True)
    to_frame_new(pd.DataFrame(original_rows), OUT_FEATURES / "original_13_feature_matrix_reconstructed.csv")

    summary_rows: list[dict[str, Any]] = []
    meta_columns = {"query_id", "feature_source", "component_feature_source", "retrieval_reconstruction_id", "rrf_alignment_gate", "dense_snapshot_hash", "corpus_hash", "feature_schema_version", "component_missing_rank_convention"}
    for column in feature_frame.columns:
        if column in meta_columns or not pd.api.types.is_numeric_dtype(feature_frame[column]):
            continue
        values = feature_frame[column].dropna().astype(float)
        if not len(values):
            continue
        summary_rows.append({"feature": column, "source": "frozen_pool_direct" if column.startswith("pool_") else ("reconstructed_component" if column.startswith(("component_", "bm25_", "dense_", "rrf_reconstructed_")) else "original_13_reconstructed"),
                             "n": len(values), "missing_n": len(feature_frame) - len(values), "mean": values.mean(), "std": values.std(ddof=0),
                             "min": values.min(), "p25": values.quantile(.25), "median": values.median(), "p75": values.quantile(.75), "max": values.max()})
    write_csv_new(OUT_FEATURES / "feature_distribution_summary.csv", summary_rows, list(summary_rows[0]) if summary_rows else ["feature", "source", "n", "missing_n", "mean", "std", "min", "p25", "median", "p75", "max"])
    feature_report = [
        "# TEST Feature Provenance", "",
        f"- Frozen TEST rows: {len(feature_frame)}; IDs come only from `locked_configurations.json`.",
        f"- Reconstruction run: `{manifest['retrieval_run_id']}`; RRF validation: `{retrieval_gate}`.",
        f"- Corpus SHA-256: `{manifest['corpus_sha256']}`.",
        f"- Dense snapshot manifest SHA-256: `{manifest['dense_snapshot']['manifest_sha256']}`; revision label: `{manifest['dense_snapshot']['label']}`.",
        "- Frozen-pool features use the saved Phase 2 `pool_ids` and current corpus texts. Lexical normalization is lowercase Unicode `\\w+` tokenization; similarity is token-set Jaccard. Top-20 and Top-100 pair means are unweighted averages over all unordered pairs. Empty documents remain in the denominator and have similarity zero.",
        "- Exact duplicate content ratio uses normalized token sets. Token-length and unique-token ratios are corpus-content descriptors. Frozen RRF positions are ordinal; no historical scores are inferred.",
        "- Reconstructed BM25, Dense, and RRF ranks and score distributions are kept in separately named fields. BM25, Dense, and RRF raw scores are never combined or compared in magnitude.",
        "- Sparse-Dense overlap uses the respective Top-k prefixes. Rank correlation and displacement use only shared Top-100 documents; absent documents have no imputed rank.",
        f"- Original Phase 2 feature formulas are in `src/evaluation/phase2_inputs.py` (SHA-256 `{source_feature_impl_hash}`), in the 13-feature order from `src/pooladapt/selector.py`.",
        f"- Original 13-feature formulas were {'recomputed from reconstructed retrieval outputs' if retrieval_gate == 'GATE_A_EXACT_MATCH' else 'not used as frozen-pool test features because reconstructed RRF did not reach Gate A'}; computed values remain reconstructed and do not recover historical component scores.",
        "- `qrels`, relevance labels, and frozen outcome columns were not read by this feature script.",
        "- `results/phase2/01_candidate_logging/candidate_features.csv` contains candidate-level rows, not the 13 query-level selector features. The original 13 formulas are implemented in `src/evaluation/phase2_inputs.py`; development `pool_features.csv` was read only to confirm the corresponding feature table exists and was not used as TEST input.",
    ]
    (OUT_FEATURES / "feature_provenance.md").parent.mkdir(parents=True, exist_ok=True)
    p = OUT_FEATURES / "feature_provenance.md"
    if p.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {p}")
    p.write_text("\n".join(feature_report) + "\n", encoding="utf-8")
    print(json.dumps({"feature_rows": len(feature_frame), "feature_columns": len(feature_frame.columns),
                      "rrf_gate": retrieval_gate, "original_13_reproduced": retrieval_gate == "GATE_A_EXACT_MATCH",
                      "output": str(OUT_FEATURES)}, indent=2))


if __name__ == "__main__":
    main()
