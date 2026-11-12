# Phase 1/Phase 2 Retrieval Artifact Inventory

**Verdict: PARTIALLY_AVAILABLE**

## Summary

Phase 1 completed SciFact BM25 + Dense retrieval, RRF fusion to a 100-document pool, and reranking on 300 frozen TEST queries. Phase 2 extracted candidate/query features and developed PoolAdapt and SAGE-SLO on 809 train/validation queries. Phase 2's final TEST prediction log additionally preserves each TEST query's ordered 100 candidate IDs in `pool_ids`. It does not preserve the BM25 and Dense source runs or their raw scores/ranks for TEST, and the TEST feature matrix is absent.

The persisted candidate IDs are sufficient to recover the Phase 2 final-evaluation pool for the exact 300 TEST IDs. They do not establish, by independent comparison, that every position in those pools is identical to the historical Phase 1 pool. Phase 1 retains only reranked Top-10 document IDs, not its full pool. Full 13-feature TEST vectors therefore cannot be recovered from saved numeric artifacts alone.

## Phase 1 artifacts

| Artifact | Preserved content and population | Availability |
|---|---|---|
| `results/phase1/baseline_predictions.json` (SHA-256 `db3532c651a2f6264705db8d59a131f22f76a2073b81bb8e8e70ee4bf78b19a2`) | 9 methods × 300 TEST queries. Each record has `query_id`, reranked `document_ids` (10), pair count and latency. Metadata identifies BM25 + Dense + RRF and the dense model. | No BM25/Dense run scores/ranks, no RRF scores/ranks, and no full 100-ID pool. It is a reranking output. The Phase 1 Full Rerank Top-10 lists match Phase 2 Full Rerank's Top-10 for all 300/300 shared queries; this verifies only those ten returned IDs. |
| `results/phase1/candidate_pool_diagnostics.csv` (SHA-256 `8be04e3d9a02a6970b28f81ab4bbeb6cb66e8d7c474370d4db65fc63726e9f6c`) | 17 metric/value rows; aggregate diagnostics report 300 queries, pool size 100 and candidate recall/hit statistics. | No query IDs, document IDs, ranks or scores. |
| `results/phase1/failure_analysis.csv` (SHA-256 `944f18ac6ab0abc58c7634ed177e82fda0e8fbf11b1194d2d7bb0069a6077d2d`) | 768 failure/comparison rows over 221 distinct TEST query IDs. Contains full-rerank and method output document ID lists, each Top-10. | Partial query subset and reranked output lists only; not candidate-pool membership/ranks/scores. |
| `experiments/run_retrieval.py` | Source calls BM25 Top-100, Dense Top-100, then RRF Top-100. | Script logs only the first ten fused IDs to stdout; no runfile writer. Repository search found no separate `.trec`/`.run` BM25, Dense or fused runfiles in project results/data. |

## Phase 2 feature artifacts

The exact selector schema is `src/pooladapt/selector.py:FROZEN_FEATURES` (13 values). The model input is assembled by joining `pool_features.csv` and `query_features.csv` on `query_id` in `experiments/run_pooladapt.py` and `experiments/run_sage_slo.py`; it is not one combined CSV.

- `results/phase2/01_candidate_logging/pool_features.csv`: 809 unique query rows, nine pool features: `top1_top2_rrf_margin`, `top20_overlap`, `rank_correlation_union`, `score_entropy`, `score_gini`, `redundancy`, `concentration`, `sparse_dense_agreement`, `sparse_dense_disagreement`. SHA-256 `5f97dd8d7f9b50b93da4d03d72a15079a59c156d0591b29bb3a7f35e2f8f64c9`.
- `results/phase2/01_candidate_logging/query_features.csv`: 809 unique query rows; the four frozen values are `query_length`, `num_numeric_tokens`, `avg_idf`, `max_idf`. `entity_count` is an additional blank/unavailable column, not a model feature. SHA-256 `6e66b11572c4aa0de2a95d804e542370031fdf8d08c342ecf6de1b64dbd0cbac`.
- `results/phase2/01_candidate_logging/candidate_features.csv`: 80,900 rows (100 candidates × 809 queries), with query/document IDs and BM25/Dense ranks and scores, RRF rank/score, local RRF margin and rank displacement. This is candidate-level supporting data, not the 13-value per-query model matrix. SHA-256 `bab7bc8bafb5c60857cde13ee096ef76537694259f2f0eff14a93aa8e9404b0f`.
- `results/phase2/01_candidate_logging/metadata.json`: records 809 queries, 80,900 candidate rows, pool size 100, corpus/query SHA-256 values, model identifier and feature-generation descriptions. The query list is `results/phase2/01_candidate_logging/train_validation_query_ids.csv`; split metadata identifies 647 train + 162 validation queries, with no TEST overlap. The feature extractor explicitly does not read qrels, so these features are pre-reranking and outcome-independent. Metadata SHA-256: `d3c6b1219b315312b2d69e13fc1d13464e9066c60db34c65f8eaf8ae3cecd370`.

Against the authoritative frozen IDs in `improvement/improvement_1/matched_budget/query_level_evaluation.csv` (300 distinct IDs; SHA-256 `ff6be565f2437ef10642ccc93394bae83f2d82547c0065fac999eed1b2bc0609`), each of the three feature files and the train/validation ID list has **809 distinct IDs, 0 overlap, 300 missing TEST IDs, and 809 non-TEST IDs**. No TEST feature vectors were found in the feature tables or Improvement 1 handoff. The handoff's `controller_features` values are blank.

## TEST candidate IDs preserved in Phase 2 evaluation

`results/phase2/evaluation/final_gpu/test_predictions.jsonl` (SHA-256 `62943e2459a7e6b02a7c63338aaf6157c191145fb3875c8b811b8136bb2ed087`) contains 11,100 execution records across methods/seeds and 300 distinct TEST query IDs. Every record has `pool_ids`: exactly 100 unique document IDs. The ordered list is identical across all records for each query (300 unique per-query pool orders). Comparing its query IDs to the authoritative Improvement 1 handoff gives **300 overlap, 0 missing, 0 unexpected**. The handoff points to `results/phase2/evaluation/final_gpu/execution_manifest.json` with manifest SHA `4f205c56115b8bda21a0ca30869bafe47500f35a126919cfa3a6c6ad38d820c7`; the manifest records the corpus, qrels, retrieval/evaluation source and configuration hashes.

In `experiments/run_phase2_evaluation.py`, the pool is built by calling BM25 and Dense with Top-100 and `fuse_ranked_lists(..., top_n=100)`, and each prediction row serializes the ordered IDs from that pool. Its resume validation compares saved `pool_ids` with the generated pool. This establishes an exact saved Phase 2 evaluation candidate list/order for each frozen query, but there is no independent full Phase 1 pool runfile against which to verify all 100 positions.

## Retrieval configuration and limits

- **BM25:** `src/retrieval/bm25.py`, `BM25Okapi`; documents are `title + " " + text`, tokenized by lowercase plus `\w+`; queries use the same tokenizer. No BM25 parameters are passed, so the implementation relies on `rank_bm25` defaults (execution manifest records version 0.2.2). Ties sort by document ID.
- **Dense:** `src/retrieval/dense.py`; `sentence-transformers/all-MiniLM-L6-v2`; document input is title + space + text, query input is the query string; embeddings are normalized and ranked by dot product/cosine similarity, ties by document ID. The model name is recorded, but no immutable model revision/hash is recorded in the inspected metadata.
- **Corpus and queries:** Feature metadata and the Phase 2 execution manifest record corpus SHA-256 `dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6` and queries SHA-256 `8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313`. Retrieval is built in memory; no separate persisted BM25 index or dense index was found.
- **RRF:** `src/retrieval/rrf.py`; reciprocal-rank fusion with `k=60`, source ranks from Top-100 lists, score contributions `1/(60+rank)`, descending RRF score then document-ID tie-break, truncated to Top-100. No score normalization is applied.
- **Search result:** `results/phase2/01_candidate_logging/candidate_features.csv` contains full component ranks/scores only for the 809 development queries. TEST `pool_ids` preserve only fused candidate IDs/order; raw TEST BM25/Dense lists/scores, RRF numeric scores, and the 13 TEST feature values are absent. `improvement/improvement_1/matched_budget/query_level_evaluation.csv` stores outcomes, latency and empty controller-feature/group fields, not retrieval features.

## Assessment

**PARTIALLY_AVAILABLE.** The original Phase 1 BM25/Dense runfiles and full Phase 1 fused rankings are not preserved as separate artifacts. However, the final Phase 2 TEST records preserve exact ordered 100-ID candidate pools for precisely the 300 frozen Improvement 1 queries, with no ID anomalies, and the returned Top-10 agrees with Phase 1 for all queries. This supports candidate-ID-level inspection of those saved Phase 2 pools. It does not recover the raw BM25/Dense scores/ranks or the numeric 13-feature TEST matrix. Reconstructing those numeric features would require a new retrieval/feature computation or another missing artifact; neither was run or generated in this read-only inventory. Historical Phase 1 equivalence of all 100 pool positions remains unverified.

