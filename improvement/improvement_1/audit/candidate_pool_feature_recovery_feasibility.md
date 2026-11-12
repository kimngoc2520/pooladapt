# Candidate-Pool Feature Recovery Feasibility — Frozen TEST

**Recommendation: PARTIAL ANALYSIS IS FEASIBLE.** The Phase 2 TEST log preserves each frozen TEST query's ordered RRF pool IDs, but not the component BM25/Dense rankings or scores. This supports descriptive analysis of the saved fused pools and corpus-based content characteristics; it does not support recovery of the complete 13-feature matrix or validation that the saved Phase 2 pools equal the historical Phase 1 pools.

## Artifacts and schemas inspected

### Phase 2 TEST candidate pools

`results/phase2/evaluation/final_gpu/test_predictions.jsonl` is JSON Lines. Its 11,100 records share the schema:

`query_id, method, target_budget, seed, reranked_pairs, reranking_latency, document_ids, selected_ids, pool_ids, frontloaded_ids, execution_manifest_sha256`.

`pool_ids` is an array of 100 document IDs in the RRF-fused candidate order. The array index encodes the 1-based RRF rank; each ID has no separate rank, score, retriever-source, or fusion-score field. It is neither a BM25-only list nor a Dense-only list. `selected_ids` is method/budget-specific, `frontloaded_ids` is a later policy ordering, and `document_ids` is the Cross-Encoder output order; none should be interpreted as the original pool order.

The source in `experiments/run_phase2_evaluation.py` builds the pool as `fuse_ranked_lists({'bm25': ..., 'dense': ...}, top_n=100)` and serializes `[c['id'] for c in pool]` to `pool_ids`. Therefore these arrays represent the final hybrid/RRF pool in retrieval order. The file does not retain the source lists or their numeric scores.

Integrity/population checks: 300 distinct query IDs; all 300 match the authoritative frozen IDs in `improvement/improvement_1/matched_budget/query_level_evaluation.csv` (0 missing, 0 unexpected). All 11,100 rows contain 100 unique IDs; the per-query ordered list is consistent across its 37 method/seed records, giving 300 distinct ordered pools. All pool IDs resolve in the current SciFact corpus (5,094 distinct document IDs across the 300 pools). The frozen handoff points to `results/phase2/evaluation/final_gpu/execution_manifest.json`; the prediction file SHA-256 recorded in the prior inventory is `62943e2459a7e6b02a7c63338aaf6157c191145fb3875c8b811b8136bb2ed087`.

### Historical Phase 1 and development feature artifacts

- `results/phase1/baseline_predictions.json`: 300 query records per method, including Full Rerank. Each record contains only the post-reranking Top-10 `document_ids`, not a candidate pool. Its metadata says “BM25 + Dense + RRF” and names the Dense model, but has no component ranks/scores or pool IDs.
- `results/phase1/candidate_pool_diagnostics.csv`: aggregate metric/value rows only; no query or document IDs.
- `results/phase1/failure_analysis.csv`: selected failure rows with post-reranking Top-10 IDs; not candidate lists.
- `results/phase2/01_candidate_logging/candidate_features.csv`: candidate-level rows for 809 train/validation queries, with `query_id, doc_id, bm25_rank, bm25_score, dense_rank, dense_score, rrf_rank, rrf_score, local_rrf_margin, rank_displacement`. These are not TEST data.
- `results/phase2/01_candidate_logging/pool_features.csv` and `query_features.csv`: the nine pool and four query features for those same 809 development queries. The IDs are disjoint from frozen TEST (0 overlap); no TEST feature rows are present.

Thus no separately preserved Phase 1 BM25 run, Dense run, or full RRF candidate ranking is available for a per-query historical comparison. Phase 1 Full Rerank Top-10 outputs agree with Phase 2 Full Rerank Top-10 outputs, but these are Cross-Encoder result rankings and cannot establish candidate-pool equality or RRF rank agreement.

## Feature availability

“Directly computable” means the required fields are already present in the saved TEST pool/query/corpus inputs; the feature values themselves are not claimed to have been generated. “Computable with limitations” requires a reconstruction step beyond the preserved TEST artifacts. No scores are inferred from rank positions.

| Feature | Status | Required inputs and available evidence |
|---|---|---|
| Candidate pool size | **DIRECTLY_COMPUTABLE** | `len(pool_ids)` is 100 for every TEST query. |
| RRF ordinal rank profile / rank-position summaries | **DIRECTLY_COMPUTABLE** | `pool_ids` order gives RRF positions 1–100. Positional counts or ID-level rank summaries can be formed. These are rank-only summaries, not score concentration. |
| RRF score concentration (`sum(top 10 RRF scores) / sum(all pool RRF scores)`) | **NOT_COMPUTABLE** | Requires per-document RRF scores; none are in the TEST log. Rank positions do not recover the score mass. |
| Sparse-Dense Top-20 overlap (`top20_overlap`) | **NOT_COMPUTABLE** | Requires the separate BM25 and Dense Top-20 ID sets. Only their fused Top-100 output is saved. |
| Sparse-Dense disagreement (`sparse_dense_disagreement`) | **NOT_COMPUTABLE** | Requires BM25 and Dense ranks for pool documents to calculate normalized rank displacement. Component ranks are absent. |
| Sparse-Dense rank correlation (`rank_correlation_union`) | **NOT_COMPUTABLE** | Requires paired source ranks for documents appearing in both component lists. Neither list/rank column exists for TEST. |
| Top-1/Top-2 RRF score margin (`top1_top2_rrf_margin`) | **NOT_COMPUTABLE** | Requires the top two numeric RRF scores, not just their order. |
| RRF score entropy (`score_entropy`) | **NOT_COMPUTABLE** | Requires the complete numeric RRF score distribution for the 100 candidates. |
| RRF score Gini (`score_gini`) | **NOT_COMPUTABLE** | Requires numeric RRF scores for the 100 candidates. |
| Top-20 content redundancy (`redundancy`) | **DIRECTLY_COMPUTABLE** | Requires the first 20 `pool_ids` and their title/text in the corpus. Both are available; corpus SHA-256 is `dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6`, matching the Phase 2 feature metadata. The Phase 2 definition is mean pairwise token-set Jaccard using the project's tokenizer. Values have not been generated here. |
| Query length (`query_length`) | **DIRECTLY_COMPUTABLE** | Requires frozen query text and the defined lowercase `\w+` tokenizer; query text is available in the SciFact query file. |
| Numeric-token count (`num_numeric_tokens`) | **DIRECTLY_COMPUTABLE** | Requires the tokenized frozen query; available query text and tokenizer suffice. |
| Mean/max BM25 IDF (`avg_idf`, `max_idf`) | **COMPUTABLE_WITH_LIMITATIONS** | Requires query tokens and the corpus-derived BM25 IDF table. Query/corpus and implementation are available, but no TEST IDF values/index are stored. Rebuilding the index with the recorded implementation and library defaults would be a new computation and was not done. |
| `entity_count` (extra CSV column) | **NOT_COMPUTABLE** | It is blank and marked unavailable in Phase 2 metadata; it is not one of the frozen 13 model inputs and no extraction method is specified. |

The 13 frozen model inputs are the nine `pool_features.csv` fields plus the four named query features in `src/pooladapt/selector.py:FROZEN_FEATURES`. Candidate-level rank/score fields in `candidate_features.csv` are supporting data, not additional entries in that 13-feature schema.

## Historical Phase 1 correspondence

**Classification: NO_COMPARABLE_HISTORICAL_ARTIFACT.** No Phase 1 per-query 100-document candidate list exists to compare against the Phase 2 `pool_ids`. Therefore candidate ID set equality, ordered equality, common-document counts, Top-10/Top-20 pool overlap, and rank agreement are **not measurable for any of the 300 queries** from the preserved Phase 1 artifacts. The Phase 1 Top-10 reranked outputs are not a valid substitute: reranking changes order and truncates the list.

Available configuration evidence establishes how the saved Phase 2 pool was produced, but not exact Phase 1 equivalence:

- **BM25:** `src/retrieval/bm25.py` uses `rank_bm25.BM25Okapi`; title and text are joined with a space and tokenized by lowercase plus `\w+`; queries use the same tokenizer. No parameters are passed in code (library defaults); the Phase 2 execution environment records `rank-bm25` 0.2.2. Results sort descending by score, then document ID.
- **Dense:** `src/retrieval/dense.py` uses `sentence-transformers/all-MiniLM-L6-v2`, encodes title-plus-text documents and raw query strings, normalizes embeddings, and ranks by cosine-equivalent dot product with document-ID tie-breaks. An immutable model revision/hash is not recorded.
- **Corpus/index:** Phase 2 metadata and execution manifest record corpus SHA-256 `dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6` and query SHA-256 `8ff84a7c903f722981cd8d595c022660140c51867b27608a6d4910db86080313`. Retrieval indexes are created in memory; no persisted BM25/Dense index was found. Phase 1 baseline metadata does not provide a comparable input hash manifest.
- **Top-K and fusion:** the Phase 2 evaluation calls each retriever with Top-100, then `src/retrieval/rrf.py` applies RRF with `k=60`, score contributions `1/(60+rank)`, score-descending/document-ID tie-break, and final Top-100. The Phase 1 prediction metadata names the hybrid method but does not itself record these parameters or a full run.

These settings are consistent in the inspected Phase 2 code/metadata with the general Phase 1 description, but missing Phase 1 full candidate rankings, pinned Dense revision, and Phase 1 input/config manifest prevent a claim that the 300 Phase 2 pools exactly match the historical Phase 1 pools.

## Feasibility for TEST analysis

- **Full candidate-set characteristic analysis:** not supported from current saved TEST values. The nine pool features cannot all be recovered because source ranks and numeric RRF scores are missing; the complete frozen 13-feature vectors are not present.
- **Partial rank/content analysis:** supported for the saved Phase 2 fused pools. Valid descriptive questions include pool membership and size, RRF ordinal position patterns, which corpus documents occupy the leading positions, and content redundancy of the RRF Top-20. Query length/token count can also be summarized from frozen query text. These results must be described as analysis of the saved Phase 2 pools; historical Phase 1 pool equivalence is unverified.
- **Query-group comparison:** not currently supported as a labeled TEST analysis. The Improvement 1 `query_group`/controller feature handoff fields are blank, and this investigation creates no labels. Candidate descriptions could be reported by query ID or alongside independently established groups if such labels are later supplied; qrels or evaluation/reranking outcomes must not be used to create them.

**Recommendation:** Use the existing pool IDs only for a clearly scoped partial, rank/content-based descriptive analysis. If the research question requires the frozen Phase 1 BM25-versus-Dense diagnostics or all 13 features on the same 300 TEST queries, supplementary retrieval/feature reconstruction is necessary. Before claiming historical equivalence, obtain or recreate a provenance-backed Phase 1 reference and compare all 300 complete ordered pool lists; this pass did not perform such reconstruction.
