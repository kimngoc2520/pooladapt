# Improvement 1 Final Completion Audit

Audit date: 2026-11-12 (Asia/Saigon)

## Final status

**VERIFIED WITH LIMITATIONS.** Random latency percentiles are standardized and validated against the frozen query-level handoff. Query-group analysis remains **UNAVAILABLE** for retrieval ambiguity because validated per-query TEST retrieval features were not saved. The updated report was generated and visually checked, but the existing DOCX could not be replaced because an Office-style `~$` lock file is present and the write failed with `PermissionError`. The ready-to-review candidate is `PoolAdapt_CT1_Evaluation_Report_updated.docx`; the canonical PDF was regenerated from that candidate. No application process was closed.

The upstream provenance status remains 12 PASS, 2 FAIL, and 3 NOT VERIFIED. This work does not establish complete historical reproducibility.

## 1. Query-group analysis

**Verdict: UNAVAILABLE.** No labels or group comparisons were created.

- The authoritative TEST population remains the 300 unique IDs in `matched_budget/query_level_evaluation.csv` (11,100 execution records). Its `controller_features` and `query_group` fields are blank.
- The handoff contains frozen query text, but no per-query TEST BM25/dense scores or ranks, RRF scores, or saved pre-reranking retrieval feature vectors. `results/phase2/evaluation/final_gpu/test_predictions.jsonl` contains rankings and execution fields without those features.
- Phase 2 `query_features.csv`, `pool_features.csv`, and `candidate_features.csv` cover 809 train/validation IDs and have zero overlap with the 300 TEST IDs. The Phase 2 M* Easy/Hard labels are qrels/reranking-outcome-derived and cannot be transferred to TEST.
- Query length can be tokenized from frozen query text without looking at outcomes, but it measures text length rather than retrieval agreement or ranking decisiveness. It does not support the requested retrieval-ambiguity interpretation, so no median split was forced.
- No qrels, retrieval metrics, reranking outcomes, or PoolAdapt outcomes were used to assign labels. No empty placeholder table or query-group figure was made.

Evidence and source paths: `audit/phase2_easy_hard_reuse_assessment.md`, `audit/data_handoff_specification.md`, `matched_budget/query_level_evaluation.csv`, `results/phase2/evaluation/final_gpu/test_predictions.jsonl`, `results/phase2/01_candidate_logging/{query_features,pool_features,candidate_features}.csv`, and `results/phase2/02_oracle/mstar_labels_tau099.csv`.

## 2. Random latency percentile standardization

Raw observations were read from `matched_budget/query_level_evaluation.csv`. There are 6,000 Random rows: 300 unique TEST queries × seeds 0-4 × four budgets (B10/B20/B30/B50). Every query has exactly one finite, nonnegative latency per seed and budget. There are zero missing or duplicate query-seed observations. All 11,100 handoff rows use the same scope: pair preparation, inference, and sorting; retrieval, selection, model load, and warm-up are excluded. Non-Random methods have one observation per query and operating point.

Primary convention: for each query and budget, average its five Random seed latencies, then calculate P50/P95 across 300 query means. The independent query population is 300. The pooled 1,500 query-seed observations are a sensitivity distribution combining query and seed variation; they are not treated as 1,500 independent queries.

| Budget | Primary P50 / P95 (ms) | Per-seed P50 median [range] (ms) | Per-seed P95 median [range] (ms) | Pooled query-seed P50 / P95 (ms) | Historical mean-seed P50 / P95 (ms) |
|---|---:|---:|---:|---:|---:|
| B10 | 56.7452 / 59.6465 | 57.7249 [57.6300, 58.1720] | 60.6558 [60.2878, 60.7400] | 57.8374 / 60.5365 | 57.8169 / 60.5484 |
| B20 | 116.2014 / 118.9335 | 116.6969 [115.8931, 116.9563] | 119.4505 [119.4246, 119.9835] | 116.6184 / 119.5838 | 116.5799 / 119.5742 |
| B30 | 177.3916 / 181.5635 | 178.1743 [175.6571, 178.3744] | 182.8733 [181.6503, 186.4401] | 177.9165 / 183.8240 | 177.6468 / 183.6907 |
| B50 | 252.3046 / 284.9789 | 252.5789 [250.5857, 253.2787] | 285.1919 [283.7482, 289.1274] | 252.1483 / 285.4466 | 252.2593 / 285.6550 |

Per-seed values are in `latency_cost/random_latency_seed_percentiles.csv`; all aggregation conventions and historical comparisons are in `latency_cost/random_latency_percentile_sensitivity.csv`. `latency_cost/latency_cost_summary.csv` and `tables/main_results_summary.csv` use the primary query-level mean convention. The main report distinguishes these estimates and explains why the historical mean of per-seed percentiles differs. This resolves the prior reporting ambiguity; the old source values remain documented and are not used as validation targets for the different primary estimand.

## 3. Updated and preserved artifacts

Updated or created under `improvement/improvement_1`:

- `scripts/standardize_random_latency.py` — reproducible standard-library aggregation with complete-coverage and duplicate checks.
- `scripts/generate_report.py` and `scripts/run_all.py` — report and validation text updated; `run_all.py` was not run.
- `latency_cost/random_latency_percentile_sensitivity.csv` and `latency_cost/random_latency_seed_percentiles.csv` — primary and sensitivity results.
- `latency_cost/latency_cost_summary.csv` and `tables/main_results_summary.csv` — Random P50/P95 standardized to the primary convention.
- `audit/downstream_validation_report.md` and `audit/downstream_consistency_audit.md` — convention and query-group evidence clarified.
- `PoolAdapt_CT1_Evaluation_Report_updated.docx` — revised report candidate generated because the original path is locked. SHA-256: `46b2caf7163ee1b8968651b8581cf6117218b5d952cc551acffaf6383a984ed5`.
- `PoolAdapt_CT1_Evaluation_Report.pdf` — regenerated from the revised candidate; SHA-256: `e0a8e74d7a38015325901525c5fa534ed6588d2ee1c41cfc6c04e41738599c2a`. This is the only PDF in the Improvement 1 tree.

The original `PoolAdapt_CT1_Evaluation_Report.docx` remains unchanged (271,381 bytes) until its lock is released; attempts to write or hash it failed because another process has it open. Other existing Improvement 1 deliverables, scripts, figures, tables, and frozen source records were preserved. No files were deleted.

## 4. Validation

- Latency standardization script: PASS. It verified 11,100 handoff rows, 300 unique query IDs, 6,000 Random records, seeds 0-4, 300 observations per seed and operating point, zero missing/duplicate records, and one shared latency scope.
- Cross-artifact percentile check: PASS. All four Random primary P50/P95 pairs match in the sensitivity CSV, latency-cost summary, and main results table.
- Improvement 1 handoff unit tests: PASS, 14 tests (`tests.test_improvement_1_handoff`).
- Downstream pytest suite: NOT RUN. The available Windows Python launcher cannot start its Windows Store Python runtime; the offline `uv` attempt was blocked because the default cache path was denied. The direct standard-library latency validation ran successfully. No dependency was installed.
- DOCX candidate: ZIP integrity PASS; 569 paragraphs, 6 tables, 9 embedded images; revised query-group and latency text present.
- PDF: LibreOffice Writer 26.8.0.3 export; 13 letter-size pages; unencrypted. Every rendered page was visually inspected. Text, tables, and figures are legible with no clipping, overlap, blank pages, or broken tables.
- The canonical PDF was regenerated only after the report candidate was produced. Its content includes the UNAVAILABLE query-group verdict and primary Random values.

## 5. Frozen-input integrity and scope

The handoff SHA-256 remains `ff6be565f2437ef10642ccc93394bae83f2d82547c0065fac999eed1b2bc0609`, matching the current observed hash in `audit/downstream_input_manifest.json`. Current hashes checked after this work: TEST predictions `62943e2459a7e6b02a7c63338aaf6157c191145fb3875c8b811b8136bb2ed087`; TEST metrics `b49e1c84cb22d2feddbc66bd43fa63dcc528affa5f7a06d608aff67f2ce711e`; TEST qrels `0864bb985e0ca2367ba217977e72004d549054b2b06666ed9d4825ac7c21284c`. Phase 2 feature and M* label files still match their existing inspection-manifest hashes. These current hashes record inspected bytes and do not resolve the previously documented missing historical frozen-output hashes.

No retrieval, reranking, model inference, training, experiment, or `run_all.py` pipeline was run. No frozen inputs or qrels were modified. Changes were limited to Improvement 1 analysis/report/audit artifacts; no unrelated files were changed or deleted.

## Remaining limitation

To put the revised report at the original DOCX path, close the document that owns `PoolAdapt_CT1_Evaluation_Report.docx` and release its `~$` lock file. The revised candidate and matching canonical PDF are ready; the original open document was not forcibly closed or overwritten.
