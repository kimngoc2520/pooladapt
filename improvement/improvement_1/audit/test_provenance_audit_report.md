# TEST Provenance Audit Report — Improvement 1

## Scope and inherited audit work

This audit reuses the existing Phase 2 final-evaluation records and prior reviews: docs/phase2_final_evaluation.md, docs/phase2_dependency_audit.md, results/phase2/evaluation/locked_configurations.json, reviewed_source_manifest.json, provenance_amendment.json, inspection_manifest.json, results/phase2/evaluation/final_gpu/phase2_summary.json, execution_manifest.json, run.log, test_predictions.jsonl, test_query_metrics.csv, and experiments/audit_phase2_final.py / preflight_phase2_test.py. Frozen TEST outputs and inputs were not modified; the experiment was not rerun.

The current reviewed source manifest has 64 entries, all of which match the checked out source tree using LF-normalized SHA-256. The execution manifest pins an earlier hash for experiments/run_phase2_evaluation.py; this is the one source file among its execution inputs that does not match. The current amendment and reviewed-source-manifest files also do not match the historical SHA-256 values embedded in the execution manifest. Project documentation attributes those script and metadata differences to the post-run path migration. These mismatches are recorded as FAIL, even though the changes are documented as post-run provenance work and the current source tree matches its own manifest.

## Audit item statuses

| Audit ID | Description | Status | Verification method and evidence | Outstanding issue |
|---|---|---|---|---|
| LOCK-01 | Frozen lock/configuration hash | PASS | Summary locked-configuration SHA-256 matches the preserved lock file. | None. |
| DATA-01 | TEST population, query text and qrels | PASS | 300 frozen SciFact IDs; all query texts and positive qrels exist; qrels digest matches execution input hash. | None. |
| SRC-01 | Reviewed source manifest internal integrity | PASS | All 64 source entries match canonical source bytes and recorded Git blob IDs. | Does not resolve historical execution-manifest digest mismatch below. |
| SRC-02 | TEST execution source identity | FAIL | Execution input hash for experiments/run_phase2_evaluation.py differs from current file hash. | Exact source used in TEST is not identical to current file. |
| PROV-01 | Execution-to-amendment/manifest reference | FAIL | Current amendment and reviewed-source-manifest SHA-256 differ from digests embedded in execution manifest. | Historical referenced versions are not present at current paths. |
| EXEC-01 | Execution completion and prediction linkage | PASS | Summary status COMPLETE; all 11,100 rows carry execution-manifest SHA; run log reports progress through 300/300. | No standalone run ID/time in manifest. |
| PRED-01 | Method/query coverage and duplicates | PASS | 37 strata, 300 unique TEST queries per stratum; no missing, duplicate, or unexpected query IDs. | None. |
| PRED-02 | Ranking shape and actual pair counts | PASS | Top-10 ranking present; selected IDs unique; selected length equals recorded reranked_pairs in all rows. | Padding/removal and hidden retries have no separate trace. |
| QREL-01 | Qrels validity | PASS | 339 judgments; every TEST query has positive qrels; no extra qrel query IDs. | None. |
| MET-01 | Query metric reproduction | PASS | Recomputed nDCG@10, Recall@10, MRR@10 match all 11,100 saved rows within 1e-12. | None. |
| MET-02 | Aggregate metric consistency | PASS | Recomputed aggregates match all 21 summary rows within 1e-10; Random aggregated over five seed-level runs. | None. |
| CKPT-01 | Checkpoint/configuration references | PASS | PoolAdapt and SAGE-SLO checkpoint hashes match metadata; sidecars record successful TRAIN-only reload verification; locked model and reranker configurations are present. | Execution does not separately log loaded checkpoint bytes. |
| COST-01 | Actual reranking-pair data | PASS | Actual pair counts are present for each execution; Full Rerank mean is 100. | Pair counts happen to equal selected K; definitions remain separate. |
| LAT-01 | Reranking latency data | PASS | Per-query latency is present; execution manifest defines seconds and pair-preparation/inference/sorting scope. | Retrieval, selection, loading and warm-up are excluded. |
| FEATURE-01 | TEST controller features and query groups | NOT VERIFIED | Prediction records contain no per-query feature vectors or group labels. | Fields left blank in handoff. |
| RUN-01 | Explicit run ID and timestamp | NOT VERIFIED | Execution manifest has no standalone execution ID or wall-clock timestamp. | Manifest SHA is the run reference. |
| HASH-01 | Frozen output digest for predictions/metrics | NOT VERIFIED | Outputs have internal linkage and consistency checks, but no expected output digest is recorded. | Do not claim independently frozen output hashes. |

Audit item counts: 12 PASS, 2 FAIL, 3 NOT VERIFIED. The artifact hash CSV has 86 PASS, 3 FAIL, and 2 NOT VERIFIED rows.

**Overall:** query-level TEST data and metrics are internally consistent, but exact provenance is not fully verified because the current evaluation script and provenance metadata do not match the values pinned by the execution manifest. Do not characterize the complete TEST evaluation as reproducible without resolving that historical reference chain.

## Population, predictions and qrels

- Frozen TEST population: 300 queries from locked configurations and the final summary.
- Validated query population: 300; no exclusions.
- Qrels: 339 judgments; all TEST query IDs are represented with positive judgments.
- Predictions: 11,100 execution records across 37 method/operating-point/seed strata; each stratum covers the same 300 IDs.
- Complete methods: PoolAdapt, Fixed Prefix, SAGE-SLO, PACE-EF, Random, and Full Rerank. Required PoolAdapt B10/B20/B30/B50, Fixed Prefix B10/B20/B50, and Full Rerank B100 are present. Fixed Prefix B30 also exists.
- Missing, duplicate, and unexpected query IDs: none.
- Query metric mismatches: none (0 / 11,100).
- Aggregate mismatches: none (0 / 21).

## Frozen configuration and execution findings

The lock is results/phase2/evaluation/locked_configurations.json. It specifies SciFact, 300 TEST queries, candidate pool size 100, budget grid 10/20/30/50/100, random seeds 0–4, and Cross-Encoder cross-encoder/ms-marco-MiniLM-L-6-v2. TEST was not used for calibration. Execution metadata records Tesla T4, cuda:0, CUDA 13.0, and Python 3.11.15. The execution-manifest digest is 4f205c56115b8bda21a0ca30869bafe47500f35a126919cfa3a6c6ad38d820c7; each prediction record contains this digest.

The run log records successful progress through TEST 300/300. It also prints a historical journal path under results/phase2/07_evaluation_gpu/, while completed outputs are in results/phase2/evaluation/final_gpu/. The path inconsistency is recorded; artifacts were not moved or changed.

## Cost normalization

Average K = mean selected_k per method/operating point (and seed for Random). Average Reranked Pairs/Query = mean actual reranked_pairs recorded per execution; never infer it from selected K. These fields agree on every current record, but remain distinct. Candidate removals, padding, and retry logs are not recorded separately. Random's five seeds are genuine repeated executions and are preserved as seed strata.

Full Rerank B100 reference = 100 actual pairs/query. Compression Ratio = method average actual pairs / Full Rerank average actual pairs. Cost Reduction = 1 − Compression Ratio. Both are pair-consumption ratios, not latency or monetary/GPU-cost measures.

| Method | Operating point | Seed(s) | Execution rows | Average K | Average actual pairs/query | Compression ratio | Cost reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| PoolAdapt | B10 | — | 300 | 10.267 | 10.267 | 0.102667 | 0.897333 |
| PoolAdapt | B20 | — | 300 | 20.833 | 20.833 | 0.208333 | 0.791667 |
| PoolAdapt | B30 | — | 300 | 32.867 | 32.867 | 0.328667 | 0.671333 |
| PoolAdapt | B50 | — | 300 | 49.333 | 49.333 | 0.493333 | 0.506667 |
| Fixed Prefix | B10 | — | 300 | 10.000 | 10.000 | 0.100000 | 0.900000 |
| Fixed Prefix | B20 | — | 300 | 20.000 | 20.000 | 0.200000 | 0.800000 |
| Fixed Prefix | B30 | — | 300 | 30.000 | 30.000 | 0.300000 | 0.700000 |
| Fixed Prefix | B50 | — | 300 | 50.000 | 50.000 | 0.500000 | 0.500000 |
| SAGE-SLO | B10 | — | 300 | 10.267 | 10.267 | 0.102667 | 0.897333 |
| SAGE-SLO | B20 | — | 300 | 25.533 | 25.533 | 0.255333 | 0.744667 |
| SAGE-SLO | B30 | — | 300 | 34.267 | 34.267 | 0.342667 | 0.657333 |
| SAGE-SLO | B50 | — | 300 | 54.800 | 54.800 | 0.548000 | 0.452000 |
| PACE-EF | B10 | — | 300 | 10.000 | 10.000 | 0.100000 | 0.900000 |
| PACE-EF | B20 | — | 300 | 20.000 | 20.000 | 0.200000 | 0.800000 |
| PACE-EF | B30 | — | 300 | 30.000 | 30.000 | 0.300000 | 0.700000 |
| PACE-EF | B50 | — | 300 | 50.000 | 50.000 | 0.500000 | 0.500000 |
| Random | B10 | 0–4 | 1,500 | 10.000 | 10.000 | 0.100000 | 0.900000 |
| Random | B20 | 0–4 | 1,500 | 20.000 | 20.000 | 0.200000 | 0.800000 |
| Random | B30 | 0–4 | 1,500 | 30.000 | 30.000 | 0.300000 | 0.700000 |
| Random | B50 | 0–4 | 1,500 | 50.000 | 50.000 | 0.500000 | 0.500000 |
| Full Rerank | B100 | — | 300 | 100.000 | 100.000 | 1.000000 | 0.000000 |

Latency source values are seconds and the handoff converts them to milliseconds. Scope is measured pair preparation, Cross-Encoder inference and sorting; retrieval, selection, loading and warm-up are excluded. This is reranking-only latency, not end-to-end latency.

## Artifact hash audit and limitations

See artifact_consistency_audit.csv for paths, expected and observed digests, canonicalization method and status. Text source files are normalized to LF for source-manifest verification; non-source execution inputs use raw-file SHA-256. The three failing rows are the evaluation-script hash and the two historical provenance-reference files. The two output files lack frozen expected digests.

Per-query controller features, query groups, explicit execution ID, and timestamp are unavailable. Exact current source-to-run identity is the principal limitation; consult data_handoff_specification.md before interpreting downstream results.

## Tests and deliverables

Added 14 targeted unit tests in tests/test_improvement_1_handoff.py and reusable validation/cost helpers in src/evaluation/handoff_metrics.py. Ran the new tests together with tests/test_cost_metrics.py and tests/test_latency_metrics.py: **43 passed**. The TEST pipeline and predictions were not rerun.

Created:

- improvement/improvement_1/audit/test_provenance_audit_report.md
- improvement/improvement_1/audit/artifact_consistency_audit.csv
- improvement/improvement_1/audit/data_handoff_specification.md
- improvement/improvement_1/matched_budget/query_level_evaluation.csv
