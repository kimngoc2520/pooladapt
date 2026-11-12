# Improvement 1 data handoff specification

## Frozen and validated population

The authoritative file is improvement/improvement_1/matched_budget/query_level_evaluation.csv. It contains 300 frozen SciFact TEST query IDs and 11,100 execution-level records derived from results/phase2/evaluation/final_gpu/test_predictions.jsonl, validated against test_query_metrics.csv and data/scifact/qrels/test.tsv. All 300 queries have query text and positive qrels. No query was excluded.

Validated methods and operating points: PoolAdapt B10/B20/B30/B50; Fixed Prefix B10/B20/B30/B50; SAGE-SLO B10/B20/B30/B50; PACE-EF B10/B20/B30/B50; Random B10/B20/B30/B50 at seeds 0–4; Full Rerank B100. Each method/operating-point/seed stratum contains the same 300 TEST IDs. Random has five genuine reranking executions per query and operating point. Preserve the seed field or aggregate seed measurements within query before cross-query statistical analysis.

## CSV schema

| Field | Definition / units |
|---|---|
| query_id | Frozen TEST query identifier (string). |
| query_text | SciFact query text. |
| method | Method label from execution record. |
| operating_point | B10, B20, B30, B50, or B100. |
| baseline_method | Normalized method key (pooladapt, fixed_prefix, sage_slo, pace_ef, random, full_rerank). |
| seed | Random seed; blank when not applicable. |
| selected_k | Number of selected candidate IDs; controller decision. |
| controller_features | Blank; TEST feature vectors are unavailable. |
| actual_reranked_pairs | Actual per-query pair count reported by execution record. |
| execution_id | Blank; no standalone execution ID exists. |
| execution_manifest_sha256 | Execution manifest digest attached to the prediction. |
| reranking_execution_count | 1 for each execution-level row. Random seeds are separate rows. |
| ndcg_at_10 | Repository linear-gain nDCG@10, recomputed from qrels. |
| recall_at_10 | Relevant retrieved top-10 documents / all qrels-relevant documents. |
| mrr_at_10 | Reciprocal rank of first relevant document in top 10, or 0. |
| reranking_latency_ms | Source latency seconds × 1000. |
| latency_measurement_scope | Pair preparation, inference, and sorting; exclusions noted. |
| execution_environment_reference | Path to final execution manifest and GPU/device. |
| query_group | Blank; no validated group labels were found. |

Blank values mean unavailable, not zero. Row uniqueness is (query_id, method, operating_point, seed). Do not deduplicate Random seeds or treat them as extra independent queries.

## Metric definitions and validation

The repository's src/evaluation/retrieval_metrics.py defines nDCG with linear relevance gain: DCG@10 = sum(relevance_i / log2(rank_i + 1)), divided by ideal DCG. Recall@10 is the number of relevant unique documents in the top 10 divided by the total relevant qrels documents. MRR@10 is reciprocal rank of the first relevant document within 10. Recomputed query metrics match all 11,100 saved rows within 1e-12. Aggregation reproduces all 21 final summary rows within 1e-10; Random is summarized across its five seed-level aggregates.

## Reranking cost policy

- Average K = mean selected_k over the common TEST population for a method and operating point; keep Random seed-specific unless explicitly averaging within query.
- Average Reranked Pairs/Query = mean actual_reranked_pairs from execution records over the same population. Never substitute selected K.
- Exclude candidates removed before reranking. Padding is not an actual query-document pair. Count genuine repeated inference in separate execution records. Avoid duplicate records using the row uniqueness key above.
- Current execution logs show selected count equals actual pairs on every row. They do not separately record candidate removals, padding, or retry causes.
- Full Rerank B100 is the common reference at 100 actual pairs/query.
- Compression Ratio = Average Reranked Pairs_Method / Average Reranked Pairs_Full. Compute from aggregate means on the same TEST population; do not average per-query ratios. Full Rerank ratio equals 1.
- Cost Reduction = 1 − Compression Ratio. Report as a fraction. Neither ratio measures latency or monetary/GPU cost.

Use these definitions and the helpers in src/evaluation/handoff_metrics.py for this handoff. The legacy Phase 1 helper has its own historical cost-field convention and is outside this definition contract.

## Latency, missingness, and known limitations

Per-query latency is available and converted to milliseconds. Scope includes measured pair preparation, Cross-Encoder inference, and sorting. Retrieval, selection, loading, and warm-up are excluded, so it is not end-to-end latency. Environment reference is results/phase2/evaluation/final_gpu/execution_manifest.json (Tesla T4, cuda:0).

No records were excluded. TEST controller features and query groups, explicit execution ID, timestamp, retries/padding accounting, and frozen output hashes are unavailable. Current source files match the current source manifest, but the run manifest's evaluation-script digest and its references to the amendment/source manifest do not match current post-run provenance files; exact provenance traceability remains a limitation. See the audit report.

## Required downstream paths

1. improvement/improvement_1/matched_budget/query_level_evaluation.csv — authoritative query-level observations.
2. improvement/improvement_1/audit/test_provenance_audit_report.md — audit findings and limitations.
3. improvement/improvement_1/audit/artifact_consistency_audit.csv — SHA-256 expectations, observations, and statuses.
4. improvement/improvement_1/audit/data_handoff_specification.md — this contract.
