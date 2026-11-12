# Downstream handoff validation report

Status: **PASS**
Rows: 11100; unique TEST queries: 300; unique composite keys: 11100; strata: 37.
Methods: Fixed Prefix, Full Rerank, PACE-EF, PoolAdapt, Random, SAGE-SLO.
Operating points: B10, B100, B20, B30, B50.
Random seeds: 0, 1, 2, 3, 4; cross-query aggregation averages seeds within query.
Positive qrels: 339 judgments across 300 query IDs; TEST IDs missing positive judgments: 0.
Execution-manifest linkage values: 1 unique hash(es).
Missing-value counts: `{"controller_features": 11100, "execution_id": 11100, "query_group": 11100, "seed": 5100}`.
Metric ranges: `{"mrr_at_10": [0.0, 1.0], "ndcg_at_10": [0.0, 1.0], "recall_at_10": [0.0, 1.0]}`.

Critical checks: full query coverage in each stratum; unique composite keys; selected_k equals actual pairs; finite nonnegative latency; metric range [0,1]; positive qrels coverage; execution manifest hash populated.

The handoff's blank controller features, query groups, and execution IDs are unavailable values, not zeros. This downstream hash records the current observed input bytes; it does not claim a historical frozen-output hash.

All critical structural, range, coverage, pair-count, and linkage checks passed.

Source aggregate comparison: PASS for the quality, pair-count, mean-latency, and non-Random percentile fields; historical Random percentile fields are not compared because they average per-seed percentiles. Improvement 1 now designates the percentiles of the 300 within-query five-seed means as primary. Complete per-seed, pooled query-seed, and historical comparisons are in `latency_cost/random_latency_percentile_sensitivity.csv`; seed coverage is complete (300 queries x five seeds per operating point), with zero missing observations. The 300 queries remain the independent population.
