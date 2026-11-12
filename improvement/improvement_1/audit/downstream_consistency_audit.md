# Downstream consistency audit

## Checks
- Phase 0 validation: PASS (11100 execution rows; 300 unique queries).
- Source aggregate comparison: PASS for quality, pair-count, mean-latency, and non-Random percentile fields; historical Random percentile fields use a different aggregation and are retained separately.
- Primary matched-budget comparisons use 300 query-level observations: PASS.
- Random primary P50/P95 are computed after averaging the five seed latencies within each query; per-seed and pooled query-seed sensitivities are recorded, with no 1,500-query inference.
- B30 is supplementary and marked unmatched where its pair-count difference exceeds ±5%.
- Compression ratio and pair reduction use actual reranked-pair means relative to Full Rerank; labels avoid monetary/GPU costs.
- Latency fields are reranking-stage latency, not end-to-end latency.
- Paired tests align by query, report bootstrap CIs, permutation p-values and per-metric Holm adjustment.
- Pareto objective uses quality versus pairs and quality versus latency separately.
- Failure cases: 151 threshold-qualified descriptive candidates; feature mechanisms remain unavailable.
- Query-group analysis is unavailable: no per-query TEST retrieval scores/ranks or retrieval feature vectors exist; query length alone does not measure retrieval ambiguity, and Phase 2 Easy/Hard labels are outcome-derived on disjoint train/validation IDs.
- Provenance limitations from upstream audit remain: two FAIL and three NOT VERIFIED; current input hashes do not establish historical output hashes.
- Raw query-level input preserved; no Phase 2 predictions, qrels, checkpoints, configurations, manifests, or other frozen source outputs were modified.

## Generated population
- Raw executions: 11100
- Query-level analysis rows after within-query Random seed averaging: 6300
- Paired tests: 9
- Pareto rows: 42
- Failure candidates: 151

See the generated CSVs and `audit/downstream_input_manifest.json` for exact values and hashes.
