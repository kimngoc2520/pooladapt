# Adapted PACE-EF baseline

`experiments/run_pace_ef.py` evaluates Evidence Frontloading on the existing
Phase 2 SciFact train + validation population. It starts from each logged
100-document BM25 + Dense + RRF pool and changes only its ordering before
selecting Prefix-K budgets `{10, 20, 30, 50, 100}`.

This is an adaptation of PACE Evidence Frontloading for PoolAdapt: it replaces
the original SPLADE-v3 sparse representation with document-side BM25 term
factors using the repository BM25 tokenizer, IDF values, `k1`, `b`, and
`avgdl`; uses each candidate's existing within-pool RRF score for relevance
normalization; and applies the greedy maximum-evidence marginal to candidate
reranking on SciFact. It does not implement the original system's
Pressure-Adaptive Budgeting or soft-anchor mechanisms and is not a faithful
reproduction of full PACE.

Selection receives candidate lexical factors, RRF scores/ranks, and IDs only.
Train qrels are loaded separately for evaluation metrics after reranking.
The runner writes query-level results to
`results/phase2/06_pace_ef/pace_ef_results.csv` and a separate permutation
audit to `results/phase2/06_pace_ef/pace_ef_ordering_audit.csv`.

Run with:

```powershell
python experiments/run_pace_ef.py
```
