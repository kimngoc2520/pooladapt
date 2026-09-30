# Phase 2.2 Oracle M* methodology

For every TRAIN or VALIDATION query, Phase 2.2 defines the **oracle-derived
query-specific reranking budget** as the smallest member of
`{10, 20, 30, 50, 100}` whose nDCG@10 is at least `tau × nDCG@10(Full Rerank)`.
The thresholds are `tau ∈ {0.95, 0.98, 0.99}`. If no candidate budget meets the
condition, M* is 100.

`experiments/oracle_mstar.py` reads qrels and independently generated reranking
outputs only. It requires separate train and validation qrels/results, verifies
that IDs exactly match, rejects any artifact marked `test`, and combines only
the two permitted splits. It never reads feature, heuristic, selector, or
PoolAdapt artifacts.

Run it after producing fixed-prefix rerank results at every required budget and
an independent Full Rerank result for each train and validation query:

```powershell
python experiments/oracle_mstar.py --train-qrels data/scifact/qrels/train.tsv --validation-qrels <validation-qrels.tsv> --train-results <train-reranks.json> --validation-results <validation-reranks.json>
```

It writes the three label CSVs and distribution/G1 JSON summaries under
`results/phase2/02_oracle/`. `M_star_is_min` and `M_star_is_max` are audit
indicators only, as are the selected/full nDCG fields. G1 passes for a threshold
only when at least three M* levels occur and population standard deviation is
greater than zero. These labels are not a claim of a true or ground-truth
optimal budget.
