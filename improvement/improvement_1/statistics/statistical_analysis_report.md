# Paired statistical analysis

Paired comparisons use the same 300 frozen TEST query IDs. PoolAdapt and Fixed Prefix are aligned by query before all tests. Random executions are averaged within each query before any cross-query analysis; Random therefore contributes 300 query-level observations, not 1,500 independent queries. The primary budget set is B10, B20, B50; B30 is excluded because actual pair means differ by more than ±5%.

Bootstrap intervals use 10,000 query-level paired bootstrap resamples (seed 20261112). The two-sided paired permutation test independently sign-flips query differences using 100,000 Monte Carlo replicates and a plus-one p-value. Holm correction is applied to three operating points separately for each metric. Wilcoxon signed-rank is supplementary with exact zero differences excluded. Cohen's dz is mean paired difference divided by sample SD; it is undefined if SD=0. A non-significant result is not evidence of equivalence.

| Budget | Metric | Mean difference | 95% bootstrap CI | Raw p | Holm p | Cohen dz |
|---|---|---:|---:|---:|---:|---:|
| B10 | nDCG@10 | 0.00085 | [-0.00090, 0.00337] | 0.75037 | 1 | 0.0460 |
| B10 | Recall@10 | 0.00233 | [0.00000, 0.00667] | 0.50006 | 1 | 0.0751 |
| B10 | MRR@10 | -0.00000 | [-0.00167, 0.00167] | 1 | 1 | -0.0000 |
| B20 | nDCG@10 | -0.00117 | [-0.01518, 0.01371] | 0.87857 | 1 | -0.0091 |
| B20 | Recall@10 | -0.01028 | [-0.03683, 0.01483] | 0.43158 | 1 | -0.0458 |
| B20 | MRR@10 | -0.00083 | [-0.01413, 0.01266] | 0.91027 | 1 | -0.0069 |
| B50 | nDCG@10 | 0.00205 | [-0.00317, 0.00741] | 0.44107 | 1 | 0.0445 |
| B50 | Recall@10 | 0.00000 | [-0.01333, 0.01333] | 1 | 1 | 0.0000 |
| B50 | MRR@10 | 0.00276 | [-0.00101, 0.00735] | 0.2229 | 0.66869 | 0.0742 |

Latency and pair-count associations are descriptive Spearman correlations and do not establish causality. See `latency_cost/latency_cost_summary.csv`. Query-group analysis is unavailable because validated TEST group labels and controller features are absent.
