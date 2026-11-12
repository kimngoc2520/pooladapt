# Failure analysis

Thresholds were defined before selecting cases. Under-allocation candidates require fewer actual pairs than same-budget Fixed Prefix and an nDCG@10 decrease of at least 0.20. Over-allocation candidates require at least 10% and at least two additional actual pairs, with no nDCG@10 gain. Allocation categories are assessed only at primary matched budgets B10/B20/B50. All qualifying records are retained.

Validated under-allocation candidates: 11.
Validated over-allocation candidates: 132.
Representative high-latency cases included: 8 (top three Tukey outliers per PoolAdapt B20, B30, and B50).

No TEST controller features or sparse/dense pool signals are available, so signal-related mechanisms cannot be tested. These are descriptive observations, not demonstrated causal failures. Full Rerank metrics are included per query for context. The CSV contains query text, paired metrics, counts, criteria, and evidence limitations.
