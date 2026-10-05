# Phase 2.1/2.3 feature handling

Candidate logging is inference-time only: it reads corpus/query inputs and an
explicit TRAIN/VALIDATION query-ID list, never qrels. Candidate relevance labels
remain in the separate qrels-derived `candidate_labels.csv` audit artifact and
are not features.

For Feature--M* association analysis, candidate records are first reduced from
exactly 100 candidates to one query record. `bm25_rank`, `dense_rank`,
`rank_displacement`, `bm25_score`, `dense_score`, and `local_rrf_margin` use
the mean over N=100 (`aggregation_method = mean_N100`). Pool and query features
already have one value per query (`aggregation_method = none`). The analysis
rejects duplicate IDs and any feature set whose query IDs do not exactly match
the official Oracle M* labels; each reported correlation audits the two row
counts separately.

`entity_count` remains in `query_features.csv`. No reliable NER system is part
of this project, so it is explicitly unavailable/blank and excluded from the
numeric Feature--M* analysis; no surrogate entity-count mechanism is invented.

The runner accepts only the three official Oracle M* label files generated from
TRAIN + VALIDATION. It rejects a test-descriptive directory and cannot run when
the official labels are absent. Spearman associations receive Holm correction;
selection uses `abs(rho) > 0.2` and `p_holm < 0.05`. Associations do not imply
causality.
