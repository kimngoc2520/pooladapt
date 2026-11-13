# PoolAdapt: Candidate-Pool-Aware Selective Reranking for Large Candidate Pools in Agentic Hybrid RAG under Inference Cost Constraints

*Supplementary Query-group Analysis for Improvement 1 · frozen SciFact TEST set (300 queries)*

> This is a subsequent descriptive analysis of reconstructed retrieval features. It updates the earlier “unavailable” status recorded in the Improvement 1 report for query-group analysis. The original handoff fields `controller_features` and `query_group` remain blank; the groups below are analytical strata, not controller outputs or a new evaluated policy.

## 1. Executive Summary

PoolAdapt studies whether candidate-pool signals can guide how much of a hybrid retrieval pool is sent to a Cross-Encoder. Query-group Analysis asks whether the frozen TEST queries differ in ways that may matter to this decision. It is a supplementary analysis of Improvement 1, not a separate algorithmic contribution.

The analysis uses 300 frozen SciFact TEST queries, saved BM25, Dense, and RRF Top-100 rankings, reconstructed pre-reranking features, and the frozen query-level PoolAdapt and Fixed Prefix outcomes. The reconstructed RRF list matched the frozen Phase 2 pool in exact order and membership for all 300 queries. Two balanced, outcome-independent median splits were examined: Top-20 lexical redundancy and Top-20 BM25–Dense set overlap.

The clearest pattern is that the high-overlap group has stronger RRF retrieval quality before reranking than the low-overlap group: mean nDCG@10 is 0.7308 versus 0.6325, and mean Recall@10 is 0.8651 versus 0.7467. At target B30, PoolAdapt’s nDCG@10 is 0.7399 in the high-overlap group and 0.6113 in the low-overlap group; its paired difference from Fixed Prefix is +0.0179 and −0.0027, respectively. These are descriptive contrasts, not a tested group-by-method interaction. Realized pair counts also differ from the target label, so B30 is not an equal-cost comparison.

The evidence supports heterogeneous retrieval conditions and suggests that candidate-pool characteristics may help identify different reranking needs. It does not show that an adaptive policy causally improves quality or cost. Dense component ranks and scores are reconstructed with a model revision that is not verified as historical. The RRF ordering match validates only the fused candidate order; it does not verify historical BM25/Dense component rankings or raw scores.

## 2. Research Motivation

Uniformly reranking a large pool spends Cross-Encoder computation on every query, even when the fused candidates may already be well ordered. Conversely, restricting reranking uniformly can omit useful evidence for queries whose sparse and dense retrievers disagree. PoolAdapt’s research question is whether observable characteristics of the candidate pool can inform a query-level reranking budget while preserving quality under inference-cost constraints.

Query groups make this question inspectable. They can reveal whether retrieval quality, reranking outcomes, and measured reranking cost vary across pre-reranking pool conditions. Such contrasts help motivate candidate-pool-aware selection, but they do not establish that the feature caused a reranking outcome or that a selector can exploit it reliably.

## 3. Research Questions

- **RQ1 — Feature variation:** How do pool redundancy, candidate overlap, rank agreement, score profiles, and query descriptors vary across the frozen TEST queries? **Supported descriptively**, with the Dense reconstruction caveat below.
- **RQ2 — Sparse–dense complementarity:** How much do BM25 and Dense Top-*k* lists overlap, and how similar are their shared-document ranks? **Supported for reconstructed component rankings**; exact RRF order does not prove historical BM25/Dense ranks.
- **RQ3 — Group differences:** Do the preassigned groups differ in retrieval quality and PoolAdapt-versus-Fixed-Prefix outcomes? **Supported descriptively**; no confirmatory interaction test was run.
- **RQ4 — Quality and cost:** How do nDCG@10, reranked pairs, and measured reranking latency vary by group? **Supported at the saved operating points**; latency is reranking-stage only, and target-budget labels do not guarantee equal realized costs within a group.
- **RQ5 — Policy implication:** Can these observations establish the value of an adaptive selector? **No.** They motivate further tests but do not validate a new TEST-time policy.

## 4. Experimental Data and Methodology

### Frozen population and provenance

The TEST population is the 300 unique IDs in `results/phase2/evaluation/locked_configurations.json`. The query-group artifacts cover exactly these IDs. The saved BM25, Dense, and RRF tables each contain 30,000 rows: 100 unique, contiguous ranks for each of 300 queries, with no duplicate query–document or query–rank pairs. The RRF reconstruction passed `GATE_A_EXACT_MATCH`: ordered Top-100 and Top-100 membership matched 300/300 frozen pools. This validates the fused candidate order, not the historical raw BM25/Dense scores or the historical Dense model revision.

Dense retrieval was reconstructed locally on CPU with `sentence-transformers/all-MiniLM-L6-v2`; its revision is explicitly `RECONSTRUCTED_CANDIDATE_REVISION_NOT_VERIFIED_HISTORICAL`. Reconstructed component scores are not claimed numerically identical to the original GPU run. BM25, Dense, and RRF score scales are kept separate and are never compared in magnitude.

### Features and reconstruction

The feature table has one row per TEST query, 109 columns, no missing cells, and no duplicate query IDs. The original 13 PoolAdapt features were recalculated and matched the canonical `frozen_features` function for all 3,900 query-feature pairs at `rtol=1e-10`, `atol=1e-12`. The computed TEST values remain reconstructed. The separate 809-row TRAIN/VALIDATION feature table was used only for schema and distribution checks; it has no TEST ID overlap and supplied no substitute TEST values.

The 13 features fall into four research-relevant families:

1. **RRF decisiveness:** top-1/top-2 score margin, normalized score entropy, score Gini, and Top-10 score concentration describe how sharply the fused ranking’s reconstructed score mass is distributed. They could indicate whether a short prefix is decisive, but they are score-derived and subject to the reconstruction limitation.
2. **Sparse–dense complementarity:** Top-20 overlap, correlation of BM25 and Dense ranks on shared Top-100 documents, and normalized rank displacement/derived agreement describe whether the two retrieval signals return the same candidates and place shared candidates similarly. They could indicate complementary evidence, but overlap and rank agreement are distinct properties.
3. **Pool text redundancy:** mean token-set Jaccard among RRF Top-20 documents measures lexical repetition. It is only a lexical proxy; it does not identify semantic duplicates or evidence coverage.
4. **Query descriptors:** token count, numeric-token count, and mean/max BM25 IDF describe query length and lexical specificity. These may affect retrieval difficulty but were not used for the two group splits.

Pool size and ordinal-position summaries are constant because every query has a 100-candidate pool; they cannot distinguish queries in this experiment. Empty-document and exact duplicate-content ratios are also constant at zero.

### Grouping and evaluation

Groups were assigned before joining outcomes, using feature medians only. Ties are assigned to the lower group. The redundancy split uses mean pairwise token-set Jaccard among the RRF Top-20. The sparse-dense candidate-overlap group is based on Top-20 set Jaccard from reconstructed BM25 and Dense lists. For backward compatibility, the assignment artifact retains the internal column name `sparse_dense_agreement_group`; it denotes this candidate-set-overlap split, not rank-order agreement. Each split contains 150 low-overlap and 150 high-overlap queries. Neither qrels, relevance labels, nor reranking outcomes were used to assign groups. TEST outcomes were joined later by exact `query_id`.

Pre-reranking retrieval quality was calculated from saved Top-100 rankings and TEST qrels with the repository’s `ndcg_at_k`, `recall_at_k`, and `mrr_at_k` definitions (linear relevance gain for nDCG). This is a read-only metric calculation; no retrieval, reranking, or model inference was run. Post-reranking subgroup comparisons use the existing frozen `test_query_metrics.csv`, paired by query, for PoolAdapt and Fixed Prefix at target budgets 10, 20, 30, and 50. The existing subgroup table reports group means, medians, standard deviations, paired differences, and 2,000 paired query bootstrap percentile intervals (seed 42). Its intervals are exploratory, unadjusted across the many group/metric/budget comparisons, and are not treated as confirmatory significance tests.

Quality metrics are nDCG@10, Recall@10, and MRR@10. Cost is reported as actual reranked pairs per query and measured reranking latency. The latency timer covers pair preparation, Cross-Encoder inference, and sorting; it excludes retrieval, selection, model loading, and warm-up. It is not end-to-end latency or monetary/GPU cost.

## 5. Results and Analysis

### 5.1 Query Feature Analysis

The following summaries are query-level means or medians over 300 queries. The two overlap measures use different denominators: the original PoolAdapt `top20_overlap` is intersection/20, while the group feature is Jaccard, intersection/union.

| Feature family / measure | Observed summary | Interpretation and limitation |
|---|---:|---|
| Candidate pool size | 100 for every query | Fixed by design; pool size itself cannot drive query-level adaptation here. |
| BM25–Dense Top-20 intersection | Mean 6.02 of 20; `top20_overlap` mean 0.301 | The retrievers contribute many non-shared candidates near the top. |
| Top-20 set Jaccard | Mean 0.190; median 0.160 | Substantial query variation; the median split separates low from high overlap. |
| Top-100 set overlap | Mean Jaccard 0.165; median 0.149; mean shared count 27.26 | Most documents in each source Top-100 are not shared. The fused Top-100 truncates their union. |
| Shared-rank correlation and displacement | Pearson rank correlation mean 0.278; mean absolute displacement 26.25 ranks; derived normalized agreement mean 0.735 | Set overlap alone does not describe ordering agreement. The value labeled `component_shared_rank_spearman` in the feature CSV is computed with `numpy.corrcoef` and is Pearson correlation; this report follows the computation, not that auxiliary column’s name. |
| Rank-Biased Overlap, p=0.90 | Mean 0.355 | Full-list order agreement is moderate to low in this reconstruction. |
| RRF Top-20 lexical redundancy | Mean Jaccard 0.1085; median 0.1069 | Top-20 documents have limited shared token content on average, with a usable range for a descriptive split. |
| Exact duplicates / empty documents | 0 across Top-20 and Top-100 | No exact token-set duplicates or empty documents were found; semantic redundancy is not ruled out. |
| Unique-token ratio | Mean 0.524 in Top-20; 0.310 in Top-100 | The union of document tokens becomes less unique relative to the sum of document token sets as the pool expands. This is a lexical diversity proxy. |
| Reconstructed RRF scores | Mean entropy 4.531 (maximum ln(100)=4.605); Gini 0.206; Top-10 mass 0.191; top-1/top-2 margin 0.00145 | Suggests a relatively spread score profile, but score summaries are reconstructed, not verified historical scores. |
| Query descriptors | Mean 13.07 tokens; 0.213 numeric tokens; mean-query average IDF 3.401 and max IDF 6.877 | Length and IDF vary, but neither formed the reported groups; IDF values depend on the reconstructed BM25 index. |

These distributions show why a fixed pool size does not imply uniform pool structure. Candidate overlap, ordering, and lexical redundancy vary even though all queries supply 100 fused candidates. They are plausible signals for an allocation policy, but the observed distributions alone do not demonstrate predictive value for reranking utility.

### 5.2 Query-group Distribution

| Grouping strategy | Low group mean / median | Threshold | High group mean / median | Counts |
|---|---:|---:|---:|---:|
| Top-20 lexical redundancy | 0.0977 / 0.0990 | 0.106945 | 0.1194 / 0.1160 | 150 / 150 |
| BM25–Dense Top-20 candidate-set overlap | 0.0914 / 0.1111 | 0.159664 | 0.2890 / 0.2500 | 150 / 150 |

The splits are balanced and interpretable as descriptive strata. Median thresholds are sample-dependent, discard within-group variation, and do not establish operational cutoffs. The candidate-overlap groups depend on reconstructed BM25/Dense rankings and therefore inherit their provenance limitation.

### 5.3 Candidate-pool Retrieval Quality Before Reranking

| Saved ranking | Mean nDCG@10 | Mean Recall@10 | Mean MRR@10 |
|---|---:|---:|---:|
| BM25 | 0.6519 | 0.7740 | 0.6186 |
| Dense | 0.6451 | 0.7833 | 0.6047 |
| RRF | 0.6816 | 0.8059 | 0.6472 |

RRF has the highest mean among these saved/reconstructed rankings on all three metrics. This is a descriptive comparison on the 300 TEST queries. Exact RRF candidate order matches the frozen pool, while Dense’s historical model revision is unverified; the component results should therefore not be presented as a verified historical retrieval comparison.

RRF quality also differs across the preassigned groups. Here, high-overlap and low-overlap refer to BM25-Dense Top-20 candidate-set Jaccard, not rank-order agreement:

| Group | Query count | RRF nDCG@10 mean (SD) | Recall@10 mean | MRR@10 mean |
|---|---:|---:|---:|---:|
| Redundancy low | 150 | 0.6679 (0.4082) | 0.7833 | 0.6332 |
| Redundancy high | 150 | 0.6954 (0.3669) | 0.8284 | 0.6611 |
| Low overlap | 150 | 0.6325 (0.4233) | 0.7467 | 0.6017 |
| High overlap | 150 | 0.7308 (0.3429) | 0.8651 | 0.6926 |

All four groups have median Recall@10 and MRR@10 of 1.0, so their means and distributions are more informative than medians for these ceiling-prone metrics. The high-overlap group starts with stronger RRF retrieval quality; later reranking differences must be interpreted against this baseline rather than as evidence that candidate overlap itself causes reranking gains.

### 5.4 Retrieval Rank Agreement, Candidate Overlap, Redundancy, and Diversity

Mean BM25–Dense Top-20 Jaccard is 0.190 (median 0.160), and mean Top-100 Jaccard is 0.165. At Top-100, the two lists share about 27 of 100 documents on average. Shared-document rank correlation is much lower than perfect agreement (mean Pearson 0.278), while the separately defined normalized displacement agreement averages 0.735. These measures answer different questions: one measures rank displacement among shared documents; the other measures set overlap at a chosen depth. Neither by itself indicates whether the Cross-Encoder will change the best Top-10 ordering.

Top-20 lexical redundancy averages 0.1085, but exact duplicate-content ratio is zero. The result supports variation in lexical overlap, not a claim that high-redundancy pools contain duplicate evidence or that reranking can safely be skipped. No semantic diversity or evidence-coverage metric is available.

### 5.5 Reranking and Quality–Cost Trade-off by Group

The table gives a comparison at the B30 target-budget label. The label is distinct from each method's actual reranked pairs and measured reranking-stage latency. RRF is pre-reranking quality; PoolAdapt and Fixed Prefix values come from frozen query-level outcomes. Differences are paired PoolAdapt minus Fixed Prefix; intervals are 2,000-resample exploratory bootstrap intervals. Relative nDCG change is relative to the Fixed Prefix group mean.

| Group | RRF nDCG@10 | Fixed Prefix nDCG@10 | PoolAdapt nDCG@10 | Δ nDCG@10 (95% CI; relative) | Mean pairs, PoolAdapt / Fixed | Mean latency (ms), PoolAdapt / Fixed |
|---|---:|---:|---:|---:|---:|---:|
| Redundancy high | 0.6954 | 0.6870 | 0.6961 | +0.0091 [−0.0060, 0.0266]; +1.32% | 37.07 / 30.00 | 193.49 / 175.84 |
| Redundancy low | 0.6679 | 0.6490 | 0.6551 | +0.0060 [−0.0161, 0.0284]; +0.93% | 28.67 / 30.00 | 150.60 / 174.80 |
| High overlap | 0.7308 | 0.7221 | 0.7399 | +0.0179 [0.0022, 0.0369]; +2.47% | 31.80 / 30.00 | 169.62 / 175.88 |
| Low overlap | 0.6325 | 0.6140 | 0.6113 | −0.0027 [−0.0256, 0.0203]; −0.45% | 33.93 / 30.00 | 174.47 / 174.76 |

The high-overlap B30 interval is above zero in this unadjusted exploratory table, but no multiple-comparison correction or group-by-method interaction test was performed; it is not a confirmatory significance claim. At B30, PoolAdapt averages 31.8 pairs in high-overlap queries and 33.93 in low-overlap queries, versus 30 for Fixed Prefix. Thus the target label does not produce equal realized group costs.

The B50 results give a second cost contrast. In high-overlap queries, PoolAdapt uses 45.27 pairs versus 50 for Fixed Prefix (−9.47%) and averages 234.07 ms versus 257.38 ms, with a nDCG@10 difference of +0.00268 (95% CI [0.00000, 0.00781]). In low-overlap queries it uses 53.40 pairs (+6.80%) and averages 263.78 ms versus 250.70 ms; the nDCG@10 difference is +0.00142 (95% CI [−0.00850, 0.01042]). These are descriptive, group-specific workload and outcome contrasts, not proof that candidate-set overlap predicts a causal benefit.

The broader Improvement 1 analysis predeclared B10, B20, and B50 as its primary matched-budget comparisons; B30 is supplementary because overall actual pair means differ by more than the ±5% matching tolerance. QGA’s group-level B30 comparisons are therefore especially unsuitable as equal-cost treatment comparisons. Across the full Improvement 1 primary tests, the report records no Holm-adjusted permutation p-value below 0.05.

No QGA-specific figure existed in the inspected artifacts. The available Improvement 1 figures summarize aggregate methods rather than these groups, so the reports use verified tables and do not reuse those figures as if they were group analyses.

## 6. Discussion

The groups separate different retrieval conditions. High-overlap queries have materially stronger pre-reranking RRF quality than low-overlap queries, and the low-overlap group has larger nDCG variability. This suggests that candidate-set overlap group membership is associated with query difficulty or retrieval evidence quality, but it does not isolate reranking's causal effect. The component rankings defining overlap are reconstructed; RRF gate success verifies only the fused candidate order and does not verify historical BM25/Dense rankings, raw scores, or Dense model revision.

Reranking outcomes vary by group and budget. At B30, PoolAdapt's difference from Fixed Prefix is positive in the high-overlap group and slightly negative in the low-overlap group, while realized pairs and measured reranking-stage latency vary across strata. At B50, the high-overlap stratum has lower observed pair count and reranking-stage latency for PoolAdapt, whereas the low-overlap stratum has higher values. These patterns are consistent with heterogeneous reranking needs, but intervals, multiple comparisons, group construction, and budget mismatch limit interpretation. The component rankings used to form these overlap groups are reconstructed; matching the RRF pool order does not verify historical BM25/Dense rankings or Dense model revision.

For PoolAdapt, the practical implication is to treat candidate-pool descriptors as candidate signals for a future TRAIN/VALIDATION-calibrated selector, not as fixed TEST rules. A valid policy test would freeze its features, model, and calibration before TEST; compare methods at matched realized pair budgets; and assess quality and latency on a separate untouched evaluation population. This group analysis alone does not validate an adaptive policy.

## 7. Limitations and Threats to Validity

- **Population:** 300 SciFact TEST queries, split into groups of 150. Results may not generalize to FiQA, HotpotQA, other corpora, or other Agentic Hybrid RAG systems.
- **Reconstruction:** Exact RRF ordering does not verify historical BM25/Dense ranks, raw scores, or Dense model revision. Feature values dependent on those reconstructed components are provisional.
- **Feature naming:** The auxiliary `component_shared_rank_spearman` field is computed as Pearson correlation in the extractor. The frozen model input `rank_correlation_union` is also Pearson correlation by the canonical function. This naming issue is disclosed, not silently rewritten.
- **Grouping:** Median splits are sensitive to this TEST sample, discard continuous information, and are not learned policy thresholds. The two grouping dimensions are analyzed separately; their intersections are not studied.
- **Metric scope:** Recall@10 uses total qrels-relevant documents. The reranking handoff does not provide Recall@100. Latency omits retrieval, selection, loading, and warm-up and is not end-to-end or monetary cost.
- **Inference:** Bootstrap intervals are exploratory and not corrected across all subgroup comparisons. No formal interaction test was performed; mean differences do not establish causality.
- **Operational availability:** The original Improvement 1 handoff still has blank `controller_features` and `query_group` columns. The analysis groups are computed separately and were not inserted into that handoff.
- **Diversity:** Token-set Jaccard and unique-token ratio are lexical proxies; semantic redundancy, evidence sufficiency, and document relevance are not measured by these features alone.

## 8. Conclusion and Implications for PoolAdapt

The frozen RRF pool is exactly 100 candidates for every query, but its pre-reranking characteristics vary. Top-20 redundancy and sparse–dense overlap form balanced, outcome-independent descriptive groups. The candidate-overlap split aligns with a large difference in RRF retrieval quality before reranking. PoolAdapt-versus-Fixed-Prefix results then vary by group and operating point, including contrasting pair-count and reranking-stage latency patterns at B50. The overlap-group rankings are reconstructed; matching the RRF pool order does not verify the historical BM25/Dense component rankings or Dense model revision.

These findings support the research premise that query and pool conditions may matter when allocating reranking effort. They do not prove an adaptive policy’s effectiveness, establish a causal mechanism, or resolve historical Dense provenance. Improvement 1 remains the main experiment; Query-group Analysis adds a descriptive lens and a set of hypotheses for a separately calibrated policy evaluation.

## 9. Reproducibility and Artifact References

- **Inputs:** frozen IDs in `results/phase2/evaluation/locked_configurations.json`; TEST qrels in `data/scifact/qrels/test.tsv`; frozen reranking outcomes in `results/phase2/evaluation/final_gpu/test_query_metrics.csv`; saved candidate IDs in `test_predictions.jsonl`.
- **Retrieval and feature tables:** [BM25](../reconstruction/bm25_test_top100.csv), [Dense](../reconstruction/dense_test_top100.csv), [RRF](../reconstruction/rrf_test_top100.csv), [query features](../features/test_query_features.csv), [13-feature matrix](../features/original_13_feature_matrix_reconstructed.csv), and [group assignments](../groups/query_group_assignments.csv).
- **Results:** [pre-reranking quality by group](pre_reranking_retrieval_quality_by_group.csv) and [PoolAdapt–Fixed Prefix subgroup comparisons](subgroup_comparison.csv). The latter contains all four target budgets, five metrics, group means/medians/SDs, and paired bootstrap intervals.
- **Methods and validation:** [grouping methodology](../groups/grouping_methodology.md), [feature provenance](../features/feature_provenance.md), [feature consistency audit](../features/development_feature_consistency.md), [RRF validation](../reconstruction/rrf_pool_validation_summary.md), [data integrity audit](../audit/data_integrity_audit.md), [leakage audit](../audit/leakage_audit.md), and [final validation report](../audit/final_validation_report.md).
- **Scripts:** `reconstruct_test_retrieval.py`, `extract_test_features.py`, `assign_query_groups.py`, `evaluate_query_groups.py`, and the supplementary `evaluate_retrieval_quality.py` under `../scripts/`. The supplementary command is `py -3.11 improvement/improvement_1/query_group_analysis/scripts/evaluate_retrieval_quality.py`; it reads saved rankings and qrels and refuses to overwrite its output. Run it only in a fresh output tree or after preserving the existing summary. The full reconstruction and subgroup scripts also protect existing outputs; do not rerun the completed pipeline merely to read these reports.
- **Reproducibility checks:** focused tests and syntax/import checks were run after the report and analysis changes; no retrieval, Cross-Encoder inference, training, or full experimental pipeline was run for this report.

The `reconstruction/artifact_checksums.json` file records the original reconstruction outputs and predates this supplementary quality table and these reports. It has not been overwritten; the new analysis CSV is reproducible from its script and the cited frozen inputs.
