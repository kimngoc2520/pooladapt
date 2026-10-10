# Phase 2 Prefix-10 Negative-Gap Audit

## 1. Audit objective

Determine whether negative values of `gap = full_ndcg - prefix10_ndcg` in the existing train/validation development artifacts are reproducible, and whether source code or stored predictions show an implementation, metric, or data inconsistency. This audit did not change any experiment output or rerun retrieval/reranking.

Inputs inspected:

- `results/phase2/04_prefix10_gap_audit.csv`
- `results/phase2/development_reranking/train/reranking_results.json`
- `results/phase2/development_reranking/validation/reranking_results.json`
- `results/phase2/02_oracle/mstar_labels_tau095.csv`
- `data/scifact/qrels/train.tsv`
- `src/baselines/full_rerank.py`, `src/baselines/fixed_prefix.py`, `src/reranking/cross_encoder.py`, `src/evaluation/retrieval_metrics.py`, `experiments/run_baselines.py`, and `experiments/run_evaluation.py`

The 809 query IDs in the stored train and validation predictions are disjoint and together match all 809 IDs in `qrels/train.tsv`. No Test-300 query split was used.

## 2. Gap distribution

I independently recomputed nDCG@10 from each stored Top-10 document ID list and its query's qrels, then recomputed the gap. The result matches every row in `04_prefix10_gap_audit.csv` for `full_ndcg`, `prefix10_ndcg`, and `gap` (zero mismatches at `1e-12` tolerance).

| Measure | Recomputed value |
|---|---:|
| Queries | 809 |
| Gap < 0 | 107 (13.2262%) |
| Gap = 0 | 650 (80.3461%) |
| Gap > 0 | 52 (6.4277%) |
| Mean gap | 0.001060333781 |
| Median gap | 0 |
| Minimum gap | -0.569323441927 |
| Maximum gap | 1.0 |

This agrees with the existing notebook's rounded summary: mean 0.001060, median 0, gap=0 for 80.35%, and gap>0 for 6.43%.

## 3. Negative-gap cases

The ten most negative cases, ordered from smallest gap upward:

| query_id | full_ndcg | prefix10_ndcg | gap |
|---:|---:|---:|---:|
| 306 | 0.4306765581 | 1.0000000000 | -0.5693234419 |
| 1218 | 0.0000000000 | 0.4306765581 | -0.4306765581 |
| 1398 | 0.0000000000 | 0.4306765581 | -0.4306765581 |
| 389 | 0.0000000000 | 0.4306765581 | -0.4306765581 |
| 391 | 0.0000000000 | 0.4306765581 | -0.4306765581 |
| 69 | 0.0000000000 | 0.3978088012 | -0.3978088012 |
| 1186 | 0.0000000000 | 0.3868528072 | -0.3868528072 |
| 88 | 0.1934264036 | 0.5706417190 | -0.3772153153 |
| 1139 | 0.6309297536 | 1.0000000000 | -0.3690702464 |
| 1157 | 0.6309297536 | 1.0000000000 | -0.3690702464 |

M* is present for all these queries. All **107/107 negative-gap queries have M*=10** at τ=0.95. Overall, 757/809 queries have M*=10. This describes concentration in the available labels; it does not identify a cause.

## 4. Candidate-pool consistency

The checked-in runner `experiments/run_baselines.py` builds one fused BM25 + Dense + RRF pool per query, enforces exactly 100 candidates, enriches that pool once, and then:

- passes all candidates to `rerank_full`;
- passes `select(candidates, budget)` to Fixed Prefix; `select` returns the first `budget` entries without reordering them.

The source implementation therefore defines Full Rerank as reranking all 100 candidates and Prefix-10 as reranking the first 10 from the same hybrid order. Both use the same `query` value and candidates enriched from the same corpus records in that runner.

Stored development artifacts report pool size 100, Top-K 10, and `reranked_pairs=100` for Full Rerank versus 10 for Prefix-10. There are no duplicate document IDs in the saved Top-10 lists. Full Rerank and `fixed_prefix_100` output lists are identical for all 809 stored query predictions. For Full vs Prefix-10, the saved Top-10 lists have no pairwise ordering reversals among documents shared by both lists.

**Evidence limitation:** the train/validation JSON stores only the resulting Top-10 document IDs, pair counts, and latency. It does not store the original 100 candidate IDs, the exact Prefix-10 input list, query/document text, or reranker scores. Also, the checked-in `run_baselines.py` is configured for the test split; no checked-in train/validation generation runner was found. Consequently, the common-pool and no-retrieval-rerun properties are established by the shared source implementation and consistent output metadata, but cannot be independently reconstructed per query from these development artifacts alone. No mismatch is visible in the stored IDs or metadata.

## 5. Reranking consistency

Source code and artifact metadata show:

- Full Rerank budget `K=100`; Prefix-10 `K=10`.
- Both are recorded with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- `rerank_full` calls `Reranker.rerank`; Prefix calls that same rerank method with the selected prefix.
- `CrossEncoderReranker` creates `(query, document_text)` pairs, uses one `model.predict` scoring path, and sorts both result sets by `reranker_score` descending.
- The experiment runner shares one reranker instance for all methods in a query loop and passes the same query string.
- For the same candidate ID, document text is taken from the same enriched corpus record in the common runner.

No distinct scoring or sort-direction implementation for Full vs Prefix-10 was found. Exact train/validation query text and document text cannot be audited against saved predictions because they are not stored there. Full vs Prefix-100 exact Top-10 parity on all 809 queries is consistent with shared full-pool ranking behavior.

## 6. Metric/evaluation consistency

`src/evaluation/retrieval_metrics.py::ndcg_at_k` is applied to the ordered document IDs and qrels mapping. In `experiments/run_evaluation.py`, prediction document IDs and query IDs are cast to strings before qrels lookup; qrels IDs are strings as well. The independent recomputation used the same string mapping and the same DCG/IDCG definition:

`DCG@10 = sum(rel_i / log2(i + 1))`, with one-based rank `i`; `IDCG@10` sorts the query's qrels relevance values descending and takes the first ten.

For this SciFact train qrels population, every relevance grade is 1 and all 809 qrels queries have at least one relevant document. Thus no no-relevant-query special case affected this audit. The metric returns 0 if IDCG is 0 in general. It uses the first ten IDs passed to it and does not use candidate-pool size in the calculation. It does not have special tie-group handling: it evaluates the emitted order. The reranker uses a descending Python stable sort, so exact score ties preserve input candidate order.

Both methods were evaluated with the same function-equivalent calculation, same query qrels, same cutoff, and same ID mapping. No metric inconsistency was found. Actual tied score pairs cannot be examined from the saved development JSON because per-document reranker scores are not included.

## 7. Representative negative-gap query inspection

The qrels for each example below contain one relevant document, so IDCG@10=1. Labels in brackets are qrels relevance (1 relevant, 0 otherwise). The listed document order is the stored reranked Top-10 order.

### Query 306 (M*=10)

- Qrels relevant document: `7821634`.
- Full Rerank: `35962023[0], 3981033[0], 23848916[0], 7821634[1], 15248287[0], 23403754[0], 37444589[0], 20738970[0], 24624992[0], 6121555[0]`.
- Prefix-10: `7821634[1], 23403754[0], 20738970[0], 10852047[0], 9169645[0], 30580263[0], 238409[0], 5254463[0], 40608679[0], 33667484[0]`.
- The relevant document is at rank 4 for Full Rerank and rank 1 for Prefix-10. The scores are `1/log2(5)=0.4306765581` and `1.0`, respectively; gap `-0.5693234419`.

### Query 1218 (M*=10)

- Qrels relevant document: `15635366`.
- Full Rerank: `3829232[0], 24790460[0], 4312169[0], 11902109[0], 12685434[0], 1649738[0], 14191255[0], 37686718[0], 14328288[0], 5099266[0]`.
- Prefix-10: `3829232[0], 14328288[0], 16562534[0], 15635366[1], 8208212[0], 9646449[0], 9791313[0], 6268106[0], 19572798[0], 28086354[0]`.
- The relevant document is outside Full Rerank's Top-10 and at rank 4 for Prefix-10. The scores are `0.0` and `1/log2(5)=0.4306765581`; gap `-0.4306765581`.

### Query 1398 (M*=10)

- Qrels relevant document: `17717391`.
- Full Rerank: `36399107[0], 19343151[0], 470625[0], 18218379[0], 21219071[0], 7898952[0], 1574014[0], 21562657[0], 16627684[0], 29422484[0]`.
- Prefix-10: `19343151[0], 18218379[0], 7506409[0], 17717391[1], 25419778[0], 8533245[0], 83308790[0], 6492658[0], 6422576[0], 37362689[0]`.
- The relevant document is outside Full Rerank's Top-10 and at rank 4 for Prefix-10. The scores are `0.0` and `1/log2(5)=0.4306765581`; gap `-0.4306765581`.

These examples show the score arithmetic from stored rankings and qrels. They do not establish why a particular ranking occurred.

## 8. Interpretation of negative gaps

Under this protocol, **Full Rerank** scores all 100 candidates and evaluates the first 10 after reranking. **Prefix-10** scores only the first 10 retrieved candidates and evaluates those ten after reranking.

The candidate sets and returned rankings are not nested after scoring: Full Rerank can move documents from below the original prefix into its final Top-10, while Prefix-10 can rank an initially retrieved relevant document higher. Adding more scored candidates does not mathematically guarantee a higher per-query nDCG@10 because the metric measures the resulting ordered Top-10, not the number of candidates scored. Therefore negative per-query gaps are possible under this protocol without themselves indicating a bug.

## 9. Final audit verdict

**D. No implementation inconsistency found; negative gaps are valid per-query outcomes under the current protocol.**

The independent metric recomputation exactly reproduces the audit CSV; source code uses the same reranker/scoring/sort path for the two methods; and inspected examples' nDCG values follow directly from their stored ranks and qrels. The report's candidate-pool provenance limitation remains: exact input candidate IDs/text for the train/validation artifacts are not saved, and their generating runner is absent from the checked-in source.

## 10. Recommended next step

For future reranking artifacts, persist (or hash) each query's ordered Top-100 candidate IDs, Prefix-10 input IDs, query text, model/config identifier, and per-candidate reranker scores. Keep the negative-gap cases in the analysis. These additions would allow exact per-query input and score provenance checks without changing the present results or protocol.
