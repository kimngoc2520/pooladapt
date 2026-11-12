# Phase 2 Easy/Hard Reuse Assessment

**Verdict: NOT REUSABLE** for Improvement 1 TEST query-group analysis. The Phase 2 feature/label tables cover 809 train/validation queries, with zero ID overlap against the frozen 300-query TEST set. Easy/Hard labels are derived from reranking outcomes. The Phase 2 final TEST metrics do cover the same 300 IDs, but they contain no per-query feature vectors or Easy/Hard labels.

## Source chain and represented population

The aggregate is `results/phase2/05_easy_hard_analysis/feature_easy_hard_analysis.csv` (14 feature rows). Its producer, `experiments/analyze_easy_hard_features.py`, reads:

- `results/phase2/01_candidate_logging/pool_features.csv`
- `results/phase2/01_candidate_logging/query_features.csv`
- `results/phase2/02_oracle/mstar_labels_tau099.csv`

The per-query ID source is `results/phase2/01_candidate_logging/train_validation_query_ids.csv`. It has 809 unique IDs and exactly matches the union of `data/scifact/splits/train_query_ids.txt` (647 IDs) and `data/scifact/splits/validation_query_ids.txt` (162 IDs); the two splits do not overlap. `data/scifact/splits/split_metadata.json` says these are an 80/20 query-level split of original-train IDs (seed 42), with zero TEST overlap. This is not the Improvement 1 TEST population.

The per-query files contain 809 unique rows each in `query_features.csv`, `pool_features.csv`, and `mstar_labels_tau099.csv`, with no duplicate query IDs and identical ID sets. `candidate_features.csv` contains 80,900 rows: 100 candidates for each of the same 809 IDs. The label table has 809 unique IDs. The Easy/Hard summary reports 757 Easy and 52 Hard for each usable feature; its `entity_count` row is unavailable (`n_easy=0`, `n_hard=0`, `missing_count=809`).

## Easy/Hard label construction

`experiments/oracle_mstar.py` loads TRAIN and VALIDATION qrels and their reranking outputs and explicitly rejects a TEST result artifact. It writes `source_splits: [train, validation]`. `src/analysis/oracle_mstar.py` defines `M*` as the smallest fixed prefix budget in {10, 20, 30, 50, 100} whose qrels-based nDCG@10 is at least `tau × Full Rerank nDCG@10`. The file used by the Easy/Hard script is `mstar_labels_tau099.csv`, so tau is 0.99. `analyze_easy_hard_features.py` labels M*=10 as Easy and M*>10 as Hard.

Thus the groups are **not independent of evaluation outcomes**: the label uses qrels and observed fixed-prefix and Full Rerank rankings/metrics. The feature extractor itself does not read qrels (`metadata.json` records `qrels_read_by_feature_extractor: false`), but that does not make the outcome-derived group label independent. The M* distribution is 757 at 10, 33 at 20, 7 at 30, 11 at 50, and 1 at 100.

## Feature values and preprocessing

`experiments/run_candidate_logging.py` produces the per-query features from retrieval results and query text. It records BM25 Top-100 plus dense Top-100, RRF Top-100 with k=60, and `sentence-transformers/all-MiniLM-L6-v2` in `metadata.json`.

- Pool features: top-1 minus top-2 RRF score; BM25/Dense Top-20 overlap divided by 20; Pearson correlation of BM25 and dense ranks for candidates present in both Top-100 lists; entropy of normalized RRF scores; Gini of RRF scores; mean pairwise token-set Jaccard of title+text for the RRF Top-20; Top-10 RRF score mass divided by total pool score; and sparse/dense agreement and disagreement, calculated from mean absolute rank displacement normalized by 99 (agreement is one minus disagreement).
- Query features: token count and numeric-token count using the BM25 tokenizer (`lower()` plus `\\w+`); average and maximum BM25 IDF among query tokens found in the corpus index; and `entity_count`, which is blank/unavailable.

The Phase 2 metadata's corpus and query SHA-256 values match the current SciFact `corpus.jsonl` and `queries.jsonl`. Its retrieval pool parameters match the documented Improvement 1 retrieval design (100-candidate pool, BM25+dense RRF, k=60), so the **feature definitions are methodologically relevant**. This does not provide TEST feature values or prove exact per-query feature equality with the historical Improvement 1 retrieval execution. Improvement 1's handoff has blank `controller_features`; its provenance audit still records historical traceability failures and unverified items.

## TEST ID comparison and per-query search

The authoritative comparison set is the unique `query_id` set in `improvement/improvement_1/matched_budget/query_level_evaluation.csv` (11,100 execution rows, 300 unique queries).

| Compared file/set | Rows | Unique query IDs | Overlap with Improvement 1 TEST | Missing TEST IDs | Phase 2 IDs outside TEST | Duplicate query IDs in the per-query set |
|---|---:|---:|---:|---:|---:|---:|
| Phase 2 `train_validation_query_ids.csv` / query and pool features / M* labels | 809 each | 809 | 0 | 300 | 809 | 0 |
| Improvement 1 authoritative handoff | 11,100 | 300 | — | — | — | 10,800 repeated execution rows; 0 duplicate IDs after deduplication |
| Phase 2 `evaluation/final_gpu/test_query_metrics.csv` | 11,100 | 300 | 300 | 0 | 0 | Repeated method/budget/seed rows, as expected |

For the feature/label set, the exact overlap is empty; the missing TEST set is all 300 IDs in the authoritative handoff, and the 809 unexpected IDs are all IDs in the Phase 2 train/validation list. The Phase 2 TEST metrics have the same 300-ID set as the handoff, but their columns are query ID, method/budget/seed, and evaluation metrics/latency—not features or Easy/Hard labels. `evaluation/final_gpu/test_predictions.jsonl` contains rankings and execution fields, also with no feature vector or M* label. No per-query TEST file combining IDs, Easy/Hard labels, and feature values was found among the Phase 2 query-ID CSVs or TEST outputs.

## Compatibility conclusion and limitations

The Easy/Hard artifact cannot support an Improvement 1 TEST group comparison: its 809 feature/label IDs are disjoint from TEST, and the labels are oracle outcomes rather than independent query-group labels. The same TEST IDs appear in Phase 2 test metrics, but no saved TEST features or independent group labels accompany them. Deriving groups from those final outcomes would violate the stated constraint and introduce outcome leakage. Reusing the feature definitions as a future extraction specification is possible, but obtaining validated TEST vectors/labels would require a separately authorized, provenance-reviewed procedure; this assessment did not perform one.

The current Easy/Hard aggregate hash matches the value recorded for it in `results/phase2/evaluation/inspection_manifest.json`. That confirms artifact-byte consistency only. It does not remove Improvement 1's documented upstream limitations (12 PASS, 2 FAIL, 3 NOT VERIFIED) or establish historical reproducibility.

**Improvement 1 Query-group analysis should remain UNAVAILABLE.** No existing Improvement 1 report or analysis output was modified. No experiment, retrieval, feature extraction, or pipeline was run.