# CT2.1 — Controlled feature ablation report

**Run ID:** `ct2_1_ablation_20261118_pending`  
**Status:** **BLOCKED — no confirmatory experiment performed**  
**Date:** 2026-11-18

## 1. Objective

CT2.1 is intended to compare Full (13 frozen features), Query-only (4 frozen query features), and Pool-only (9 frozen candidate-pool features) for adaptive reranking budget allocation. The primary contrast is Full versus Query-only; Full versus Pool-only and Pool-only versus Query-only are secondary. Gate C is a separate Current-versus-cost-matched-null policy comparison and is not a direct feature ablation.

## 2. Existing evidence reused

This package inherits, without repeating, the completed feature, split, M*, selector, and existing-artifact audits. They establish 809 development query rows, the frozen feature definitions/order, no feature-label leakage in the intended selector inputs, `tau=.99` M* labels, and the absence of historical Query-only/Pool-only controlled-ablation artifacts. The historical binary Easy/Hard feasibility result and historical final TEST evaluation are not CT2.1 evidence.

## 3. Population and feature configurations

The intended population is SciFact TRAIN=647 plus VALIDATION=162 (`N=809`); TEST is excluded. The candidate pool is 100; budgets are `{10,20,30,50,100}`; M* is a target only; selection remains RRF Prefix-K.

- **Full:** the nine pool features plus `query_length`, `num_numeric_tokens`, `avg_idf`, and `max_idf`.
- **Pool-only:** frozen features 1–9.
- **Query-only:** frozen features 10–13.

`entity_count` is excluded. Qrels, relevance, M*, Hard, query IDs, and post-reranking outcomes are not selector features.

## 4. CV, training, and OOF methodology

The draft design specifies repeated stratified 5-fold CV × 10 repeats (seeds 0–9), three strata `{10}`, `{20,30}`, `{50,100}`, inner stratified 3-fold CV, and `C` grid `{.01,.1,1,10}`. All configurations must use identical folds and preprocessing fit only on each training fold.

No folds have been constructed, no model has been fit, and no OOF prediction exists. The existing `run_pooladapt.py` official-TRAIN/VALIDATION runner and `run_multivariate_feasibility.py` binary feasibility workflow are not compatible substitutes for this registered CT2.1 design.

## 5. Approved decision records

**None.** The CT2 protocol is draft and essential choices remain pending group approval: inner objective/aggregation/C tie-break; repeat OOF aggregation; absent-class handling; calibration, temperature, lambda and budget matching; null policy and targets; inference, CI, multiplicity and metric hierarchy; and Gate-C interpretation.

`configuration_pending_approval.json` records concrete proposals, rationale, and impact for each area. Those proposals are **not approval** and must not be implemented or changed based on observed CT2.1 outputs.

## 6. Quality, cost, contrasts, and Gate C

No quality metrics, allocation/cost metrics, per-class outcomes, comparisons, calibration, Gate-C null outputs, or statistical results have been generated. It would be invalid to report the historical Full-feature validation or TEST result as any of these outputs.

When and only when approved, the quality endpoints are nDCG@10, Recall@10, and MRR@10; cost includes predicted/actual K, average K, average pairs, compression, and registered latency. The primary and secondary contrasts must remain separately labeled. Gate C, if approved, must use its distinct registered Current-versus-cost-matched-null procedure, not the direct ablation result.

## 7. Limitations

- Severe target imbalance includes only one TRAIN `K=100` label; all class-support and absent-class behavior must be retained and reported.
- M* is outcome-derived and does not establish deployable controller performance.
- CT2.1 cannot alone establish causal contribution beyond its registered controlled comparison, nor automatically authorize CT2.4.
- TEST must not affect design, folds, selection, calibration, matching, or interpretation.

## 8. Reproducibility and deviations

This new non-overwriting directory pins intended development inputs and code references in `input_manifest.json`. `output_manifest.json` explicitly lists the confirmatory artifacts intentionally absent. The repository commit, protocol digest, and execution stop are recorded in `reproducibility_manifest.json`.

**Deviation from requested execution:** confirmatory CT2.1 work was stopped before any stage-specific validation or experiment operation because the essential procedure is unapproved and the required controlled runner does not yet exist.

## 9. Conclusion

CT2.1 is **not complete and not ready to execute**. The group must freeze the protocol and approve the recorded methodological choices. Then implement and verify a development-only, non-overwriting nested repeated-CV runner with fold manifests, class-aware OOF outputs, and a separate approved Gate-C workflow. CT2.2 cannot start until the approved Full-model aggregate OOF dependency exists.
