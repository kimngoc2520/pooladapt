# CT2.2 — Controller diagnosis report

**Run ID:** `ct2_2_controller_diagnosis_20261118_pending`  
**Status:** **BLOCKED — no confirmatory diagnosis performed**  
**Date:** 2026-11-18

## 1. Objective

CT2.2 is intended to diagnose allocation behavior of the current Full-feature controller using an approved development OOF prediction for each query. Its diagnostic scope includes allocation error, under/over-allocation, Hard-query detection, and quality-cost consequences. It does not redesign the controller, perform feature selection, or activate CT2.4.

## 2. Dependency and provenance

The mandatory input is one approved, leakage-safe, query-level aggregated OOF record from the Full CT2.1 controlled ablation for every development query. That artifact must contain class-label-aligned probabilities, predicted K, quality/cost information, and fold/repeat provenance.

It does not exist. The A2 package explicitly records that no folds, Full model, or OOF outputs were generated because CT2.1 is approval- and implementation-blocked. No file matching a CT2.1 OOF, controller-diagnosis, allocation-confusion, or hard precision/recall artifact was found. Historical TEST outputs, historical validation predictions, and the binary Easy/Hard feasibility fold summaries are prohibited substitutes.

## 3. Existing evidence reused

The M* audit is inherited as COMPLETE. On development at `tau=.99`, M* has 809 labels: 757 Easy and 52 Hard. The selector audit establishes a five-class Logistic Regression behavior but does not produce CT2.1 OOF evidence. The existing-artifact audit confirms hard precision/recall and allocation confusion matrices are missing, while the feasibility artifact is binary and not a five-class allocation diagnosis.

## 4. Population and Hard definition

The intended population is SciFact TRAIN+VALIDATION (`N=809`) only. TEST is excluded. Hard is `1[M*(q)>10]`; its inherited prevalence is `52/809` (about 6.43%).

Given an approved class-label-aligned OOF vector, the Hard score is `P(20)+P(30)+P(50)+P(100)`. It must never assume fixed probability-column positions, treat repeated folds as independent queries, or fabricate probabilities for an absent class.

## 5. Approved OOF and diagnosis procedure

**None.** The upstream Full OOF artifact is missing, and the protocol leaves OOF aggregation, absent-class handling, hard-score aggregation, the discrete Hard decision rule, ordinal error definition, and uncertainty/inference pending group approval.

`configuration_pending_approval.json` supplies proposals and their impacts. In particular, it proposes inheriting the approved CT2.1 aggregate unchanged; using AP on one aggregate score/query vector; and classifying predicted Hard via aggregate `K_pred>10` without threshold tuning. These are not approved procedures and have not been executed.

## 6. Allocation errors, Hard detection, and quality-cost results

No allocation-error records, Hard metrics, confusion matrix, query-level quality/cost records, or error-pattern results have been generated. Consequently, there is no evidence-supported finding about underallocation, overallocation, controller failure patterns, quality loss, additional cost, or latency.

After approval and input validation, the required allocation records are signed and absolute numeric-K error, approved ordinal error, and under/exact/over-allocation indicators. Hard reporting must include prevalence, AP, and—only under an approved discrete rule—precision, recall, F1, confusion matrix, per-class support/macro metrics, and rare-class uncertainty. Quality-cost records must distinguish quality, allocation, and cost rather than present them as policy-comparison inference.

## 7. Limitations

- M* is outcome-derived and optimistic; diagnosis does not establish deployment performance or causal contribution of features.
- The 52 Hard queries and rare M* classes require explicit support counts and cautious uncertainty interpretation.
- CT2.2 does not substitute for Gate A, Gate C, or Gate E, and does not authorize controller redesign or CT2.4.
- TEST was not accessed or used and cannot be used to select a Hard threshold, discover error patterns, or change the diagnosis method.

## 8. Reproducibility and deviations

This non-overwriting package records available inputs, the missing upstream contract, the current commit and protocol digest, and intentionally absent output artifacts. The OOF dependency and every approved procedural choice must be added by a future execution manifest, not backfilled into this package.

**Deviation from requested execution:** all diagnostic analysis was stopped before incremental alignment validation because its required approved CT2.1 OOF input and essential decisions are absent.

## 9. Conclusion

CT2.2 is **not complete and not ready to execute**. Complete and approve CT2.1 first; then supply its Full-model aggregate OOF artifact, freeze the diagnosis procedure, validate query-level alignment, and run CT2.2 under a new run ID. CT2.2 remains diagnostic evidence only and cannot automatically activate CT2.4.
