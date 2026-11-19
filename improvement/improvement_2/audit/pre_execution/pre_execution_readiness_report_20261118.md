# A0 — Member A CT2 preparation and readiness report

**Date:** 2026-11-18  
**Scope:** Readiness only for A1 / CT2.0 (Gate A), A2 / CT2.1, and A3 / CT2.2. This report inherits the completed audits; it does not repeat their repository-wide checks, recompute results, or use TEST.

## 1. Objective

Establish the concrete inputs, code/artifact dependencies, unresolved methodological decisions, and stage-specific readiness needed before Member A can run CT2.0, CT2.1, or CT2.2. A stage is not considered runnable merely because historical Phase 2 artifacts exist.

## 2. Sources read

- Coordination protocol: `CT2_Experiment_Protocol.md` (draft, 2026-11-18).
- Audit baseline: `README.md`, `01_feature_extraction_audit.md`, `02_dataset_split_and_provenance_audit.md`, `03_oracle_mstar_audit.md`, `04_selector_implementation_audit.md`, `05_existing_experimental_artifacts.md`, `06_research_gaps_and_recommendations.md`, and `audit_summary.json`.
- Detailed research/method documents: `README.md`, `docs/experiment_protocol.md`, `docs/oracle_mstar_methodology.md`, `docs/feature_mstar_methodology.md`, and `docs/research_notes.md`.
- Stage-relevant implementation/artifacts: `experiments/run_pooladapt.py`, `experiments/run_multivariate_feasibility.py`, `experiments/run_phase2_evaluation.py`, `src/evaluation/phase2.py`, `src/evaluation/phase2_execution.py`, `src/analysis/oracle_mstar.py`, and the referenced `results/phase2/` inputs.

## 3. Inherited audit baseline

The following findings are treated as established evidence rather than re-audited here:

- SciFact split is TRAIN=647, VALIDATION=162, TEST=300, with no split overlap; development is 809 queries.
- Development feature extraction is complete for the frozen 13-feature set; historical TEST feature provenance remains only partial/not verified.
- Oracle M* is complete on development for `tau=0.99`; the reported population is 757 Easy (`M*=10`) and 52 Hard (`M*>10`) queries.
- The current selector is a 5-class, balanced Logistic Regression with median imputation and scaling. The extreme class imbalance, including one TRAIN `K=100` row, remains a material limitation.
- Existing artifacts include the full-feature selector, M* labels, feature analyses, and historical TEST evaluation artifacts. They do **not** provide a controlled query-only/pool-only ablation, CT2 allocation confusion matrix, or CT2 hard precision/recall.
- TEST historical provenance has unresolved limitations (including the previously reported two FAIL and three NOT VERIFIED checks). Reconstructed TEST features are not historical-provenance proof and are out of scope for A1–A3.

## 4. Governing context and document conflict

The CT2 protocol governs this readiness assessment: SciFact development only, candidate pool `N=100`, budgets `{10,20,30,50,100}`, primary `tau=0.99`, and frozen features 1–13 (pool 1–9; query 10–13). M* is an outcome-derived label, never a selector input. All A1–A3 work must exclude TEST.

There is a material document conflict that requires explicit resolution by using the CT2 protocol as the controlling document for CT2 runs:

- `docs/experiment_protocol.md` is an earlier initialization document. It still says the split, calibration, seeds, and exact protocol are “TO BE FINALIZED”; its illustrative split is 70/10/20 and its planned budget grid includes 70.
- `CT2_Experiment_Protocol.md` specifies the current CT2 population and grid above, but is itself marked draft and awaiting group review/freeze.

No code or run should silently blend the older document's split/budgets with CT2. The approved CT2 version/hash must be recorded before execution.

## 5. Dependencies by Member A stage

### A1 — CT2.0 Oracle headroom / Gate A

| Dependency | Required state | Evidence / status |
|---|---|---|
| Development population and qrels | Exact 809 development query population, with aligned qrels and no TEST | Inherited split audit COMPLETE; development reranking inputs are present under `results/phase2/development_reranking/{train,validation}/reranking_results.json`. |
| Prefix and full rankings | RRF prefix budgets `{10,20,30,50,100}` plus full rerank, candidate pool 100 | Development ranking JSON metadata declares SciFact, its split, `candidate_pool_size=100`, and the reranking methods. M* audit is COMPLETE. |
| Oracle labels | `mstar_labels_tau099.csv`, one valid label/query on development | Present and compatible (incremental check below). |
| Gate-A computation | Per-query `d_q = nDCG@10(M*(q))-nDCG@10(10,q)`, summary/distribution and 95% paired CI | M* implementation and labels exist; no dedicated CT2.0 Gate-A reporting runner/manifest was found. |
| Registered decision rule | Confirmed `delta_A=0.005`, CI method/statistic, resampling count and seed | Pending group approval; protocol forbids starting Gate A before this registration. |
| Reproducibility envelope | Protocol version/hash, commit, inputs/hashes, command, environment, output manifest | Historical Phase 2 manifests do not constitute a new CT2.0 run manifest; a new non-overwriting run convention is needed. |

### A2 — CT2.1 controlled feature ablation / Gate C

| Dependency | Required state | Evidence / status |
|---|---|---|
| Development features and labels | 809 aligned feature rows and M* labels; Full=13, Pool-only=9, Query-only=4 | Present; schemas and coverage incrementally verified. `entity_count` is present in the raw query table but excluded by the frozen 13-feature CT2 set. |
| Ranking/metric inputs | Prefix/full development outcomes and qrels, all derived without TEST | Present for development; inherited M* and split audits support their intended population. |
| Controlled nested repeated CV | Repeated stratified 5-fold × 10 repeats (seeds 0–9), inner stratified 3-fold, leakage-safe fold preprocessing | **Missing implementation.** `run_pooladapt.py` fits the official TRAIN and evaluates VALIDATION; it is not this controlled CV workflow. |
| Model selection rule | Inner objective, aggregation across inner folds, C selection and tie-break for `C={.01,.1,1,10}` | Pending group approval. |
| Calibration | Approved population, probability/logit aggregation, T treatment, shared lambda policy/search/tie-break, absent-class treatment, targets/tolerance and unavailable-target rule | Pending group approval. Current `phase2.py` is validation-only calibration with 5% relative tolerance and targets `{10,20,30,50}`; it is not automatically the CT2 controlled-CV procedure. |
| Gate C null | Budget-distribution-preserving, query-feature-budget-link-breaking null; randomization/expectation choice, seeds/repeats, inference and matching policy | Missing and pending verification/group approval. Proposed CT2 targets `{11,12,14,17,20}` and absolute `±0.1` tolerance are not approved. |
| Inference/reporting | Registered paired CI/test, multiplicity correction, alpha, metric hierarchy; per-fold/OOF predictions, actual/predicted K, quality/cost | Pending group approval; CT2 output schema and run ID convention are not implemented. |
| Reproducibility | New immutable per-run manifests and no overwrite of historical artifacts | Required by protocol; not yet established for CT2.1. |

### A3 — CT2.2 controller diagnosis

| Dependency | Required state | Evidence / status |
|---|---|---|
| Full-model CT2.1 OOF | One approved, leakage-safe aggregated OOF prediction vector/query from the Full ablation | **Missing; produced only after A2.** |
| Labels and allocation outcomes | Development `M*`, Hard definition (`M*>10`), predicted K, per-K prefix quality/cost | M* labels and development inputs exist; OOF predicted K and controlled-CV allocation outcomes do not. |
| Hard score | `P(Hard)=P(20)+P(30)+P(50)+P(100)`, aligned by class label | Existing binary feasibility code is not a 5-class controller diagnostic. The required five-class mapping must operate on the approved OOF representation. |
| OOF aggregation / AP | Query-level aggregation of repeated OOF scores, then `average_precision_score` with Hard=1; prevalence reported as reference | Pending group approval. Fold rows must not be treated as independent queries. |
| Discrete diagnosis rule | Rule for Hard precision/recall/F1 and allocation confusion matrix | Pending group approval. |
| Diagnosis outputs | Per-class/macro/support, confusion matrix, signed/absolute/ordinal K error, under/over-allocation, quality loss/cost, rare-class uncertainty | Missing; no CT2.2 diagnostics runner/output schema found. |

## 6. Incremental verification performed

These checks were limited to dependencies not established as runnable by the baseline audits. No rankings were recomputed and no TEST artifact was opened or used.

| Check | Inherited evidence / reason for incremental check | Result | Evidence |
|---|---|---|---|
| Development IDs and M* usability | Split/M* audits are COMPLETE, but A1/A2 need exact row coverage | PASS: 647 TRAIN + 162 VALIDATION = 809 unique, no train/validation overlap. `mstar_labels_tau099.csv` has 809 unique rows, none missing from development; schema includes `query_id`, `M_star`, `tau`, and prefix/full nDCG fields. | Split files and `results/phase2/02_oracle/mstar_labels_tau099.csv`; read-only count/schema check on 2026-11-18. |
| Feature matrix availability for the three ablations | Feature audit is COMPLETE, but A2 needs stage-specific Full/Pool/Query matrices | PASS, with implementation gap: `pool_features.csv` has 809 unique query rows and nine frozen pool columns; `query_features.csv` has 809 unique rows and the needed four frozen query columns (`query_length`, `num_numeric_tokens`, `avg_idf`, `max_idf`). The raw extra `entity_count` is excluded as specified. | `results/phase2/01_candidate_logging/{pool_features.csv,query_features.csv}`; read-only count/schema check. |
| Development ranking input availability/schema | A1 requires per-query prefix/full outcomes, not only labels | PASS for availability: train and validation `reranking_results.json` files exist and declare their split, SciFact, candidate pool 100, and reranking metadata. M* audit remains the authoritative correctness evidence; this check does not recreate it. | `results/phase2/development_reranking/{train,validation}/reranking_results.json`. |
| Qrels input naming and availability | Avoid a stage failure from schema mismatch | PASS with code-awareness: `train.tsv` and `validation.tsv` exist and use `query-id`, `corpus-id`, `score` headers. CT2 code must use the repository qrels loader/header normalization rather than assume `query_id`. ID alignment itself remains inherited from audit 02. | `data/scifact/qrels/{train,validation}.tsv`; header inspection. |
| CT2.1 controlled-CV executable | Existing selector audit does not establish a nested controlled-ablation runner | FAIL / MISSING: `run_pooladapt.py` takes official train IDs, validation IDs, a single output path, and reports validation-only sanity calibration. It has no repeated nested CV, three feature-set orchestration, OOF output, or Gate-C null. | `experiments/run_pooladapt.py` CLI and implementation inspection. |
| CT2.2 OOF/diagnosis executable | Existing feasibility results are explicitly partial scope | FAIL / MISSING: `run_multivariate_feasibility.py` uses binary Easy/Hard repeated-fold AP/ROC summaries and writes no aggregated query-level 5-class OOF vector, allocation confusion matrix, or Hard allocation precision/recall. | `experiments/run_multivariate_feasibility.py`, especially its `RepeatedStratifiedKFold` and `average_precision_score` use. |
| Existing manifest/output conventions | CT2 protocol requires per-run immutable manifests and no overwrite | PARTIAL: `run_phase2_evaluation.py` and `phase2_execution.py` demonstrate manifests and isolated output handling for historical locked TEST work. They do not provide an A1/A2/A3 development-only run schema. The historical source manifest is at `results/phase2/evaluation/reviewed_source_manifest.json`, not the phase2 root. | `experiments/run_phase2_evaluation.py`; `src/evaluation/phase2_execution.py`; `results/phase2/evaluation/`. |

## 7. Blockers

1. **Protocol is not frozen.** `CT2_Experiment_Protocol.md` is draft and explicitly requires group review/approval, version, hash, commit reference, and run mapping.
2. **A1 decision blocker:** Gate-A CI method, statistic/resampling count/seed, and confirmation of `delta_A=0.005` are pending group approval.
3. **A2 design blocker:** inner objective, C selection/tie-break, repeated-OOF probability/logit aggregation, calibration design, absent-class handling, target/tolerance behavior, and inference plan are unapproved.
4. **A2 implementation blocker:** no controlled nested repeated-CV ablation runner or CT2 run/manifest schema exists. This work must be designed only after the preceding decisions are approved.
5. **Gate-C blocker:** cost-matched-null construction, randomization versus expectation, seeds/repeats, matching targets/tolerance, unavailable target handling, and inference are unapproved/unimplemented.
6. **A3 dependency blocker:** approved Full CT2.1 OOF predictions do not exist; the current binary feasibility artifact cannot substitute for them.
7. **A3 decision/implementation blocker:** discrete diagnostic rule, repeated-OOF aggregation, and 5-class diagnosis outputs are pending/missing.
8. **Historical TEST provenance caveat:** not a prerequisite to development-only A1–A3, but it blocks any claim that a later TEST run has fully verified historical feature provenance. TEST must remain isolated throughout preparation and diagnosis.

## 8. Pending methodological decisions (do not infer approval)

| Area | Protocol-fixed or existing behavior | Pending decision / status |
|---|---|---|
| CT2 constants | SciFact development=809, N=100, budgets `{10,20,30,50,100}`, `tau=.99`, Hard=`M*>10`, feature groups 13/9/4, CV family and C grid are specified | Group freeze/approval and protocol hash still pending. |
| Gate A | Proposed `delta_A=.005`; per-query paired difference and 95% CI reporting specified | CI statistic/method, resampling count/seed, and historical approval of delta are pending group approval. |
| CT2.1 inner CV | Repeated stratified outer 5×10 and inner 3-fold are protocol-specified | Inner objective, fold-score aggregation, C choice and tie-break are pending group approval. |
| Calibration | Existing code: validation-only, fixed T default 1, shared lambda, policy-region sweep, clamp, targets 10/20/30/50 and 5% relative tolerance | CT2 population and OOF aggregation, T, lambda search/tie-break, shared-lambda choice, absent class mask/clamp, targets/tolerance, and unavailable-target treatment are pending group approval. Existing behavior is not CT2 approval. |
| Gate C | Current vs cost-matched null only; proposed targets 11/12/14/17/20 and absolute ±0.1 are documented as proposed | Null construction, expected versus permutation form, seeds/repeats, cost aggregation, CI/test, and Gate-C decision rule are pending verification/group approval. |
| CT2.2 | Hard positive and label-based five-class-to-Hard probability mapping are specified; AP must be sklearn AP, not trapezoidal PR-AUC | Repeated OOF aggregation; whether/how AP applies after it; absent-class handling; discrete hard/allocation decision rule; inference/uncertainty are pending group approval. |

## 9. Stage-specific readiness

| Stage | Readiness | Rationale |
|---|---|---|
| A1 / CT2.0 Gate A | **NOT READY — inputs prepared; approval-blocked** | Development rankings, qrels, and M* labels are available, but the draft protocol has not frozen Gate-A CI/resampling and threshold decisions. No registered CT2.0 output manifest exists. |
| A2 / CT2.1 controlled ablation | **NOT READY — approval- and implementation-blocked** | Development input coverage is sufficient, but essential CV/calibration/Gate-C decisions are pending and the required controlled nested-CV/OOF/null workflow does not exist. Historical full-feature results do not satisfy the stage. |
| A3 / CT2.2 controller diagnosis | **NOT READY — upstream- and approval-blocked** | A3 requires the approved Full CT2.1 aggregated OOF artifact. It is absent, and diagnosis aggregation/discrete-decision choices plus a 5-class diagnosis implementation remain unresolved. |

## 10. Next action

Hold a group protocol-freeze decision before any confirmatory run. The amendment should resolve, at minimum: Gate-A CI/resampling; CT2.1 inner objective/C tie-break; repeated OOF and calibration/absent-class rules; Gate-C null and matching/inference; and CT2.2 OOF aggregation plus discrete diagnosis rule. Record approver, date, protocol version/SHA-256, repository commit, and affected run IDs.

After approval, implement a development-only, non-overwriting CT2 runner with an explicit schema/versioned manifest and run IDs. Execute in dependency order: A1; then the approved A2 Full/Query-only/Pool-only workflow and Gate C; then A3 from the resulting Full OOF. Do not access TEST for preparation, selection, calibration, Gates A/C/E, or controller diagnosis.
