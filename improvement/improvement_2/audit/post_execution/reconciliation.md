# CT2.0--CT2.2 post-execution reconciliation

**Reconciled:** 2026-11-19 (artifact inspection only)  
**Execution run:** `development_execution_20261118T175713Z`  
**Population:** SciFact TRAIN+VALIDATION, 809 queries; `test_used: false`

## Scope and provenance

This record reconciles existing artifacts only. No experiment, cross-validation,
ranking, reranker inference, metric computation, or TEST-data operation was
performed during reconciliation. Original result CSVs remain under
`../../results/development_execution_20261118T175713Z/`.

The pre-execution packages were moved to `../pre_execution/` and retain their
historical pending/blocked decisions. Relocation-only path fields in three
input manifests and the draft protocol's audit index were normalized; the
protocol SHA-256 values were updated accordingly. The person-based filename
`member_A_readiness_report.md` was renamed to
`pre_execution_readiness_report_20261118.md`; its contents were not edited.
The historical draft protocol was moved to
`../../docs/CT2_Experiment_Protocol_draft_20261118.md`; this is not evidence
of a frozen or approved protocol.

## What was planned versus what occurred

| Stage | Pre-execution record | Existing execution output | Reconciled interpretation |
|---|---|---|---|
| CT2.0 | Gate A blocked pending protocol/statistical approval | Oracle metrics, query results, statistical analysis, and recovered report | Executed development-only oracle analysis; output existence does not cure the earlier approval gap. |
| CT2.1 | Controlled ablation and Gate C blocked pending decisions/implementation | Nested-CV fold rows, aggregate OOF, query-level records, comparisons, and cost-matched-null files | Executed, but not confirmed compliant with the pending pre-execution procedure. |
| CT2.2 | Dependent on approved CT2.1 OOF and diagnostic decisions | Detection metrics, confusion matrix, allocation records, failure groups, and recovered report | Executed as descriptive diagnosis; no formal approval or confirmatory inference follows. |

The execution timestamp is `20261118T175713Z`. The report-recovery timestamp is
`20261118T182311Z`; its own record states `result_computation_rerun: false`.

## Protocol-compliance finding

The sole in-package protocol source is explicitly a draft. Searches of the
repository and the supplied `improvement_2.zip` found no approved Protocol
v8.1 artifact. Therefore its exact content, hash, approver, and run mapping
could not be independently verified.

Separately, the execution manifest and `scripts/run_pooladapt_ct2_execution.py`
agree that `macro_f1` was used to select C: the script averages five-class
macro-F1 over the three inner folds and selects the best C, breaking ties by
smaller C. This confirms optimization use. The user-supplied reconciliation
brief says v8.1 requires pooled inner-OOF PR-AUC for Hard detection (with
log-loss secondary), but the source protocol file is absent. Accordingly this
is recorded as an **unresolved, likely protocol deviation**, not silently
relabelled as compliant. It affects CT2.1 model selection and downstream
CT2.2 interpretation. The results may be retained as exploratory evidence;
a dated approval/amendment and any required registered rerun decision remain
necessary before confirmatory use.

## Evidence checks

- The execution manifest lists five input SHA-256 values and all five source
  files exist with matching hashes.
- Result schemas support the reported population: query-level CT2.0 records,
  each CT2.1 configuration evaluation, and CT2.2 allocation records each have
  809 data rows; the fold result has 24,270 rows (3 configurations x 809 x 10).
- All three report source paths now resolve relative to `reports/`.
- The original result CSV SHA-256 values are recorded in
  `result_integrity_20261119.json` and were generated after the relocation;
  no result files were edited or relocated.

## Stage status

| Stage | Execution status | Protocol compliance | Approval status | Remaining issues |
|---|---|---|---|---|
| CT2.0 | Executed; outputs recovered | Unverified; pre-execution Gate-A procedure was pending | Not approved | Frozen protocol/statistical procedure and approval/run mapping absent. |
| CT2.1 | Executed; outputs recovered | Unverified; macro-F1 selection conflicts with the brief's claimed v8.1 objective | Not approved | Approved v8.1 unavailable; objective deviation and other pending CV/Gate-C decisions require disposition. |
| CT2.2 | Executed descriptively; outputs recovered | Unverified; depends on CT2.1 and unapproved diagnosis rules | Not approved | Upstream protocol disposition, approved OOF/Hard decision rules, and formal inference status unresolved. |

