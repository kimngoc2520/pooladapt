# CT2.0 — Oracle headroom / Gate A report

**Run ID:** `ct2_0_gate_a_20261118_pending`  
**Status:** **BLOCKED — no confirmatory analysis performed**  
**Date:** 2026-11-18

## 1. Objective

Evaluate, on the registered SciFact development population only, whether the outcome-derived Oracle M* allocation has sufficient nDCG@10 headroom over Fixed RRF Prefix-10 under Gate A. This stage is restricted to the Oracle-M* versus Fixed-10 comparison; it does not evaluate feature contribution, the PoolAdapt controller, controller diagnosis, or CT2.4 activation.

## 2. Population

The intended population is SciFact development: 647 TRAIN queries plus 162 VALIDATION queries (`N=809`). TEST is excluded from this package and must not be used for input selection, method selection, calibration, analysis, or interpretation.

The registered threshold is `tau=0.99`. For each query, M* is the smallest budget in `{10,20,30,50,100}` whose prefix nDCG@10 reaches at least 0.99 of full-rerank nDCG@10, with 100 used when no candidate budget qualifies. It is an optimistic, outcome-derived reference, not a deployable controller.

## 3. Reused sources and evidence

- `CT2_Experiment_Protocol.md`, Sections 2.1–2.2, 4–5, 11–12, and 14–15.
- `03_oracle_mstar_audit.md`: M* audit is inherited as COMPLETE; it is not recomputed.
- `02_dataset_split_and_provenance_audit.md`: the development split/provenance finding is inherited; TEST caveats remain in force.
- `member_A_readiness_report.md`: inputs were found available, but CT2.0 was approval-blocked.
- Existing development M*, qrels, split, ranking, and metric artifacts are pinned in `input_manifest.json`.

The input manifest records the current repository commit and SHA-256 values to permit a future approved execution to verify the exact inputs. It is not a claim that the required execution-time alignment validation has been completed.

## 4. Registered comparison and method

The intended query-level paired difference is:

`d_q = nDCG@10(M*(q), q) − nDCG@10(10, q)`.

The aggregate estimand is `Delta_A = mean(d_q)`. The Gate A threshold is `delta_A=0.005`. The existing `src/evaluation/retrieval_metrics.py` implementation remains the metric implementation: linear-gain nDCG@10 and zero when IDCG is zero. A future run must validate exact query-ID alignment among split IDs, qrels, rankings, Fixed-10 outcomes, and M* before calculating any `d_q`.

## 5. Approved statistical procedure

**None.** The current CT2 protocol is a draft, explicitly marks the Gate-A CI method and resampling seed/count as pending group approval, and requires these to be registered before analysis. Therefore this report does not designate the proposals as approved procedure.

`configuration_pending_approval.json` provides a proposed procedure for group consideration:

- two-sided 95% percentile nonparametric paired-bootstrap CI for `mean(d_q)`;
- resample complete query-level pairs;
- 10,000 resamples with seed `20261118`;
- preserve metric-defined zero-IDCG and zero-difference observations; fail on any alignment/population discrepancy; and
- no additional unregistered significance test.

The proposal preserves the pairing and targets the stated mean, but it changes the actual Gate-A CI bounds if a different valid inference procedure is approved. It must not be selected or revised after examining CT2.0 results.

## 6. Results and CI

No query-level paired result file, descriptive summary, mean difference, distribution, confidence interval, or Gate-A result has been generated. Producing them would be confirmatory analysis before the essential statistical procedure and protocol freeze are approved.

## 7. Gate A interpretation

No Gate A decision is made. Once an approved procedure is executed, the only registered interpretation is:

- lower 95% CI bound greater than `0.005`: evidence supports headroom above the threshold;
- upper 95% CI bound below `0.005`: evidence supports headroom below the threshold;
- CI containing `0.005`: inconclusive.

Non-significance must not be described as equivalence.

## 8. Limitations

- M* uses reranking outcomes and is an optimistic oracle, not an implementable policy.
- Gate A cannot establish a causal contribution of pool features, effectiveness of the current PoolAdapt controller, superiority of that controller over Fixed Prefix-K, a deployment advantage, or permission to activate CT2.4.
- Historical TEST provenance limitations remain documented by the inherited audits, but TEST was not accessed or used here.
- The older general experiment protocol has unresolved/further-draft settings that conflict with CT2 values; the approved CT2 version must govern this stage.

## 9. Reproducibility and deviations

This package is a new, non-overwriting pre-execution run directory. `input_manifest.json` pins intended inputs; `output_manifest.json` records intentionally absent confirmatory files; and `reproducibility_manifest.json` records commit, protocol digest, status, and required approvals.

**Deviation from requested execution:** the paired analysis was deliberately not started. This is required by the CT2 protocol and task instruction because CI/resampling decisions and protocol freeze are essential and remain pending.

## 10. Conclusion

CT2.0 is **not complete and not ready to execute**. The next action is group approval/freeze of the Gate-A statistical procedure (including `delta_A` confirmation), recorded in a protocol amendment/version with approver, date, SHA-256, and run mapping. Only then may the stage-specific alignment validation and paired analysis run under a new approved run ID.
