# Final Phase 2 evaluation

`experiments/run_phase2_evaluation.py` is the single owner of final cross-method
calibration. It does not train classifiers. Supply trusted local pickled fitted
models from the existing development recipes. The evaluator verifies checkpoint
integrity, exact effective recipes, feature order, grid, and 647-query training
population before calibration. It audits agreement with historical validation
budgets/sanity without changing a frozen checkpoint to imitate older outputs.
The final protocol explicitly names the persisted checkpoints as authoritative.
It locks the relevant Python/package versions.

`python experiments/run_phase2_evaluation.py preflight` checks recorded split
populations and gates, hashes the inspected Phase 2 artifacts, and records
`PENDING_CHECKPOINTS` in the summary. It produces no calibration lock or TEST
metrics. The three final result CSVs are created only by their actual stages;
header-only files are not presented as completed experiments.

The original development runners fitted their classifiers in memory and discarded
them. Checkpoint persistence now uses their exact extracted `build_model` recipes.
`python experiments/persist_phase2_checkpoints.py` reconstructs each once on the
recorded 647 TRAIN IDs, preserving sorted string-ID order, and writes
`results/phase2/checkpoints/{sage_slo,pooladapt}.pkl` with JSON metadata. It refuses
to refit if any checkpoint or run manifest already exists. It neither calibrates
nor opens validation/TEST artifacts. Each complete sklearn pipeline includes its
fitted preprocessing; deterministic pickle protocol 5 preserves its learned state.
Fresh reloads must have identical classes, predictions, and probabilities on a
16-query TRAIN-only sample. Sidecars record effective defaults, exact feature order,
targets, versions, provenance hashes, and the verification result.

Both development runners also expose mutually exclusive `--save-checkpoint` and
`--load-checkpoint` options. With neither option, their fitting behavior remains
unchanged. Loading verifies model identity, population/order, seed, feature order,
grid, checksum, effective recipe, and runtime versions. The final evaluator uses
the same validated loader and defaults to these persisted checkpoint paths.
No calibration or final TEST execution is implied by checkpoint creation.

The split lock files were also absent in this checkout. The evaluator uses the
existing `prepare_split` function to reconstruct the documented sorted-ID,
seed-42 80/20 split and verify it against the recorded development population.
The artifacts contain 647 TRAIN and 162 VALIDATION queries, disjoint from the
300 TEST queries. Existing development and Phase 1 outputs are read-only.

## Validation stage

```powershell
python experiments/run_phase2_evaluation.py calibrate --pooladapt-model <checkpoint.pkl> --sage-model <checkpoint.pkl>
```

Both adaptive methods expose exactly `z/T + lambda*g(k)` over
`{10,20,30,50,100}`. The sanity check uses lambda `{-5,0,5}` at T=1. Identical
average K at all three settings stops that method and records FAILED points.
For a responsive method, pairwise logit-line intersections partition the entire
real lambda axis into constant hard-argmax policies. The sweep evaluates those
boundaries, interval representatives, and both exterior intervals, retaining the
configuration closest to each target `{10,20,30,50}`. Workload ties choose lambda
closest to zero, then the smaller lambda. No quality metric participates in this
choice. Validation nDCG is computed only after the workload choice.

Positive T preserves argmax after multiplying scores by T:
`argmax(z/T + lambda*g) = argmax(z + T*lambda*g)`. Consequently a two-dimensional
T/lambda search cannot produce a policy missing from this exhaustive lambda
sweep. T remains 1. Ties among budget scores use the smaller budget.

Exact-budget methods use K=B; Full Rerank uses K=100. The frozen heuristic state
is G2 FAILED, G3a/G3b NOT_EVALUATED. Multivariate feasibility does not replace G2.
A different frozen heuristic state causes a clear protocol error requiring
inspection of its existing policy, rather than an invented implementation.

`results/phase2/07_evaluation/locked_configurations.json` is created once with
model, data, source, split, configuration, and Phase 1 hashes. It cannot be
overwritten by calibration. The summary records CALIBRATED_TEST_PENDING.

## Final TEST stage

```powershell
python experiments/run_phase2_evaluation.py test
```

This stage verifies the lock and dataset identity before model loading. It
reuses every locked parameter. Each query has one shared 100-document hybrid
pool; PACE-EF permutes this pool and selects exact Prefix-K. Random selects exact
K using the existing query-dependent sampler with seeds 0,1,2,3,4. PoolAdapt and
SAGE select the existing RRF prefix at their predicted K. Failed calibration
points have no TEST calls or fabricated metrics.

The Cross-Encoder is warmed up before any recorded call. Reranking seconds use
the existing timer, covering preparation, inference, and sorting. Loading,
warm-up, retrieval, selection, and progress printing are outside that timer.
TEST qrels are loaded for scoring only after all selection and reranking finish.
Selectors receive no labels. Prediction JSONL preserves input pool, reordered
pool, selected IDs, output ranking, K, and measured latency for each call.

Outputs include the three required CSVs, `random_seed_results.csv`,
`test_query_metrics.csv`, `statistical_analysis.csv`, `test_predictions.jsonl`,
and `results/phase2/phase2_summary.json`. Empty fields mean not applicable or not
evaluated. Failed-point rows retain achieved validation K in
`validation_average_budget`, with TEST metrics and TEST average K empty.

Compression ratio is `1 - avg(K)/100`. Latency reduction is
`1 - mean_latency/Full_mean_latency`. Random aggregate quality reports mean and
sample SD across five seeds, not SD across queries; aggregate latency percentiles
are means of the five seed-level percentiles. Query and seed data are preserved.
Adaptive TEST workload can drift outside tolerance despite successful validation
calibration. Such points remain reported with `matched_budget=false` and their
original calibration status; they are excluded from matched-budget tests. No
TEST retuning occurs.

## Predeclared statistical comparisons

The repository's paired, two-sided Wilcoxon signed-rank test excludes zero
differences. A global Holm family spans all three metrics and all eligible
predeclared comparisons: each applicable method versus Full Rerank; PACE-EF
versus Fixed Prefix; each adaptive method versus Fixed Prefix, PACE-EF, and
Random; and PoolAdapt versus SAGE-SLO. Budget-matched hypotheses require both
TEST workloads within tolerance. Every comparison verifies identical query IDs.
For Random, each query's metric is averaged over the five fixed seeds before
the paired test, while seed-level variability remains separately reported.
All-zero differences have p=1. A corrected p>=0.05 means no statistically
significant difference detected; it is not evidence of equivalence.

Results are described as Quality–Cost–Latency Trade-off Analysis. No Pareto
frontier or superiority claim is inferred without final empirical evidence.

## Verification

Run the entire repository suite with `python -m pytest`. The final evaluator's
tests cover validation-only IDs, calibration response and failure, ties,
temperature/lambda equivalence, exact budgets, seed reproducibility, label
rejection, stopped heuristic gates, aggregation, output schemas, paired tests,
immutable configuration reuse, and Phase 1 integrity checks.
