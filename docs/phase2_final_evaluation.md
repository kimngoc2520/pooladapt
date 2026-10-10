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

`results/phase2/evaluation/locked_configurations.json` is created once with
model, data, source, split, configuration, and Phase 1 hashes. It cannot be
overwritten by calibration. The summary records CALIBRATED_TEST_PENDING.

### Provenance amendment

The original lock remains unchanged historical provenance. Its SHA-256 is
`fe1a76073021fd913148c2429eedf37e322220de99febf1a05ea1e828a89048c`; the
frozen calibration remains pinned by
`abdd419d36854f9b70608282b1172caae21424dd1e3bf9c4798b6f3595389738`.
[`provenance_amendment.json`](../results/phase2/evaluation/provenance_amendment.json)
separates the experimental-decision freeze from execution provenance. The
reviewed execution source is pinned to the Stage A implementation commit
recorded in the amendment, with every Python source
dependency under `src/` and `experiments/` recorded by exact Git blob bytes in
[`reviewed_source_manifest.json`](../results/phase2/evaluation/reviewed_source_manifest.json).

The exact historical frozen `src/reranking/cross_encoder.py` bytes were not
recoverable, so byte identity is not claimed. The review found no intentional
scientific behavior change: **The reviewed GPU implementation changes
execution infrastructure but does not change the frozen experimental decision
state or evaluation semantics.** CPU and CUDA floating-point values, including
near-tie rankings, are not claimed byte-identical.

## Final TEST stage

The existing GPU TEST is complete. Frozen inputs live in
`results/phase2/evaluation/`; completed GPU outputs and their COMPLETE summary
live in `results/phase2/evaluation/final_gpu/`. The root
`results/phase2/phase2_summary.json` was intentionally deleted and is not read
or recreated. Legacy preflight/calibration status, if separately run, uses
`evaluation/stage_summary.json`; this migration creates no such file. TEST
summary construction uses the lock and does not require prior stage status.

The following commands document the original workflow, not instructions to
rerun the completed experiment. Readiness preflight requires unused outputs and
will reject the populated `final_gpu/` directory. Current source also requires
the provenance migration described below before it can pass source validation.

For a separately authorized fresh run, run the amended read-only preflight first:

```powershell
python experiments/preflight_phase2_test.py --dataset-dir data/scifact --output-dir results/phase2/evaluation/final_gpu
```

It must end with `PHASE 2 PREFLIGHT: PASS`. It verifies the historical lock,
non-source artifacts, decision projection, Phase 1 artifacts, reviewed commit
and source manifest, checkpoints, and CUDA model loading without TEST
retrieval, selection, reranking, or quality evaluation.

```powershell
python experiments/run_phase2_evaluation.py test --device cuda --output-dir results/phase2/evaluation/final_gpu --fresh-output
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
and `phase2_summary.json`, all under `results/phase2/evaluation/final_gpu/`.
Empty fields mean not applicable or not
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

### Path migration provenance remains pending

The relocated reviewed manifest still pins 64 Python files at implementation
commit `415d4728e0f4cc484e3cfa0559875e743542e522`. The historical freeze commit
is `158ba388cbef7016fb2f07d7cc990d40d2e1e55f`. Four pinned scripts now differ:
`audit_phase2_final.py`, `preflight_phase2_test.py`, `render_phase2_report.py`,
and `run_phase2_evaluation.py`, all under `experiments/`.

The original manifest and amendment remain unchanged. Their old embedded paths
describe historical provenance. Current validation resolves artifacts at the
new location, requires the manifest there in HEAD, checks canonical Git source
blobs, and requires a provenance-only HEAD whose sole parent is the reviewed
implementation commit. The historical freeze contains the manifest at its old
path and changed provenance files there. Current HEAD is merge commit
`8f4741ddd5f6c429b0c3589ee9b08631181f83a8`, with two parents, and cannot itself
satisfy the freeze rule. Tracking the moved files or
editing a digest alone cannot satisfy these checks. No provenance PASS is claimed.

A separately authorized Stage A must commit and review the migrated source,
tests, documentation, tracking rules and artifact-path changes. Stage B must
then pin that Stage A commit in a new current manifest/amendment and change only
the two provenance files under `evaluation/`. No such commits or provenance
regeneration are part of this path-fix task. Before any future replacement,
preserve the original provenance bytes and Git history for the completed run;
do not rewrite its execution manifest, log, predictions or summary.

The completed GPU execution binds the original amendment and source-manifest
digests. A future current-source freeze cannot retroactively authenticate that
execution. The final audit validates current frozen inputs/source separately
from historical execution evidence. Historical provenance comes from exact Git
objects at freeze `158ba388...`, which must remain an ancestor of HEAD and have
review `415d4728...` as its sole parent. The historical freeze may change only
its original two provenance paths. Its amendment must match the unchanged lock,
calibration, decision projection and original manifest digest. Each historical
source entry must match its reviewed Git blob and source SHA256; it is never
compared to the migrated working tree. Current source retains the default
working-tree checks and requires its own strict Stage A/B pair.

The audit compares authenticated historical identities with the unchanged GPU
execution manifest, then binds the summary and every prediction row to that
manifest. Missing Git history or different historical bytes fail closed. The
amendment filesystem digest recorded by the GPU run must match its historical
Git blob bytes. Static in-memory Git-byte hashing confirmed both original
metadata digests match the GPU execution record. The project validator and
source checks were not executed; no overall provenance PASS is claimed.
The report renderer reads completed outputs and does not certify provenance.

### Exact Stage A / Stage B plan (approval required)

Stage A includes these implementation, documentation and test files:

- `.gitattributes`
- `.gitignore`
- `experiments/audit_phase2_final.py`
- `experiments/preflight_phase2_test.py`
- `experiments/render_phase2_report.py`
- `experiments/run_phase2_evaluation.py`
- `tests/test_phase2_evaluation.py`
- `tests/test_phase2_gpu_execution.py`
- `tests/test_phase2_preflight.py`
- `tests/test_phase2_provenance_amendment.py`
- `docs/phase2_dependency_audit.md`
- `docs/phase2_final_evaluation.md`
- `docs/phase2_kaggle_gpu.md`
- `docs/phase2_required_files.txt`

Stage A also records the user's existing relocation of the two unchanged
provenance files: deletion of their tracked `results/phase2/07_evaluation/`
paths and addition of their existing `results/phase2/evaluation/` paths. This
records the relocation in Git; it performs no further filesystem move. Exclude
raw outputs and unrelated untracked files, including
`docs/04_prefix10_gap_audit_report.md` and `recovered_frozen_source/`.

Stage B changes exactly these two files after Stage A has a real commit ID:

- `results/phase2/evaluation/reviewed_source_manifest.json`
- `results/phase2/evaluation/provenance_amendment.json`

The current manifest keeps schema `pooladapt.reviewed-source-manifest.v2` and
the same 64-source coverage. Set `reviewed_commit` and `tree` from Stage A; set
each `blob` and source `sha256` from exact Stage A Git objects, never checkout
line endings. Verify all 64 paths exist and none is omitted.

The new amendment uses schema `pooladapt.provenance-amendment.v2` and scope
`post-run path migration`. Its `reviewed_execution_commit` field retains its
legacy validator name but denotes current reviewed implementation, not the
completed GPU run. Set it to Stage A, set `reviewed_source_manifest_sha256` to
the new manifest's exact Git blob-byte digest, and use current artifact paths.
Retain original lock/calibration digests and decision projection unchanged.
Use status `finalized / reviewed` only after actual review. Update timestamp,
finalization explanation and review narrative for the post-run migration;
do not carry old execution review claims forward as if they reviewed Stage A.

Add `historical_execution` containing:

- `provenance_freeze_commit`: `158ba388cbef7016fb2f07d7cc990d40d2e1e55f`
- `reviewed_execution_commit`: `415d4728e0f4cc484e3cfa0559875e743542e522`
- `reviewed_source_manifest_sha256`: original manifest Git blob-byte digest,
  recorded as `c74244c2fa1f38f1ad93cf1b3f61fad77215d8988e5dd8570a667a9f4684a15e`
- `provenance_amendment_sha256`: original amendment Git blob-byte digest; it
  must match the GPU run's recorded
  `b60e31491fc6221bc474c1da511a894dce98688771a432e37a4479312db4938f`
- `lock_sha256`, `calibration_sha256`, `decision_state_projection_sha256`:
  unchanged historical protocol digests
- `source_manifest_files`: 64

The v2 validator requires the whole historical object to equal independently
authenticated historical Git evidence. Original JSON and source identities
remain in Git commit `158ba388...`; no artifact copies are needed. Fresh clones
must retain that history; shallow clones lacking required Git objects fail.

The manifest must not reference Stage B's own commit ID. Prepare its exact LF
Git representation first, then put its digest into the amendment. Stage B's
sole parent must be Stage A and its diff may contain only the two new-location
provenance files. Stage A may follow the existing merge; no history rewriting
or relaxation of the two-stage rule is needed. No Stage B metadata is finalized
and no future SHA is invented in this task.

### CRLF revision: preserve hash contracts

The selected fix restores the two relocated provenance JSON files to their
exact historical LF Git bytes. `.gitattributes` applies `text eol=lf` only to
these two paths so Windows checkouts do not replace their LF bytes with CRLF.
JSON values and Git blob identities remain unchanged. The Stage A staging list
therefore gains `.gitattributes`; approval of the former 17-path list does not
automatically approve this expanded 18-path list. Stage B still changes only
the two provenance JSON files, written with LF bytes.

Hash producer/consumer trace:

- `sha256()` in `src/evaluation/phase2.py` hashes exact filesystem bytes.
  Calibration produces raw `lock.input_sha256` and Phase 1 hashes.
  `validate_lock()` preserves raw scientific-input checks; historical source
  hashes in that lock remain evidence rather than current source pins.
- TEST constructs execution `input_sha256` from raw relocated input files,
  including source files. `provenance_amendment_sha256` hashes the amendment's
  raw working-file bytes. Lock/calibration and Phase 1 fields likewise retain
  their raw-byte contract. None of these fields is changed to Git hashing.
- Execution `reviewed_source_manifest_sha256` already hashes exact canonical
  Git blob bytes. Current source identity checks already use canonical Git
  blob IDs. These identities are distinct from raw-file SHA256.
- `PredictionJournal` reconstructs execution metadata and requires complete
  equality with the saved manifest on resume, including raw input/amendment
  digests. It hashes the saved execution manifest's raw bytes; prediction rows
  and the completed summary bind to that digest. No mismatch is skipped and
  no existing field is reinterpreted or removed.
- The final audit checks the completed summary/row bindings and authenticates
  the original metadata/source identities against the fixed historical Git
  freeze. Historical amendment Git bytes were LF and their SHA256 was
  statically confirmed equal to the GPU run's raw-file amendment digest. This
  is an equality check on original bytes, not normalization of historical
  execution evidence. Historical raw source hashes must not be reinterpreted
  as canonical Git hashes.
- Preflight enumerates required inputs and calls the evaluator's strict lock
  validator. Checkpoint-sidecar input hashes and their verification are
  unaffected. No checkpoint metadata or scientific input is normalized.

Restoring LF is sufficient for the identified relocated-amendment CRLF issue
and safer than a hash-protocol migration. It does not promise portable resume
across changed source-file bytes, runtimes, devices or metadata versions. Raw
source-input digests remain strict: LF/CRLF changes there must still reject
resume. Migrated Stage A source is not the original GPU source; the completed
run is audited historically and is never resumed under migrated code.

Existing CRLF checkouts of these two JSON files need an explicit byte-preserving
LF restoration; adding attributes alone does not rewrite already checked-out
files. This targeted revision performs that restoration only for the two
provenance files. It does not alter any final GPU artifact or scientific input.
The complete run, manifest, journal and summary must always be retained in their
original bytes. No hash-contract code change or new execution schema is needed.

The interrupted historical CPU journal was not found. No `historical_cpu/`
directory or replacement journal is created. GPU results must remain separate.

### Test suite

Run the entire repository suite with `python -m pytest`. The final evaluator's
tests cover validation-only IDs, calibration response and failure, ties,
temperature/lambda equivalence, exact budgets, seed reproducibility, label
rejection, stopped heuristic gates, aggregation, output schemas, paired tests,
immutable configuration reuse, and Phase 1 integrity checks.
