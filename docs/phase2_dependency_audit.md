# Runtime dependency audit for the frozen Phase 2 TEST command

Audited the current `test_run` path, `validate_lock`, checkpoint loader,
feature construction, selectors, BM25/Dense/RRF, CrossEncoder, journal and
aggregation writers. Also inspected the actual validation lock, calibration
CSV, historical inspection manifest, checkpoint sidecars and persistence
manifest. No TEST inference or partial TEST quality was inspected.

There are **29 required non-Git files**: 18 artifact paths in
`lock.input_sha256`, nine Phase 1 files, and the lock/calibration files.
The same lock also requires 60 source files, supplied by the Git clone.
The exact plain checklist is [phase2_required_files.txt](phase2_required_files.txt).

## A. Required before the run, classified

1. **SciFact dataset files (8):** `data/scifact/corpus.jsonl`,
   `data/scifact/queries.jsonl`, `data/scifact/qrels/train.tsv`,
   `data/scifact/qrels/validation.tsv`, `data/scifact/qrels/test.tsv`,
   `data/scifact/splits/train_query_ids.txt`,
   `data/scifact/splits/validation_query_ids.txt`,
   `data/scifact/splits/split_metadata.json`.
2. **Retrieval/candidate/feature artifacts (3):**
   `results/phase2/01_candidate_logging/pool_features.csv`,
   `results/phase2/01_candidate_logging/query_features.csv`,
   `results/phase2/01_candidate_logging/metadata.json`.
3. **Oracle/M* provenance (1):** `results/phase2/02_oracle/mstar_labels_tau099.csv`.
4. **SAGE-SLO (2):** `results/phase2/checkpoints/sage_slo.pkl`,
   `results/phase2/checkpoints/sage_slo.metadata.json`.
5. **PoolAdapt (2):** `results/phase2/checkpoints/pooladapt.pkl`,
   `results/phase2/checkpoints/pooladapt.metadata.json`.
6. **Validation calibration/configuration (2):**
   `results/phase2/evaluation/budget_calibration.csv`,
   `results/phase2/evaluation/locked_configurations.json`.
7. **Heuristic/gate configuration (1):**
   `results/phase2/07_heuristic/heuristic_config.json`.
8. **Pretrained models:** model weights/tokenizers/configuration must be
   obtainable for `sentence-transformers/all-MiniLM-L6-v2` and
   `cross-encoder/ms-marco-MiniLM-L-6-v2`. No cache file is a mandatory Input
   upload with working Kaggle Internet. See the cache discussion below.
9. **Other provenance (10):**
   `results/phase2/development_reranking/validation/reranking_results.json`,
   and the nine exact Phase 1 files shown in the Input tree below.

The split is 647 TRAIN + 162 VALIDATION + 300 TEST. Original `qrels/train.tsv`
contains the 809 development queries; it is not the 647-query derived split.
Saved pool/query feature files cover those 809 development IDs, not TEST.
The policy feature order is `top1_top2_rrf_margin`, `top20_overlap`,
`rank_correlation_union`, `score_entropy`, `score_gini`, `redundancy`,
`concentration`, `sparse_dense_agreement`, `sparse_dense_disagreement`,
`query_length`, `num_numeric_tokens`, `avg_idf`, `max_idf`. Query IDF comes from
the live corpus BM25 index; lexical features use the repository regex tokenizer.
No saved IDF table, entity/NER model or additional feature cache is required.

## Actual I/O and integrity trace

- CLI `test` calls `test_run`. Requested CUDA is checked before inference.
  `read_json(LOCK)` loads the frozen lock; `validate_lock` pins its SHA256 to
  `fe1a76073021fd913148c2429eedf37e322220de99febf1a05ea1e828a89048c`.
- `validate_lock` also requires `budget_calibration.csv`, pinned to SHA256
  `abdd419d36854f9b70608282b1172caae21424dd1e3bf9c4798b6f3595389738`.
  The CSV is hashed, not used to select/recompute T or lambda. Runtime policies
  read the already locked values from `lock.configurations`.
- Every one of the 78 `input_sha256` paths is resolved by `locked_path` from
  the old repository root to the current clone and opened to hash its bytes.
  That includes the 18 external artifacts above and 60 Git source files.
  Scientific artifacts require exact original bytes. Source-only LF/CRLF
  conversion is accepted. The two authorized evaluator/reranker infrastructure
  files bind their current source digests to the execution journal instead of
  enforcing the old CPU implementation hash.
- `phase1_hashes()` recursively hashes **every file** in `results/phase1/` and
  compares the complete map with `lock.phase1_sha256`. All nine files are needed;
  extra files there also cause a mismatch. No Phase 1 metric is used for a new
  quality decision. The lock pins Python/numpy/scipy/sklearn/torch/
  sentence-transformers/rank-bm25 releases. CUDA torch build suffixes are allowed
  only for the same release.
- `load_beir_dataset(dataset_dir)` opens `corpus.jsonl` and `queries.jsonl`.
  Its default `qrels_split=None` does not read relevance labels.
  BM25 tokenizes corpus text and builds an in-memory `BM25Okapi` index.
  Dense loads the named SentenceTransformer, encodes corpus documents in-memory,
  and computes query embeddings during evaluation. RRF merges the two Top-100
  lists with k=60. No BM25 index, dense embedding array, FAISS index, cached
  candidate pool or recorded candidate CSV is loaded.
- `load_model` calls `load_checkpoint`. It reads `.metadata.json`, hashes the
  `.pkl`, then uses `pickle.load` for the **complete fitted sklearn Pipeline**.
  It checks model identity, feature order, grid, numpy/sklearn versions,
  reload-verification status, hyperparameters and preprocessing. No joblib
  file is used. Metadata itself is also hashed by `validate_lock`.
- Both metadata sidecars contain historical `input_sha256` references to the
  two development runner source files, pool/query feature CSVs, M* CSV and
  `results/phase2/development_reranking/train/reranking_results.json`.
  The loader does **not** recursively open those metadata references. The TRAIN
  reranking JSON is therefore not required by final TEST. Other referenced
  artifacts already appear in the actual lock and checklist. Metadata exactly
  matches the embedded `lock.checkpoint_information` in the audited repository.
- `frozen_features` calculates the 13 ordered features from current pool ranks,
  scores, corpus tokens and BM25 IDF for each TEST query. It does not load saved
  TEST feature rows or qrels. SAGE uses `model_logits`; PoolAdapt uses
  `PoolAdaptSelector.logits`; both consume that feature vector and locked T/lambda.
  PACE-EF uses BM25 term contributions and current RRF scores. Random uses
  query-keyed seeds 0,1,2,3,4. Selection functions open no external artifact.
- `CrossEncoderReranker` lazily loads the named CrossEncoder with explicit CUDA
  and performs an empty warm-up. Loading/warm-up are outside recorded latency.
  Timed `predict(pairs)` and total reranking synchronize CUDA at their existing
  boundaries. No learned PACE or heuristic checkpoint is loaded.
- For a fresh isolated output, `PredictionJournal` creates the execution manifest
  and empty prediction journal. Every appended row is flushed/fsynced.
  The manifest records lock, calibration, input and Phase 1 hashes, runtime,
  GPU/device/CUDA and current source hashes. Resume reads only this same GPU
  manifest/journal and verifies provenance and unique canonical keys.
- Only after all predictions are complete does `load_qrels` open
  `data/scifact/qrels/test.tsv` for metrics. Aggregation, five-seed Random
  summaries and Wilcoxon/Holm statistics use completed in-memory rows. A second
  `validate_lock` repeats integrity checks before result writers run.
- Isolated TEST builds its summary from the lock. It does **not** require
  a prior stage summary. `--pooladapt-model`/`--sage-model` CLI
  options do not override TEST paths: TEST reads `lock.models`.

The historical `inspection_manifest.json` has 46 references. Neither it nor
its reference map is traversed by `test_run`. `checkpoint_manifest.json` is
read by the calibration stage, not the final TEST or checkpoint loader.
These files were inspected for this audit, not promoted to runtime requirements.

## Recommended Kaggle Input tree

```text
<dataset>/
├── data/scifact/
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   ├── qrels/
│   │   ├── train.tsv
│   │   ├── validation.tsv
│   │   └── test.tsv
│   └── splits/
│       ├── train_query_ids.txt
│       ├── validation_query_ids.txt
│       └── split_metadata.json
└── results/
    ├── phase1/
    │   ├── baseline_evaluation.csv
    │   ├── baseline_evaluation.json
    │   ├── baseline_predictions.json
    │   ├── candidate_pool_diagnostics.csv
    │   ├── failure_analysis.csv
    │   ├── pareto_analysis.csv
    │   ├── per_query_analysis.csv
    │   ├── phase1_analysis_summary.json
    │   └── statistical_analysis.csv
    └── phase2/
        ├── 01_candidate_logging/
        │   ├── pool_features.csv
        │   ├── query_features.csv
        │   └── metadata.json
        ├── 02_oracle/
        │   └── mstar_labels_tau099.csv
        ├── 07_heuristic/
        │   └── heuristic_config.json
        ├── evaluation/
        │   ├── budget_calibration.csv
        │   └── locked_configurations.json
        ├── checkpoints/
        │   ├── sage_slo.pkl
        │   ├── sage_slo.metadata.json
        │   ├── pooladapt.pkl
        │   └── pooladapt.metadata.json
        └── development_reranking/validation/
            └── reranking_results.json
```

Copy unchanged bytes, after cloning the source branch:

```bash
cd /kaggle/working/pooladapt
mkdir -p data results
cp -a /kaggle/input/<dataset>/data/. data/
cp -a /kaggle/input/<dataset>/results/. results/
```

Do not copy any interrupted CPU/GPU TEST output into this Input tree.

## Pretrained models and optional offline caches

Inspected the installed SentenceTransformer and CrossEncoder constructors:
`local_files_only=False`, `revision=None`; project wrappers leave those defaults
unchanged. With working Internet, Hugging Face can fetch weights, tokenizer and
model configuration into its normal cache. With `HF_HUB_OFFLINE=1` or blocked
Internet, a complete usable cache for **both** named models is necessary.

The current validation lock pins model names, not Hugging Face revision IDs or
weight-file hashes. If byte-identical pretrained snapshots are needed, supply
the original cache rather than assume a fresh `main` download is identical.
No methodology, model loading defaults or revision was changed in this audit.

The inspected local snapshots are:

- Dense: `models--sentence-transformers--all-MiniLM-L6-v2/refs/main` points to
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
  Under `snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/` the existing files
  are `config.json`, `config_sentence_transformers.json`, `model.safetensors`,
  `modules.json`, `README.md`, `sentence_bert_config.json`,
  `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`,
  `vocab.txt`, `1_Pooling/config.json`.
- CrossEncoder: `models--cross-encoder--ms-marco-MiniLM-L-6-v2/refs/main` points
  to `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
  Under `snapshots/233902d25c440f23af6f7d6e94d2946bac0bee0a/` the existing files
  are `config.json`, `model.safetensors`, `special_tokens_map.json`,
  `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`.

For offline use, preserve these repository cache structures under
`results/phase2/hf_cache/`, including referenced blob targets if snapshots use
symlinks. Export `HF_HUB_CACHE=$PWD/results/phase2/hf_cache` and
`HF_HUB_OFFLINE=1` before starting Python. These cache paths are optional additions
to the Input tree; there is no packaging/cache manifest requirement.

## B. Automatically generated by isolated final TEST

Under `results/phase2/evaluation/final_gpu/`:

```text
execution_manifest.json
test_predictions.jsonl
comparative_results.csv
quality_cost_latency.csv
random_seed_results.csv
test_query_metrics.csv
statistical_analysis.csv
phase2_summary.json
```

These completed GPU outputs already exist in the migrated local layout. Final
audit/report consumers use this directory and its COMPLETE summary. Frozen
inputs and the two provenance JSON files eligible for Git tracking belong in
`results/phase2/evaluation/`. The root `results/phase2/phase2_summary.json`
was intentionally deleted. Legacy stage status uses `evaluation/stage_summary.json`
if separately generated; this migration does not create it.

The relocated provenance JSON files are unchanged historical evidence. Current
path-refactored source does not match their original 64-source manifest and
still requires a separately authorized Stage A implementation commit followed
by a Stage B provenance-only freeze. The validator retains its source-blob,
manifest-digest, decision-state and Git-parent checks. No current provenance
PASS or fresh-clone readiness is implied by the path corrections. The existing
GPU execution retains its original provenance digests; a new source freeze must
not replace those records. See
[the migration details](phase2_final_evaluation.md#path-migration-provenance-remains-pending).
The interrupted CPU journal was not found; no replacement is created.

The journal/manifest are created before prediction. Result CSVs and COMPLETE
summary are produced only after all 11,100 predictions/300 queries complete and
integrity checks pass. Calibration and original lock are not regenerated.
`run.log` is generated only if the shell redirects output or uses `tee`;
the evaluator itself does not create that log. Resume recovery may create
`torn_tail*.bin`; atomic commits may briefly create `.json.tmp` files.

## C. Not required; omit from the Input upload

- `results/phase2/evaluation/inspection_manifest.json`.
- `results/phase2/checkpoints/checkpoint_manifest.json`.
- `results/phase2/evaluation/stage_summary.json` (optional legacy stage status).
- `results/phase2/01_candidate_logging/candidate_features.csv`,
  `results/phase2/01_candidate_logging/train_validation_query_ids.csv`.
- `data/scifact/qrels/train_split.tsv` and any extra download archive/metadata.
- `results/phase2/development_reranking/train/reranking_results.json`.
- `results/phase2/02_oracle/mstar_analysis_tau099.json` and other oracle analyses.
- `results/phase2/09_sage_slo/sage_slo_results.csv`, `sage_slo_sanity.json`.
- `results/phase2/10_pooladapt/pooladapt_results.csv`, `pooladapt_sanity.json`.
- Candidate label tables, feature analysis outputs, visualizations, feasibility
  and heuristic diagnostic reports: their inclusion in the historical inspection
  map does not make them TEST dependencies.
- Existing CPU/GPU `test_predictions.jsonl`, execution manifests, TEST result
  CSVs/summaries, reports/canvases and logs. Never upload them as fresh inputs.
- Legacy ZIPs, packaging manifests and extracted verification workspaces.
- No learned PACE checkpoint, heuristic checkpoint, saved BM25 index, dense
  embedding matrix, FAISS index or cached TEST feature table is required.

## Read-only preflight and final command

Use [preflight_phase2_test.py](../experiments/preflight_phase2_test.py). This is
new source code: copy it into the existing clone's `experiments/` directory or
obtain it from the source branch after your manual review/update. It is not a
generated artifact or a packaging workflow.

```bash
python -u experiments/preflight_phase2_test.py \
  --dataset-dir data/scifact \
  --output-dir results/phase2/evaluation/final_gpu
```

It checks all 89 file dependencies first, calls the evaluator's actual integrity
rules, checks 647/162/300 disjoint populations, all query/document IDs, split
metadata, 809 development feature rows and their 13 features, 21 locked points,
stopped gates, and both frozen checkpoint sidecars/pipelines. It builds BM25
from corpus text, but runs no query retrieval. TEST qrels are checked only for
IDs/schema; relevance scores are not read into features or evaluated. It verifies
CUDA/GPU name and loads Dense/CrossEncoder with only empty warm-ups. It never
opens a prediction journal, fits, recalibrates or writes experiment artifacts.
The final line is `PHASE 2 PREFLIGHT: PASS` only after every check succeeds.

Do **not** invoke `run_phase2_evaluation.py preflight`: that historical preparation
stage writes inspection/status artifacts and refuses an existing calibration lock.

After PASS:

```bash
python -u experiments/run_phase2_evaluation.py test \
  --dataset-dir data/scifact \
  --device cuda \
  --output-dir results/phase2/evaluation/final_gpu \
  --fresh-output
```

Potential blockers before/during TEST: missing/altered frozen bytes or source
files, extra Phase 1 files, wrong package/Python releases, incompatible sklearn
pickle, missing query/feature rows, unavailable CUDA, incompatible driver/kernel,
incomplete offline cache or failed Hub downloads, non-writable/occupied output,
disk exhaustion, memory exhaustion, or an interruption. Preflight catches static
integrity/population/model-readiness failures; empty warm-ups cannot guarantee
resources for every later batch. It does not relax checks, change versions,
clear outputs or choose methods to bypass a failure.

The frozen runtime is Python 3.11.15, numpy 2.4.6, scipy 1.17.1,
scikit-learn 1.9.1, torch 2.14.0, sentence-transformers 6.1.0,
rank-bm25 0.2.2. Journal metadata also requires installed transformers,
tokenizers, huggingface-hub, safetensors, joblib, threadpoolctl and tqdm.
Existing [Kaggle setup instructions](phase2_kaggle_gpu.md) pin those supporting
versions. Local static verification passed; actual Kaggle CUDA readiness must
still pass there.

## Exact locked Git source paths (not Input uploads)

```text
experiments/analyze_easy_hard_features.py
experiments/audit_prefix10_gap.py
experiments/oracle_mstar.py
experiments/persist_phase2_checkpoints.py
experiments/prepare_phase2_scifact.py
experiments/run_baselines.py
experiments/run_candidate_labels.py
experiments/run_candidate_logging.py
experiments/run_candidate_pool_diagnostics.py
experiments/run_evaluation.py
experiments/run_feature_analysis.py
experiments/run_full_rerank.py
experiments/run_multivariate_feasibility.py
experiments/run_pace_ef.py
experiments/run_phase1_analysis.py
experiments/run_phase2_evaluation.py
experiments/run_pooladapt.py
experiments/run_retrieval.py
experiments/run_sage_slo.py
experiments/write_heuristic_status.py
src/__init__.py
src/agent/__init__.py
src/agent/router.py
src/agent/state.py
src/agent/workflow.py
src/analysis/__init__.py
src/analysis/correlation.py
src/analysis/development_reranking.py
src/analysis/oracle_mstar.py
src/baselines/__init__.py
src/baselines/fixed_prefix.py
src/baselines/full_rerank.py
src/baselines/pace.py
src/baselines/pace_ef.py
src/baselines/random_selection.py
src/baselines/sage_slo.py
src/data.py
src/data/__init__.py
src/data/loader.py
src/data/scifact_phase2.py
src/data/split.py
src/evaluation/__init__.py
src/evaluation/cost_metrics.py
src/evaluation/efficiency_metrics.py
src/evaluation/latency_metrics.py
src/evaluation/phase2.py
src/evaluation/phase2_inputs.py
src/evaluation/retrieval_metrics.py
src/evaluation/statistical_tests.py
src/pooladapt/__init__.py
src/pooladapt/checkpoints.py
src/pooladapt/features.py
src/pooladapt/policy.py
src/pooladapt/selector.py
src/reranking/__init__.py
src/reranking/cross_encoder.py
src/retrieval/__init__.py
src/retrieval/bm25.py
src/retrieval/dense.py
src/retrieval/rrf.py
```
