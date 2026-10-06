# Frozen Phase 2 on Kaggle: GitHub source + Kaggle Input artifacts

GitHub branch `mth/phrase2` supplies source code, tests and documentation.
Kaggle Input supplies data, fitted checkpoints and frozen experiment artifacts.
Copy artifacts into their normal repository-relative paths and run the evaluator
directly. No packaging or extraction step is required.

The scientific protocol remains frozen. Do not run training, persistence,
preflight or calibration stages. Features, targets, seeds, hyperparameters,
T/lambda, selection, aggregation and statistics remain unchanged.
G2=FAILED, G3a=NOT_EVALUATED, G3b=NOT_EVALUATED; no heuristic is reconstructed.

## Kaggle Input checklist

Create a private Dataset named `pooladapt-phase2-artifacts` with this layout:

- `data/scifact/corpus.jsonl`, `queries.jsonl`.
- `data/scifact/qrels/train.tsv`, `validation.tsv`, `test.tsv`.
- `data/scifact/splits/train_query_ids.txt`, `validation_query_ids.txt`, `split_metadata.json`.
- `results/phase2/checkpoints/sage_slo.pkl`, `sage_slo.metadata.json`, `pooladapt.pkl`, `pooladapt.metadata.json`.
- `results/phase2/07_evaluation/budget_calibration.csv`, `locked_configurations.json`.
- `results/phase2/01_candidate_logging/pool_features.csv`, `query_features.csv`, `metadata.json`.
- `results/phase2/02_oracle/mstar_labels_tau099.csv`.
- `results/phase2/development_reranking/validation/reranking_results.json`.
- `results/phase2/07_heuristic/heuristic_config.json`.
- The nine unchanged `results/phase1/` files named in the validation lock:
  `baseline_evaluation.csv`, `baseline_evaluation.json`, `baseline_predictions.json`,
  `candidate_pool_diagnostics.csv`, `failure_analysis.csv`, `pareto_analysis.csv`,
  `per_query_analysis.csv`, `phase1_analysis_summary.json`, `statistical_analysis.csv`.

Development and Phase 1 artifacts are checked for integrity only. They do not
trigger recalibration or TEST quality inspection. Other candidate feature CSVs
and checkpoint manifests may also be supplied in their original directories.
The original local Phase 2 summary is not required: existing frozen lock and
calibration digests are pinned in evaluator source. Checkpoint checksums,
metadata, recipes, feature order, runtime compatibility and locked data remain
verified. Do not include interrupted CPU predictions, CPU TEST outputs or an
old GPU run in this Input Dataset. Keep those local files untouched.
Artifacts remain gitignored; do not commit data, results, checkpoints or caches.

## Exact setup commands

Attach the Input Dataset, enable GPU and Internet, then run in a `%%bash` cell:

```bash
set -euo pipefail
cd /kaggle/working
git clone -b mth/phrase2 https://github.com/kimngoc2520/pooladapt.git pooladapt
cd /kaggle/working/pooladapt
mkdir -p data results
cp -a /kaggle/input/pooladapt-phase2-artifacts/data/. data/
cp -a /kaggle/input/pooladapt-phase2-artifacts/results/. results/

python -m pip install uv
uv python install 3.11.15
uv venv --python 3.11.15 /kaggle/working/phase2-venv
uv pip install --python /kaggle/working/phase2-venv/bin/python \
  numpy==2.4.6 scipy==1.17.1 scikit-learn==1.9.1 torch==2.14.0 \
  sentence-transformers==6.1.0 rank-bm25==0.2.2 transformers==5.17.0 \
  tokenizers==0.23.2 huggingface-hub==1.32.0 safetensors==0.8.0 \
  joblib==1.6.0 threadpoolctl==3.7.0 tqdm==4.70.1

/kaggle/working/phase2-venv/bin/python - <<'PY'
import torch
from experiments.run_phase2_evaluation import LOCK, read_json, validate_lock
from src.reranking.cross_encoder import CrossEncoderReranker
assert torch.cuda.is_available(), 'CUDA unavailable; CPU fallback is forbidden'
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('CUDA GPU:', torch.cuda.get_device_name(0))
lock = read_json(LOCK)
validate_lock(lock)
reranker = CrossEncoderReranker(model_name=lock['cross_encoder_model'], device='cuda')
reranker.warm_up()
print('Effective CrossEncoder device:', reranker.effective_device)
PY
```

This verifies frozen inputs and performs an empty untimed warm-up; no TEST
inference. Runtime releases must match the validation lock; torch may have a
CUDA build suffix for the same release. Stop on installation or CUDA warm-up
failure rather than substitute CPU or change releases. Actual Kaggle driver
compatibility must be verified on Kaggle.

Pretrained models retain their existing names and defaults and download from
Hugging Face normally. Alternatively supply the existing exact cached weights
via Input, copy them to `hf_cache/`, and set `HF_HUB_CACHE` and
`HF_HUB_OFFLINE=1` before starting Python. Caches remain outside Git. The
evaluator requires no cache manifest.

## Fresh final GPU command

```bash
set -euo pipefail
cd /kaggle/working/pooladapt
mkdir -p results/phase2/07_evaluation_gpu
/kaggle/working/phase2-venv/bin/python -u experiments/run_phase2_evaluation.py test \
  --dataset-dir data/scifact \
  --device cuda \
  --output-dir results/phase2/07_evaluation_gpu \
  --fresh-output 2>&1 | tee results/phase2/07_evaluation_gpu/run.log
```

The evaluator resolves original Windows/Linux locked paths against the clone.
Scientific artifacts still require exact original bytes. Git source line-ending
conversion is accepted without accepting source content changes. The two
authorized device/output/resume implementation files use current source digests
in the execution journal; all other originally locked source contents must
match. The validation lock and calibration are never rewritten.

Loading and warm-up remain outside latency measurements. CE timing measures
`predict(pairs)`; reranking includes preparation, inference and sorting. CUDA
synchronization waits for GPU work at the timing boundaries. Unavailable CUDA
or an effective CPU model fails explicitly. TEST labels are loaded only after
all frozen predictions complete.

## Resume the same GPU run

```bash
set -euo pipefail
cd /kaggle/working/pooladapt
/kaggle/working/phase2-venv/bin/python -u experiments/run_phase2_evaluation.py test \
  --dataset-dir data/scifact \
  --device cuda \
  --output-dir results/phase2/07_evaluation_gpu \
  --resume 2>&1 | tee -a results/phase2/07_evaluation_gpu/run.log
```

Preserve the whole output directory, same Git commit, frozen inputs, runtime
and GPU model. Do not pull new code or overwrite the journal while resuming.
Across session resets, save/restore outputs to durable storage and recreate the
same workspace/environment. Working storage must not be assumed to survive.
Run one evaluator per output directory.

Canonical key: `(query_id, method, target_budget, seed)`. Complete keys are
skipped; duplicate, foreign and corrupt records fail. Every append is flushed
and fsynced. Torn final records are preserved before safe tail recovery.
`execution_manifest.json` binds actual GPU/device/CUDA runtime, runtime versions,
current source hashes, frozen input hashes, lock, calibration and Phase 1 hashes
for the same run. CPU records cannot be imported. Resume can finish interrupted
aggregation; a COMPLETE run is refused.

## Outputs and monitoring

Under `results/phase2/07_evaluation_gpu/`: `execution_manifest.json`,
`test_predictions.jsonl`, `run.log`, `comparative_results.csv`,
`quality_cost_latency.csv`, `random_seed_results.csv`, `test_query_metrics.csv`,
`statistical_analysis.csv`, `phase2_summary.json`. Original calibration, lock,
Phase 1 and local CPU artifacts remain untouched.

```bash
tail -n 20 /kaggle/working/pooladapt/results/phase2/07_evaluation_gpu/run.log
wc -l /kaggle/working/pooladapt/results/phase2/07_evaluation_gpu/test_predictions.jsonl
nvidia-smi
```

Logs must show GPU name, effective `cuda` device, then `TEST n/300 complete`
or resume skips. Expect 37 records/query, 11,100 records for all eight successful
frozen adaptive points. Random uses seeds 0,1,2,3,4. Historical audit/render
scripts default to the original output paths; do not use those defaults to
inspect an interrupted CPU run.

Run repository tests with `python -m pytest tests -q`. Default pytest collection
also targets only `tests/` and excludes generated `results/` trees.
