"""Single owner of final validation calibration, immutable locks, and TEST runs.

Usage: python experiments/run_phase2_evaluation.py calibrate --pooladapt-model PATH --sage-model PATH
       python experiments/run_phase2_evaluation.py test
No model training is performed by this evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_pooladapt import feature_rows, index, load_qrels
from src.baselines.fixed_prefix import select_fixed_prefix
from src.baselines.pace_ef import select_prefix
from src.baselines.random_selection import select_random
from src.baselines.sage_slo import model_logits, select_sage_slo
from src.data import load_beir_dataset
from src.data.scifact_phase2 import prepare_split, query_ids, read_qrel_rows, split_original_train, validate_split
from src.evaluation.phase2 import (CALIBRATION_FIELDS, METRICS, RESULT_FIELDS, SEEDS, TARGETS,
    aggregate, calibrate, heuristic_gates, paired_statistics, predict_budgets,
    random_aggregate, sha256, tradeoff_findings, verify_population, write_csv)
from src.evaluation.phase2_inputs import frozen_features, pace_order, validate_pool
from src.evaluation.phase2_execution import PredictionJournal, expected_keys, locked_path
from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.pooladapt.selector import BUDGET_GRID, FROZEN_FEATURES, PoolAdaptSelector, select_rrf_prefix
from src.pooladapt.checkpoints import load_checkpoint, metadata_path, pipeline_configuration
from src.reranking.cross_encoder import CrossEncoderReranker, DEFAULT_MODEL_NAME
from src.retrieval import BM25Retriever, DenseRetriever, DEFAULT_MODEL, fuse_ranked_lists

OUT = ROOT / 'results/phase2/07_evaluation'
SUMMARY = ROOT / 'results/phase2/phase2_summary.json'
LOCK = OUT / 'locked_configurations.json'
# Trusted digests of the existing frozen SciFact calibration, not a packaging manifest.
FROZEN_LOCK_SHA256 = 'fe1a76073021fd913148c2429eedf37e322220de99febf1a05ea1e828a89048c'
FROZEN_CALIBRATION_SHA256 = 'abdd419d36854f9b70608282b1172caae21424dd1e3bf9c4798b6f3595389738'
PROVENANCE_AMENDMENT = OUT / 'provenance_amendment.json'
REVIEWED_SOURCE_MANIFEST = OUT / 'reviewed_source_manifest.json'


def read_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition: bool, artifact: Path | str, detail: str) -> None:
    """Fail closed with the artifact responsible for a provenance mismatch."""
    if not condition:
        raise ValueError(f'{artifact}: {detail}')


def canonical_json_digest(value: Any) -> str:
    """Hash a stable JSON projection without platform or formatting effects."""
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def decision_projection(lock: dict[str, Any]) -> dict[str, Any]:
    """Return only immutable experimental-decision state from the historical lock."""
    return {
        'checkpoint_information': lock['checkpoint_information'],
        'models': lock['models'],
        'configurations': lock['configurations'],
        'candidate_pool_size': lock['candidate_pool_size'],
        'budget_grid': lock['budget_grid'],
        'target_budgets': lock['target_budgets'],
        'calibration_tolerance': lock['calibration_tolerance'],
        'gates': lock['gates'],
        'cross_encoder_model': lock['cross_encoder_model'],
        'dense_model': lock['dense_model'],
        'cross_encoder_configuration': lock['cross_encoder_configuration'],
        'random_seeds': lock['random_seeds'],
        'test_ids': lock['test_ids'],
        'metrics': list(METRICS),
        'statistical_protocol': lock['statistical_protocol'],
        'test_used_for_calibration': lock['test_used_for_calibration'],
    }


def git_output(*arguments: str) -> str:
    """Run a read-only Git query and fail closed on repository ambiguity."""
    result = subprocess.run(('git', *arguments), cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def git_blob_sha256(path: Path, revision: str = 'HEAD') -> str:
    """Hash exact bytes from a Git blob, independent of checkout line endings."""
    relative = path.relative_to(ROOT).as_posix()
    blob = git_output('rev-parse', f'{revision}:{relative}')
    result = subprocess.run(('git', 'cat-file', 'blob', blob), cwd=ROOT, check=True,
                            capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def validate_current_git_blob(path: Path) -> None:
    """Require a tracked artifact to match its canonical Git blob."""
    relative = path.relative_to(ROOT).as_posix()
    expected = git_output('rev-parse', f'HEAD:{relative}')
    actual = git_output('hash-object', '--path=' + relative, str(path))
    require(actual == expected, path, 'working-tree bytes differ after Git canonicalization')


def validate_reviewed_source_manifest(manifest: dict[str, Any], reviewed_commit: str) -> dict[str, Any]:
    """Require exact reviewed-commit blobs, independent of line endings."""
    require(manifest.get('reviewed_commit') == reviewed_commit,
            REVIEWED_SOURCE_MANIFEST, 'reviewed commit differs')
    files = manifest.get('files')
    require(isinstance(files, list) and files, REVIEWED_SOURCE_MANIFEST, 'source manifest is empty')
    for entry in files:
        path = entry.get('path')
        require(isinstance(path, str) and path.startswith(('src/', 'experiments/')),
                REVIEWED_SOURCE_MANIFEST, f'invalid source path: {path}')
        current = ROOT / path
        require(current.is_file(), current, 'required reviewed source is missing')
        require(git_output('hash-object', '--path=' + path, str(current)) == entry.get('blob'),
                current, 'reviewed source bytes differ after Git canonicalization')
        require(git_output('rev-parse', f'{reviewed_commit}:{path}') == entry.get('blob'),
                current, 'reviewed Git blob differs')
    return manifest


def validate_provenance_amendment(lock: dict[str, Any]) -> None:
    """Validate the amendment and its immutable decision-state projection."""
    require(PROVENANCE_AMENDMENT.is_file(), PROVENANCE_AMENDMENT, 'missing provenance amendment')
    require(REVIEWED_SOURCE_MANIFEST.is_file(), REVIEWED_SOURCE_MANIFEST, 'missing reviewed source manifest')
    amendment = read_json(PROVENANCE_AMENDMENT)
    require(amendment.get('status') == 'finalized / reviewed',
            PROVENANCE_AMENDMENT, 'provenance amendment is pending')
    if amendment.get('original_lock_sha256') != FROZEN_LOCK_SHA256 or amendment.get('calibration_sha256') != FROZEN_CALIBRATION_SHA256:
        raise ValueError('provenance amendment does not pin the historical lock/calibration')
    reviewed_commit = amendment.get('reviewed_execution_commit')
    require(isinstance(reviewed_commit, str) and len(reviewed_commit) == 40,
            PROVENANCE_AMENDMENT, 'invalid reviewed implementation commit')
    projection_digest = canonical_json_digest(decision_projection(lock))
    if amendment.get('decision_state_projection_sha256') != projection_digest:
        raise ValueError('decision-bearing configuration projection differs')
    validate_current_git_blob(REVIEWED_SOURCE_MANIFEST)
    manifest_digest = git_blob_sha256(REVIEWED_SOURCE_MANIFEST)
    if amendment.get('reviewed_source_manifest_sha256') != manifest_digest:
        raise ValueError('provenance amendment source manifest differs')
    validate_reviewed_source_manifest(read_json(REVIEWED_SOURCE_MANIFEST), reviewed_commit)
    head = git_output('rev-parse', 'HEAD')
    parents = git_output('rev-list', '--parents', '-n', '1', head).split()
    require(len(parents) == 2 and parents[1] == reviewed_commit,
            'git', 'HEAD must be the provenance freeze commit whose parent is reviewed implementation commit')
    changed = git_output('diff-tree', '--no-commit-id', '--name-only', '-r', head).splitlines()
    allowed = {
        'results/phase2/07_evaluation/provenance_amendment.json',
        'results/phase2/07_evaluation/reviewed_source_manifest.json',
    }
    require(set(changed) <= allowed, 'git', 'provenance freeze commit changed non-provenance files')


def phase1_hashes() -> dict[str, str]:
    """Snapshot every frozen Phase 1 file."""
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'results/phase1').rglob('*')) if p.is_file()}


def environment_versions() -> dict[str, str]:
    """Record and lock installed packages affecting predictions and timing."""
    return {'python': platform.python_version(), **{name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'sentence-transformers', 'rank-bm25')}}


def execution_versions() -> dict[str, str]:
    """Bind supporting tokenizer/model-loading packages for same-run resume."""
    return {**environment_versions(), **{name: importlib.metadata.version(name) for name in
            ('transformers', 'tokenizers', 'huggingface-hub', 'safetensors', 'joblib', 'threadpoolctl', 'tqdm')}}


def save_summary(summary: dict[str, Any]) -> None:
    """Persist status without presenting incomplete work as final TEST results."""
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_model(path: Path, expected_name: str | None = None) -> Any:
    """Load a trusted local fitted checkpoint; never fit during evaluation."""
    return load_checkpoint(path, expected_name=expected_name)


def preflight_run(args: argparse.Namespace) -> None:
    """Record verified repository prerequisites without training or evaluating TEST."""
    if LOCK.exists():
        raise ValueError('calibration lock already exists; preflight must not replace its summary')
    original_ids = query_ids(read_qrel_rows(args.dataset_dir / 'qrels/train.tsv'))
    test_ids = query_ids(read_qrel_rows(args.dataset_dir / 'qrels/test.tsv'))
    train, validation = split_original_train(original_ids)
    validate_split(set(train), set(validation), original_ids, test_ids)
    if len(test_ids) != 300:
        raise ValueError('expected 300 SciFact TEST queries')
    for name, ids in (('train', train), ('validation', validation)):
        artifact = read_json(ROOT / f'results/phase2/development_reranking/{name}/reranking_results.json')
        for method in artifact['methods'].values():
            verify_population([str(r['query_id']) for r in method['predictions']], ids, list(test_ids))
    gates = heuristic_gates(read_json(ROOT / 'results/phase2/07_heuristic/heuristic_config.json'))
    oracle = read_json(ROOT / 'results/phase2/02_oracle/mstar_analysis_tau099.json')
    gates['G1'] = 'PASS' if oracle['number_of_distinct_M_star_levels'] >= 3 and oracle['std_M_star'] > 0 else 'FAILED'
    inspected = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'results/phase2').rglob('*')) if p.is_file() and p != SUMMARY and OUT not in p.parents}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'inspection_manifest.json').write_text(json.dumps(inspected, indent=2) + '\n', encoding='utf-8')
    configurations = [dict(method=method, target_budget=target, calibration_status='PENDING_CHECKPOINT' if method in ('PoolAdapt', 'SAGE-SLO') else 'NOT_REQUIRED', locked=False)
                      for method in ('PoolAdapt', 'SAGE-SLO', 'Fixed Prefix', 'PACE-EF', 'Random', 'Full Rerank')
                      for target in ((100,) if method == 'Full Rerank' else TARGETS)]
    save_summary(dict(status='PENDING_CHECKPOINTS', dataset='SciFact',
                      split_information={'source': 'seed-42 80/20 original TRAIN split; recorded development IDs verified', 'train_queries': len(train), 'validation_queries': len(validation), 'test_queries': len(test_ids)},
                      candidate_pool_size=100, budget_grid=list(BUDGET_GRID), target_budgets=list(TARGETS), calibration_tolerance=.05,
                      cross_encoder_model=DEFAULT_MODEL_NAME, configurations=configurations, locked_configurations=None,
                      gates=gates, calibration_parameters=None, calibration_failures=[], final_test_results=None,
                      statistical_results=None, random_seeds=list(SEEDS), protocol_notes=protocol_notes(),
                      test_used_for_calibration=False, phase1_sha256=phase1_hashes(), environment=environment_versions(),
                      known_limitations=['No trained PoolAdapt or SAGE-SLO checkpoints were found; existing runners discard their fitted models.',
                                         'Reconstruction of fitted models has not been authorized; no final calibration or TEST evaluation has run.',
                                         'Missing checkpoints are a prerequisite issue, not an adaptive calibration failure.']))
    print('Verified preflight; final experiment pending fitted checkpoints.')


def calibrate_run(args: argparse.Namespace) -> None:
    """Calibrate using locked VALIDATION IDs, write lock before allowing TEST."""
    if LOCK.exists():
        raise ValueError('configuration lock already exists; refusing to overwrite it')
    original_phase1 = phase1_hashes()
    # Verify both frozen models and metadata before any calibration operation.
    model_paths = {'PoolAdapt': args.pooladapt_model.resolve(), 'SAGE-SLO': args.sage_model.resolve()}
    loaded_models, checkpoint_information = {}, {}
    from experiments.run_pooladapt import build_model as pooladapt_recipe
    from experiments.run_sage_slo import build_model as sage_recipe
    for method, path in model_paths.items():
        model = load_model(path, expected_name=method)
        metadata = read_json(metadata_path(path))
        if metadata['n_training_queries'] != 647 or len(set(metadata['training_query_ids'])) != 647:
            raise ValueError(f'{method}: expected exactly 647 recorded TRAIN queries')
        if metadata['feature_names'] != list(FROZEN_FEATURES) or metadata['feature_order'] != list(FROZEN_FEATURES):
            raise ValueError(f'{method}: ordered features differ from frozen features')
        if metadata['random_seed'] != 42 or metadata['validation_used_for_fitting'] or metadata['test_accessed']:
            raise ValueError(f'{method}: frozen training metadata mismatch')
        expected = pooladapt_recipe() if method == 'PoolAdapt' else sage_recipe()
        if pipeline_configuration(model) != pipeline_configuration(expected):
            raise ValueError(f'{method}: model class, preprocessing, or hyperparameters differ from frozen recipe')
        loaded_models[method] = model
        checkpoint_information[method] = {'path': str(path), 'metadata': metadata}
    checkpoint_manifest = read_json(ROOT / 'results/phase2/checkpoints/checkpoint_manifest.json')
    if original_phase1 != checkpoint_manifest['phase1_sha256']:
        raise ValueError('Phase 1 artifacts differ from checkpoint persistence snapshot')
    split = prepare_split(args.dataset_dir)
    train, validation = split['train_ids'], split['validation_ids']
    if len(validation) != 162 or len(train) != 647:
        raise ValueError('expected frozen 647 TRAIN / 162 VALIDATION split')
    for method, info in checkpoint_information.items():
        if info['metadata']['training_query_ids'] != train:
            raise ValueError(f'{method}: checkpoint TRAIN population/order differs from split lock')
    test_ids = sorted(load_qrels(args.dataset_dir / 'qrels/test.tsv'))
    if len(test_ids) != 300:
        raise ValueError('final SciFact TEST must contain exactly 300 queries')
    feature_dir = ROOT / 'results/phase2/01_candidate_logging'
    pool, query = index(feature_dir / 'pool_features.csv'), index(feature_dir / 'query_features.csv')
    verify_population(list(pool), train + validation, test_ids)
    verify_population(list(query), train + validation, test_ids)
    rows = feature_rows(pool, query, validation)
    source = read_json(ROOT / 'results/phase2/development_reranking/validation/reranking_results.json')
    if source.get('split') != 'validation':
        raise ValueError('validation reranking artifact has wrong split')
    predictions = {}
    for budget in BUDGET_GRID:
        saved = source['methods'][f'fixed_prefix_{budget}']['predictions']
        verify_population([str(r['query_id']) for r in saved], validation, train + test_ids)
        predictions[budget] = {str(r['query_id']): r for r in saved}
    relevance = load_qrels(args.dataset_dir / 'qrels/validation.tsv')
    verify_population(list(relevance), validation, train + test_ids)
    configurations, sanity, historical_audit = [], {}, {}
    models = model_paths
    for method, path in models.items():
        model = loaded_models[method]
        logits = np.asarray([PoolAdaptSelector(model).logits(row) if method == 'PoolAdapt' else model_logits(model, row, FROZEN_FEATURES) for row in rows])
        saved_path = ROOT / ('results/phase2/10_pooladapt/pooladapt_results.csv' if method == 'PoolAdapt' else 'results/phase2/09_sage_slo/sage_slo_results.csv')
        with saved_path.open(encoding='utf-8', newline='') as handle:
            saved_rows = list(csv.DictReader(handle))
        verify_population([r['query_id'] for r in saved_rows], validation, train + test_ids)
        by_id = {r['query_id']: r for r in saved_rows}
        discrepancies = []
        for qid, values in zip(validation, logits):
            old = by_id[qid]
            actual = int(predict_budgets(values[None, :], float(old['temperature']), float(old['budget_bias_lambda']))[0])
            if actual != int(old['predicted_budget']):
                discrepancies.append({'query_id': qid, 'historical_K': int(old['predicted_budget']), 'frozen_checkpoint_K': actual})
        configs, sanity[method] = calibrate(method, validation, validation, train + test_ids, logits)
        saved_sanity = read_json(saved_path.with_name('pooladapt_sanity.json' if method == 'PoolAdapt' else 'sage_slo_sanity.json'))
        expected_sanity = saved_sanity['average_predicted_budget']
        historical_audit[method] = {'historical_prediction_mismatches': len(discrepancies),
                                    'query_discrepancies': discrepancies,
                                    'historical_sanity_average_K': expected_sanity,
                                    'frozen_checkpoint_sanity_average_K': sanity[method]['lambda_sanity_average_K'],
                                    'interpretation': 'Frozen checkpoints named by the final protocol are authoritative. Historical development artifacts are preserved; disagreement does not trigger retraining, model selection, or parameter changes.'}
        for config in configs:
            budgets = predict_budgets(logits, config['temperature'], config['budget_bias_lambda'])
            config['validation_nDCG@10'] = float(np.mean([ndcg_at_k(predictions[int(k)][qid]['document_ids'], relevance[qid]) for qid, k in zip(validation, budgets)]))
            config['budget_type'] = 'adaptive'
        configurations.extend(configs)
    for method in ('Fixed Prefix', 'PACE-EF', 'Random', 'Full Rerank'):
        for target in ((100,) if method == 'Full Rerank' else TARGETS):
            configurations.append(dict(method=method, target_budget=target, actual_average_budget=target,
                relative_budget_error=0., calibration_status='REFERENCE' if method == 'Full Rerank' else 'NOT_REQUIRED',
                temperature=None, budget_bias_lambda=None, other_calibration_parameter=None,
                **{'validation_nDCG@10': None}, locked=True, budget_type='exact'))
    heuristic_path = ROOT / 'results/phase2/07_heuristic/heuristic_config.json'
    gates = heuristic_gates(read_json(heuristic_path))
    oracle = read_json(ROOT / 'results/phase2/02_oracle/mstar_analysis_tau099.json')
    gates['G1'] = 'PASS' if oracle['number_of_distinct_M_star_levels'] >= 3 and oracle['std_M_star'] > 0 else 'FAILED'
    inputs = [*models.values(), *(metadata_path(p) for p in models.values()), heuristic_path, feature_dir / 'pool_features.csv', feature_dir / 'query_features.csv',
              feature_dir / 'metadata.json', ROOT / 'results/phase2/development_reranking/validation/reranking_results.json',
              ROOT / 'results/phase2/02_oracle/mstar_labels_tau099.csv',
              *(args.dataset_dir / name for name in ('corpus.jsonl', 'queries.jsonl', 'qrels/train.tsv', 'qrels/validation.tsv', 'qrels/test.tsv', 'splits/train_query_ids.txt', 'splits/validation_query_ids.txt', 'splits/split_metadata.json'))]
    # Lock implementation provenance as well as data/model parameters.
    inputs.extend(p for directory in ('src', 'experiments') for p in (ROOT / directory).rglob('*.py'))
    lock = dict(dataset='SciFact', dataset_dir=str(args.dataset_dir.resolve()), split_information={k: v for k, v in split.items() if k not in ('train_ids', 'validation_ids')},
                train_ids=train, validation_ids=validation, test_ids=test_ids, test_query_count=300,
                candidate_pool_size=100, budget_grid=list(BUDGET_GRID), target_budgets=list(TARGETS), calibration_tolerance=.05,
                cross_encoder_model=DEFAULT_MODEL_NAME, dense_model=DEFAULT_MODEL, random_seeds=list(SEEDS),
                environment=environment_versions(), cross_encoder_configuration={'model_name': DEFAULT_MODEL_NAME, 'top_k': 10, 'predict_configuration': 'unchanged shared CrossEncoderReranker defaults'},
                configurations=configurations, sanity=sanity, gates=gates,
                checkpoint_information=checkpoint_information,
                historical_reproduction_audit=historical_audit,
                models={name: str(path) for name, path in models.items()},
                input_sha256={str(path): sha256(path) for path in inputs}, phase1_sha256=original_phase1,
                statistical_protocol='Paired two-sided Wilcoxon; global Holm; predeclared comparisons in paired_statistics; Random query mean over five seeds',
                locked=True, test_used_for_calibration=False,
                timing='Existing warm-up then measured pair preparation, inference, sorting; seconds; loading and warm-up excluded')
    if phase1_hashes() != original_phase1:
        raise ValueError('frozen Phase 1 files changed')
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'budget_calibration.csv', configurations, CALIBRATION_FIELDS)
    with LOCK.open('x', encoding='utf-8') as handle:
        json.dump(lock, handle, indent=2, allow_nan=False)
    save_summary({**lock, 'status': 'CALIBRATED_TEST_PENDING', 'locked_configuration_sha256': sha256(LOCK),
                  'calibration_failures': [r for r in configurations if r['calibration_status'] == 'FAILED'],
                  'final_test_results': None, 'statistical_results': None,
                  'protocol_notes': protocol_notes(), 'known_limitations': ['Final TEST has not run.']})
    print(json.dumps({'calibration': configurations, 'sanity': sanity}, indent=2))


def protocol_notes() -> list[str]:
    """State implemented contribution and distinguish frozen protocol branches."""
    return ['G2 and multivariate feasibility are different analyses; multivariate results do not replace G2.',
            'PACE-EF uses Evidence Frontloading + fixed Prefix-K; no adaptive PACE budget policy.',
            'SAGE-SLO and PoolAdapt are adaptive-budget methods using the existing RRF prefix.',
            'TEST was not used for calibration or model selection.',
            'Random baseline uses 5 seeds in Phase 2 comparison, differing from the single-seed Random reported in the frozen Phase 1 baseline.',
            'Compression ratio = 1 - average reranked pairs / 100.',
            'Random quality SD is sample SD across seeds; percentile latencies in aggregate rows are means of seed-level percentiles.']


def validate_lock(lock: dict[str, Any]) -> None:
    """Fail closed if parameters, inputs, source, or frozen Phase 1 changed."""
    if not lock.get('locked') or sha256(LOCK) != FROZEN_LOCK_SHA256:
        raise ValueError('configuration lock was modified')
    calibration = LOCK.with_name('budget_calibration.csv')
    if sha256(calibration) != FROZEN_CALIBRATION_SHA256:
        raise ValueError('frozen budget calibration changed')
    validate_provenance_amendment(lock)
    for original, digest in lock['input_sha256'].items():
        path = locked_path(original, lock['dataset_dir'], ROOT) if lock.get('dataset_dir') else Path(original)
        relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else ''
        if relative.startswith(('src/', 'experiments/')):
            # Historical source hashes are retained in the original lock. The
            # reviewed source manifest above is the authoritative current code.
            continue
        current = sha256(path)
        if current != digest:
            raise ValueError(f'locked input changed: {original}')
    normalize_paths = lambda mapping: {key.replace('\\', '/'): value for key, value in mapping.items()}
    if normalize_paths(phase1_hashes()) != normalize_paths(lock['phase1_sha256']):
        raise ValueError('frozen Phase 1 artifacts changed')
    if lock.get('environment'):
        actual = environment_versions()
        actual['torch'] = actual['torch'].split('+')[0]
        if lock['environment'] != actual:
            raise ValueError('runtime package versions differ from the validation lock')
    if lock['random_seeds'] != list(SEEDS) or len(lock['test_ids']) != 300:
        raise ValueError('invalid final TEST population/seeds')


def test_run(args: argparse.Namespace) -> None:
    """Run every applicable locked point on the same 300 TEST queries."""
    device = getattr(args, 'device', None)
    if device is not None and device.split(':')[0] == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; CPU fallback is forbidden')
        print(f'CUDA GPU: {torch.cuda.get_device_name(0)}', flush=True)
    output = getattr(args, 'output_dir', None) or OUT
    fresh, resume = getattr(args, 'fresh_output', False), getattr(args, 'resume', False)
    isolated = getattr(args, 'output_dir', None) is not None
    if fresh and resume:
        raise ValueError('--fresh-output and --resume are mutually exclusive')
    if (fresh or resume or (device and device.startswith('cuda'))) and not isolated:
        raise ValueError('GPU/fresh/resume runs require --output-dir separate from interrupted CPU artifacts')
    if isolated and (output.resolve() == OUT.resolve() or not (fresh or resume)):
        raise ValueError('use a separate output directory and exactly one of --fresh-output or --resume')
    lock = read_json(LOCK)
    validate_lock(lock)
    dataset = locked_path(lock['dataset_dir'], lock['dataset_dir'], ROOT)
    # Preserve callers whose synthetic/local dataset directory already matches.
    if args.dataset_dir.resolve() == Path(lock['dataset_dir']).resolve():
        dataset = args.dataset_dir.resolve()
    if args.dataset_dir.resolve() != dataset.resolve():
        raise ValueError('TEST dataset directory differs from the validation-locked dataset')
    if (output / 'comparative_results.csv').exists() and not (isolated and resume):
        raise ValueError('final TEST output already exists; refusing to overwrite it')
    if isolated and resume and (output / 'phase2_summary.json').exists() and read_json(output / 'phase2_summary.json').get('status') == 'COMPLETE':
        raise ValueError('GPU run already COMPLETE; no evaluation needs to be resumed')
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / 'test_predictions.jsonl'
    if not isolated and predictions_path.exists():
        raise ValueError('partial TEST predictions exist; preserve them before starting a new run')
    if isolated and not resume and (predictions_path.exists() or (output / 'execution_manifest.json').exists()):
        raise FileExistsError('fresh output directory already contains a run')
    corpus, queries, _ = load_beir_dataset(args.dataset_dir)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, model_name=lock['dense_model'])
    models = {name: load_model(locked_path(path, lock['dataset_dir'], ROOT), expected_name=name) for name, path in lock['models'].items()}
    reranker = CrossEncoderReranker(model_name=lock['cross_encoder_model'], **({'device': device} if device is not None else {}))
    reranker.warm_up()
    journal = None
    if isolated:
        import torch
        source_manifest = read_json(REVIEWED_SOURCE_MANIFEST)
        amendment = read_json(PROVENANCE_AMENDMENT)
        execution = dict(lock_sha256=sha256(LOCK),
                         original_lock_sha256=FROZEN_LOCK_SHA256,
                         input_sha256={locked_path(path, lock['dataset_dir'], ROOT).relative_to(ROOT).as_posix(): sha256(locked_path(path, lock['dataset_dir'], ROOT)) for path in lock.get('input_sha256', {})},
                         calibration_sha256=FROZEN_CALIBRATION_SHA256,
                         provenance_amendment_sha256=sha256(PROVENANCE_AMENDMENT),
                         reviewed_execution_commit=amendment['reviewed_execution_commit'],
                         reviewed_source_manifest_sha256=git_blob_sha256(REVIEWED_SOURCE_MANIFEST),
                         decision_state_projection_sha256=canonical_json_digest(decision_projection(lock)),
                         phase1_sha256=phase1_hashes(),
                         environment=execution_versions(), requested_device=device,
                         effective_cross_encoder_device=reranker.effective_device,
                         gpu_name=torch.cuda.get_device_name(0) if reranker.effective_device.startswith('cuda') else None,
                         cuda_runtime=torch.version.cuda,
                         source_manifest_files=len(source_manifest['files']),
                         timing=lock.get('timing'), cuda_synchronization='before reranking, before and after predict; warm-up drained outside timers')
        journal = PredictionJournal(predictions_path, execution, expected_keys(lock), resume=resume)
        print(f'GPU journal: {predictions_path}; completed records: {len(journal.rows)}/{len(journal.allowed)}', flush=True)
    groups: dict[tuple[str, int, int | None], list[dict[str, Any]]] = defaultdict(list)
    if journal:
        for row in journal.rows.values():
            groups[(row['method'], row['target_budget'], row['seed'])].append(dict(row))
    population = lock['test_ids']
    with predictions_path.open('a' if journal else 'x', encoding='utf-8') as handle:
        for number, qid in enumerate(population, 1):
            if journal and all(key in journal.rows for key in journal.allowed if key[0] == str(qid)):
                print(f'TEST {number}/300 already complete (resume)', flush=True)
                continue
            query = queries[qid]
            sparse, dense_hits = bm25.retrieve(query, 100), dense.retrieve(query, 100)
            fused = fuse_ranked_lists({'bm25': sparse, 'dense': dense_hits}, top_n=100)
            pool = [dict(corpus[str(c['id'])], **c) for c in fused]
            validate_pool(pool)
            features = frozen_features(query, pool, sparse, dense_hits, corpus, bm25)
            logits = {name: PoolAdaptSelector(model).logits(features) if name == 'PoolAdapt' else model_logits(model, features, FROZEN_FEATURES) for name, model in models.items()}
            ordered = pace_order(query, pool, bm25)
            for config in lock['configurations']:
                if config['calibration_status'] == 'FAILED':
                    continue
                method, target = config['method'], config['target_budget']
                budget = int(predict_budgets(logits[method][None, :], config['temperature'], config['budget_bias_lambda'])[0]) if method in models else target
                for seed in (SEEDS if method == 'Random' else (None,)):
                    if method == 'Random':
                        selected = select_random(pool, budget, seed=seed, query_id=qid)
                    elif method == 'PACE-EF':
                        selected = select_prefix(ordered, budget)
                    elif method == 'PoolAdapt':
                        selected = select_rrf_prefix(pool, budget)
                    elif method == 'SAGE-SLO':
                        selected = select_sage_slo(pool, budget)
                    else:
                        selected = select_fixed_prefix(pool, budget)
                    if len(selected) != budget or len({c['id'] for c in selected}) != budget:
                        raise ValueError('selector did not produce the exact predicted unique budget')
                    key = (str(qid), method, target, seed)
                    if journal and key in journal.rows:
                        saved = journal.rows[key]
                        if saved['selected_ids'] != [str(c['id']) for c in selected] or saved['pool_ids'] != [str(c['id']) for c in pool] or saved['frontloaded_ids'] != [str(c['id']) for c in ordered]:
                            raise ValueError('resume candidate selection differs from saved GPU record')
                        continue
                    ranked = reranker.rerank(query, selected, top_k=10)
                    row = dict(query_id=qid, method=method, target_budget=target, seed=seed,
                               reranked_pairs=budget, reranking_latency=reranker.last_latency_rerank_seconds,
                               document_ids=[str(c['id']) for c in ranked], selected_ids=[str(c['id']) for c in selected],
                               pool_ids=[str(c['id']) for c in pool], frontloaded_ids=[str(c['id']) for c in ordered])
                    if journal:
                        journal.append(row)
                    else:
                        handle.write(json.dumps(row) + '\n')
                    groups[(method, target, seed)].append(row)
            handle.flush()
            print(f'TEST {number}/300 complete', flush=True)
    if journal and set(journal.rows) != journal.allowed:
        raise ValueError('GPU journal is incomplete; TEST labels must remain unopened')
    # Load evaluation labels only after all selector/reranker calls have finished.
    relevance = load_qrels(args.dataset_dir / 'qrels/test.tsv')
    verify_population(list(relevance), population)
    for rows in groups.values():
        verify_population([r['query_id'] for r in rows], population)
        for row in rows:
            qrels = relevance[row['query_id']]
            row.update({'nDCG@10': ndcg_at_k(row['document_ids'], qrels), 'Recall@10': recall_at_k(row['document_ids'], qrels), 'MRR@10': mrr_at_k(row['document_ids'], qrels)})
    configs = {(r['method'], r['target_budget']): r for r in lock['configurations']}
    full_latency = float(np.mean([r['reranking_latency'] for r in groups[('Full Rerank', 100, None)]]))
    results, seeds = [], []
    for (method, target, seed), rows in groups.items():
        result = aggregate(rows, dict(configs[(method, target)], seed=seed), full_latency)
        (seeds if method == 'Random' else results).append(result)
    for target in TARGETS:
        results.append(random_aggregate([r for r in seeds if r['target_budget'] == target]))
    matched = {(r['method'], r['target_budget']) for r in results if r['matched_budget']}
    statistics = paired_statistics(groups, matched, population)
    for config in lock['configurations']:
        if config['calibration_status'] == 'FAILED':
            results.append({**{field: None for field in RESULT_FIELDS}, 'method': config['method'],
                            'target_budget': config['target_budget'], 'budget_type': 'adaptive',
                            'calibration_status': 'FAILED', 'matched_budget': False,
                            'validation_average_budget': config['actual_average_budget']})
    if len({(r['method'], r['target_budget']) for r in results}) != len(results):
        raise ValueError('duplicate operating-point result keys')
    validate_lock(lock)
    for name in ('comparative_results.csv', 'quality_cost_latency.csv'):
        write_csv(output / name, results, RESULT_FIELDS)
    write_csv(output / 'random_seed_results.csv', seeds, RESULT_FIELDS)
    query_rows = [r for rows in groups.values() for r in rows]
    write_csv(output / 'test_query_metrics.csv', query_rows, ('method', 'target_budget', 'seed', 'query_id', *METRICS, 'reranked_pairs', 'reranking_latency'))
    write_csv(output / 'statistical_analysis.csv', statistics, tuple(statistics[0]) if statistics else ('metric', 'p_Holm'))
    summary = {**lock, 'locked_configuration_sha256': sha256(LOCK)} if isolated else read_json(SUMMARY)
    summary.update(status='COMPLETE', final_test_results=results, random_seed_results=seeds,
                   statistical_results=statistics, protocol_notes=protocol_notes(),
                   quality_cost_latency_findings=tradeoff_findings(results),
                   test_query_count=300, test_qrels_used_after_selection=True, phase1_unchanged=True,
                   experimental_configuration='FROZEN after final TEST; no retraining or retuning',
                   leakage_audit={'training_queries': 647, 'calibration_queries': 162, 'evaluation_queries': 300,
                                  'checkpoints_retrained': False, 'test_used_for_temperature_or_lambda': False,
                                  'test_used_for_training': False, 'test_used_for_feature_selection': False,
                                  'test_used_for_method_inclusion': False, 'test_used_for_hypothesis_selection': False,
                                  'qrels_exposed_to_selectors': False, 'phase1_modified': False},
                   test_budget_drift=[r for r in results if r['calibration_status'] == 'SUCCESS' and not r['matched_budget']],
                   known_limitations=['SciFact only; adapted baselines; no answer generation or end-to-end SLO.',
                                      'The frozen SAGE checkpoint differs from historical development predictions; checkpoint metadata/recipe verification passed. The cause is not established from available historical provenance.',
                                      'Reranking latency excludes selector and retrieval cost.',
                                      'Validation-matched policies can have TEST workload drift; such points are excluded from matched tests.',
                                      'No statistically significant difference detected does not establish equivalence.'])
    if isolated:
        summary['execution_manifest_sha256'] = journal.run_digest
        summary_path = output / 'phase2_summary.json'
        temporary = summary_path.with_suffix('.json.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            handle.write(json.dumps(summary, indent=2, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, summary_path)
    else:
        save_summary(summary)
    print('Final TEST artifacts written to', output)


def main() -> None:
    """Dispatch explicit calibration and locked TEST stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('preflight', 'calibrate', 'test'))
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/scifact')
    parser.add_argument('--pooladapt-model', type=Path, default=ROOT / 'results/phase2/checkpoints/pooladapt.pkl')
    parser.add_argument('--sage-model', type=Path, default=ROOT / 'results/phase2/checkpoints/sage_slo.pkl')
    parser.add_argument('--device', default=None, help='Explicit CrossEncoder device; CUDA never falls back to CPU')
    parser.add_argument('--output-dir', type=Path, help='Separate GPU-only run directory')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fresh-output', action='store_true', help='Start a new isolated prediction journal')
    mode.add_argument('--resume', action='store_true', help='Resume only this isolated journal with identical execution provenance')
    args = parser.parse_args()
    if args.stage == 'preflight':
        preflight_run(args)
    elif args.stage == 'calibrate':
        if args.pooladapt_model is None or args.sage_model is None:
            parser.error('calibrate requires both existing fitted checkpoint paths')
        calibrate_run(args)
    else:
        test_run(args)


if __name__ == '__main__':
    main()
