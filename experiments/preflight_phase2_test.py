"""Read-only readiness audit for the already frozen Phase 2 TEST run.

This is separate from the evaluator's historical ``preflight`` stage. It never
fits, calibrates, reads predictions, writes experimental artifacts, or runs a
TEST query through a retrieval model, selector, or CrossEncoder.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_phase2_evaluation as evaluator
from experiments.run_pooladapt import feature_rows
from src.data import load_beir_dataset
from src.evaluation.phase2 import SEEDS, TARGETS, verify_population
from src.evaluation.phase2_execution import expected_keys, locked_path
from src.pooladapt.checkpoints import metadata_path
from src.pooladapt.selector import BUDGET_GRID, FROZEN_FEATURES
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval import BM25Retriever


def require(condition: bool, artifact: Path | str, detail: str) -> None:
    """Fail with the exact artifact and its inconsistent property."""
    if not condition:
        raise ValueError(f'{artifact}: {detail}')


def required_paths(lock: dict[str, Any], root: Path = ROOT) -> list[Path]:
    """Enumerate frozen artifacts; reviewed source is checked by the manifest."""
    paths = {root / 'results/phase2/evaluation/locked_configurations.json',
             root / 'results/phase2/evaluation/budget_calibration.csv'}
    for original in lock['input_sha256']:
        path = locked_path(original, lock['dataset_dir'], root)
        relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else ''
        if not relative.startswith(('src/', 'experiments/')):
            paths.add(path)
    paths.update((root / name for name in (
        'results/phase2/evaluation/provenance_amendment.json',
        'results/phase2/evaluation/reviewed_source_manifest.json')))
    paths.update(root / p.replace('\\', '/') for p in lock['phase1_sha256'])
    for original in lock['models'].values():
        path = locked_path(original, lock['dataset_dir'], root)
        paths.update((path, metadata_path(path)))
    ordered = sorted(path for path in paths if path.name not in (
        'provenance_amendment.json', 'reviewed_source_manifest.json'))
    return ordered + sorted(path for path in paths if path.name in (
        'provenance_amendment.json', 'reviewed_source_manifest.json'))


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read development CSV rows without collapsing duplicate query IDs."""
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def check_populations(lock: dict[str, Any], dataset: Path) -> None:
    """Check split/query/document availability without reading relevance scores."""
    train, validation, test = (lock[key] for key in ('train_ids', 'validation_ids', 'test_ids'))
    for name, population, size in [('train', train, 647), ('validation', validation, 162), ('test', test, 300)]:
        require(len(population) == size and len(set(population)) == size,
                evaluator.LOCK, f'{name} population must contain {size} unique IDs')
    require(not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test)),
            evaluator.LOCK, 'split populations overlap')
    for name, population in [('train', train), ('validation', validation)]:
        path = dataset / f'splits/{name}_query_ids.txt'
        ids = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        require(ids == population, path, 'query IDs/order differ from the lock')
    corpus, queries, _ = load_beir_dataset(dataset)
    require(len(corpus) >= 100, dataset / 'corpus.jsonl', 'fewer than 100 documents')
    for qid in train + validation + test:
        require(qid in queries, dataset / 'queries.jsonl', f'missing query_id={qid}')
    for split, population in [('train', train + validation), ('validation', validation), ('test', test)]:
        path = dataset / f'qrels/{split}.tsv'
        with path.open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            require({'query-id', 'corpus-id', 'score'} <= set(reader.fieldnames or []), path, 'qrels schema mismatch')
            query_ids = set()
            for row in reader:
                # Only IDs are inspected. No scores, quality metrics or TEST predictions.
                qid, document = row['query-id'], row['corpus-id']
                require(document in corpus, path, f'unknown corpus-id={document}')
                query_ids.add(qid)
        require(query_ids == set(population), path, 'query population differs from the locked split')
    split_metadata = evaluator.read_json(dataset / 'splits/split_metadata.json')
    for key, expected in lock['split_information'].items():
        if key == 'source_qrels_sha256':
            require(evaluator.sha256(dataset / 'qrels/train.tsv') == expected, dataset / 'qrels/train.tsv', 'source_qrels_sha256 differs from lock')
            continue  # prepare_split computes this return field; it is not persisted in split_metadata.json.
        require(split_metadata.get(key) == expected, dataset / 'splits/split_metadata.json', f'{key} differs from lock')
    # The exact runtime BM25 constructor uses only corpus text; no serialized index.
    BM25Retriever(corpus)
    print(f'Population/BEIR/BM25: TRAIN={len(train)}, VALIDATION={len(validation)}, TEST={len(test)}; corpus={len(corpus)}')


def check_features(lock: dict[str, Any], root: Path = ROOT) -> None:
    """Verify 809 development rows and the exact frozen feature availability."""
    directory = root / 'results/phase2/01_candidate_logging'
    populations = lock['train_ids'] + lock['validation_ids']
    feature_indexes = []
    for name in ('pool_features.csv', 'query_features.csv'):
        path = directory / name
        columns, rows = read_rows(path)
        require('query_id' in columns, path, 'missing query_id column')
        try:
            verify_population([row['query_id'] for row in rows], populations, lock['test_ids'])
        except ValueError as error:
            raise ValueError(f'{path}: {error}') from error
        feature_indexes.append({row['query_id']: row for row in rows})
    merged_columns = set().union(*(set(next(iter(index.values()))) for index in feature_indexes))
    require(set(FROZEN_FEATURES) <= merged_columns, directory, f'missing features: {sorted(set(FROZEN_FEATURES) - merged_columns)}')
    try:
        feature_rows(*feature_indexes, populations)
    except (ValueError, KeyError) as error:
        raise ValueError(f'{directory}: invalid frozen feature value: {error}') from error
    metadata = evaluator.read_json(directory / 'metadata.json')
    require(metadata['query_count'] == 809 and metadata['candidate_pool_size'] == 100,
            directory / 'metadata.json', 'development population/pool metadata mismatch')
    print('Features: 809 TRAIN/VALIDATION rows, frozen 13 features; TEST features are computed online')


def check_configurations(lock: dict[str, Any]) -> None:
    """Check locked operating points and stopped gates without recalibration."""
    require(lock['budget_grid'] == list(BUDGET_GRID) and lock['target_budgets'] == list(TARGETS), evaluator.LOCK, 'budget grid/targets differ')
    require(lock['random_seeds'] == list(SEEDS) and not lock['test_used_for_calibration'], evaluator.LOCK, 'seed/leakage metadata mismatch')
    expected = {(method, target) for method in ('PoolAdapt', 'SAGE-SLO', 'Fixed Prefix', 'PACE-EF', 'Random', 'Full Rerank')
                for target in ((100,) if method == 'Full Rerank' else TARGETS)}
    configs = lock['configurations']
    require(len(configs) == len(expected) and {(c['method'], c['target_budget']) for c in configs} == expected,
            evaluator.LOCK, 'missing/duplicate/unknown operating point')
    for config in configs:
        require(config['locked'], evaluator.LOCK, f'unlocked operating point {config["method"]}/{config["target_budget"]}')
        if config['method'] in ('PoolAdapt', 'SAGE-SLO') and config['calibration_status'] != 'FAILED':
            require(np.isfinite(config['temperature']) and config['temperature'] > 0 and np.isfinite(config['budget_bias_lambda']),
                    evaluator.LOCK, f'invalid T/lambda for {config["method"]}/{config["target_budget"]}')
    for gate, expected_status in [('G2', 'FAILED'), ('G3a', 'NOT_EVALUATED'), ('G3b', 'NOT_EVALUATED')]:
        require(lock['gates'][gate] == expected_status, evaluator.LOCK, f'{gate} differs from frozen gate')
    print(f'Configurations: {len(configs)} locked points; {len(expected_keys(lock))} expected prediction records')


def check_checkpoints(lock: dict[str, Any], root: Path = ROOT) -> None:
    """Load the actual trusted fitted pipelines; never construct or fit recipes."""
    for name, original in lock['models'].items():
        path = locked_path(original, lock['dataset_dir'], root)
        try:
            model = evaluator.load_model(path, expected_name=name)
            metadata = evaluator.read_json(metadata_path(path))
            require(metadata == lock['checkpoint_information'][name]['metadata'], metadata_path(path), 'sidecar differs from locked checkpoint metadata')
            require(metadata['training_query_ids'] == lock['train_ids'] and metadata['n_training_queries'] == 647 and metadata['fit_count'] == 1,
                    metadata_path(path), 'TRAIN population/order or fit count differs')
            require(metadata['feature_order'] == list(FROZEN_FEATURES) and metadata['random_seed'] == 42,
                    metadata_path(path), 'feature order/seed differs')
            require(not metadata['validation_used_for_fitting'] and not metadata['validation_used_for_model_selection'] and not metadata['test_accessed'],
                    metadata_path(path), 'training leakage metadata mismatch')
            require(list(model.classes_) == list(BUDGET_GRID), path, 'fitted budget class order differs')
        except (ValueError, KeyError, ImportError) as error:
            raise ValueError(f'{path}: {error}') from error
        print(f'{name}: frozen fitted checkpoint loaded; no fitting or prediction')


def audit_inputs(dataset: Path, output: Path) -> dict[str, Any]:
    """Check all static dependencies read-only, before initializing pretrained models."""
    require(evaluator.LOCK.is_file(), evaluator.LOCK, 'missing required file')
    lock = evaluator.read_json(evaluator.LOCK)
    paths = required_paths(lock, ROOT)
    for path in paths:
        require(path.is_file(), path, 'missing required file')
    versions = evaluator.environment_versions()
    for package, expected in lock['environment'].items():
        actual = versions[package].split('+')[0] if package == 'torch' else versions[package]
        require(actual == expected, f'runtime:{package}', f'expected {expected}, found {actual}')
    evaluator.validate_lock(lock)  # Same frozen checks/hashes as the actual evaluator.
    require(dataset.resolve() == locked_path(lock['dataset_dir'], lock['dataset_dir'], ROOT), dataset, 'dataset location differs from lock')
    require(output.resolve() != evaluator.OUT.resolve(), output, 'GPU outputs must be separate from CPU artifacts')
    for name in ('test_predictions.jsonl', 'execution_manifest.json', 'comparative_results.csv'):
        require(not (output / name).exists(), output / name, 'fresh-run output already exists; preserve it and use resume separately')
    writable = output.resolve()
    while not writable.exists():
        writable = writable.parent
    require(writable.is_dir() and os.access(writable, os.W_OK), writable, 'output parent is not writable')
    check_populations(lock, dataset)
    check_features(lock, ROOT)
    check_configurations(lock)
    check_checkpoints(lock, ROOT)
    print(f'Frozen integrity: {len(paths)} required artifact/source files checked using evaluator rules')
    return lock


def check_models(lock: dict[str, Any]) -> None:
    """Verify CUDA and pretrained models using empty warm-ups only."""
    import torch
    require(torch.cuda.is_available(), 'CUDA', 'torch.cuda.is_available() is False; CPU fallback forbidden')
    print(f'CUDA GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA runtime={torch.version.cuda}')
    from sentence_transformers import SentenceTransformer
    try:
        dense = SentenceTransformer(lock['dense_model'])
        embedding = np.asarray(dense.encode([''], convert_to_numpy=True, show_progress_bar=False))
        require(embedding.ndim == 2 and embedding.shape[0] == 1 and np.isfinite(embedding).all(), lock['dense_model'], 'empty dense warm-up returned invalid embeddings')
        print(f'Dense model: loaded, empty warm-up passed; effective device={dense.device}')
    except Exception as error:
        raise RuntimeError(f'Dense model {lock["dense_model"]}: {error}') from error
    try:
        reranker = CrossEncoderReranker(model_name=lock['cross_encoder_model'], device='cuda')
        reranker.warm_up()
        require(reranker.effective_device.startswith('cuda'), lock['cross_encoder_model'], 'effective CrossEncoder device is not CUDA')
    except Exception as error:
        raise RuntimeError(f'CrossEncoder {lock["cross_encoder_model"]}: {error}') from error
    evaluator.execution_versions()  # Detect missing supporting distributions used by the journal.


def main() -> None:
    """Print PASS only after both input and CUDA/model readiness checks succeed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/scifact')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/phase2/evaluation/final_gpu')
    args = parser.parse_args()
    try:
        lock = audit_inputs(args.dataset_dir, args.output_dir)
        check_models(lock)
    except Exception as error:
        print(f'PHASE 2 PREFLIGHT: FAIL: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    print('PHASE 2 PREFLIGHT: PASS', flush=True)


if __name__ == '__main__':
    main()
