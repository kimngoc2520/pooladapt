"""Reconstruct existing Phase 2 recipes once on recorded TRAIN IDs, then save.

This command never opens validation artifacts, TEST files, or qrels. It reads
only TRAIN IDs from the existing TRAIN artifact, then filters the existing
development feature/target tables to those IDs before creating any model input.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_pooladapt import build_model as build_pooladapt
from experiments.run_sage_slo import FEATURES, build_model as build_sage
from src.pooladapt.checkpoints import checkpoint_metadata, file_digest, load_checkpoint, metadata_path, save_checkpoint
from src.pooladapt.selector import BUDGET_GRID, FROZEN_FEATURES

OUTPUT = ROOT / 'results/phase2/checkpoints'


def train_population(artifact_path: Path, lock_path: Path) -> list[str]:
    """Recover the recorded 647 TRAIN IDs, preserving original sorted fit order."""
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    if artifact.get('split') != 'train' or artifact.get('num_queries') != 647:
        raise ValueError('expected the existing 647-query TRAIN artifact')
    ids = sorted(str(row['query_id']) for row in artifact['methods']['full_rerank']['predictions'])
    if len(ids) != 647 or len(set(ids)) != 647:
        raise ValueError('TRAIN artifact must contain exactly 647 unique IDs')
    if lock_path.exists():
        locked = [x.strip() for x in lock_path.read_text(encoding='utf-8').splitlines() if x.strip()]
        if locked != ids:
            raise ValueError('recorded TRAIN population differs from existing TRAIN lock/order')
    return ids


def filtered_table(path: Path, train_ids: list[str]) -> dict[str, dict[str, str]]:
    """Keep only TRAIN rows; never parse held-out feature values or targets."""
    allowed = set(train_ids)
    selected = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            qid = row['query_id']
            if qid not in allowed:
                continue
            if qid in selected:
                raise ValueError(f'duplicate TRAIN row in {path}: {qid}')
            selected[qid] = row
    if set(selected) != allowed:
        raise ValueError(f'missing TRAIN rows in {path}')
    return selected


def training_inputs(feature_dir: Path, target_path: Path, train_ids: list[str]) -> tuple[list[list[float]], list[int]]:
    """Reproduce existing ordered feature vectors and tau=.99 TRAIN targets."""
    if tuple(FEATURES) != FROZEN_FEATURES:
        raise ValueError('SAGE and PoolAdapt frozen feature orders differ')
    pool = filtered_table(feature_dir / 'pool_features.csv', train_ids)
    query = filtered_table(feature_dir / 'query_features.csv', train_ids)
    labels = filtered_table(target_path, train_ids)
    x, y = [], []
    for qid in train_ids:
        merged = {**pool[qid], **query[qid]}
        x.append([float(merged[name]) if merged.get(name, '') != '' else np.nan for name in FROZEN_FEATURES])
        label = int(labels[qid]['M_star'])
        if label not in BUDGET_GRID or float(labels[qid]['tau']) != .99:
            raise ValueError('TRAIN target differs from existing tau=.99 M* definition')
        y.append(label)
    return x, y


def frozen_phase1_hashes() -> dict[str, str]:
    """Byte-hash Phase 1 for integrity only; never interpret its TEST outputs."""
    return {str(path.relative_to(ROOT)): file_digest(path) for path in sorted((ROOT / 'results/phase1').rglob('*')) if path.is_file()}


def persist(output: Path = OUTPUT) -> dict[str, Any]:
    """Fit each recovered recipe once, save both, and verify fresh reloads."""
    paths = {'SAGE-SLO': output / 'sage_slo.pkl', 'PoolAdapt': output / 'pooladapt.pkl'}
    manifest = output / 'checkpoint_manifest.json'
    # Prevent any new fitting when an earlier checkpoint/run already exists.
    for path in [manifest, *paths.values(), *(metadata_path(p) for p in paths.values())]:
        if path.exists():
            raise FileExistsError(f'existing checkpoint/run found; refusing to refit: {path}')
    phase1 = frozen_phase1_hashes()
    train_artifact = ROOT / 'results/phase2/development_reranking/train/reranking_results.json'
    train_lock = ROOT / 'data/scifact/splits/train_query_ids.txt'
    feature_dir = ROOT / 'results/phase2/01_candidate_logging'
    target = ROOT / 'results/phase2/02_oracle/mstar_labels_tau099.csv'
    ids = train_population(train_artifact, train_lock)
    x, y = training_inputs(feature_dir, target, ids)
    inputs = [train_artifact, target, feature_dir / 'pool_features.csv', feature_dir / 'query_features.csv',
              ROOT / 'experiments/run_sage_slo.py', ROOT / 'experiments/run_pooladapt.py']
    if train_lock.exists():
        inputs.append(train_lock)
    provenance = {str(p.relative_to(ROOT)): file_digest(p) for p in inputs}
    report = {}
    for name, builder in (('SAGE-SLO', build_sage), ('PoolAdapt', build_pooladapt)):
        model = builder(seed=42)
        model.fit(x, y)  # Exactly one fit per method; all inputs are TRAIN only.
        metadata = checkpoint_metadata(model, name, ids, provenance)
        metadata['training_class_counts'] = {str(k): y.count(k) for k in BUDGET_GRID}
        metadata['training_order'] = 'sorted string query IDs, identical to existing Phase 2 TRAIN lock'
        saved = save_checkpoint(model, paths[name], metadata, x[:16], ids[:16])
        # Exercise the same validated loader used by the final evaluator.
        load_checkpoint(paths[name], expected_name=name)
        report[name] = {'checkpoint_path': str(paths[name]), 'metadata_path': str(metadata_path(paths[name])),
                        'n_training_queries': len(ids), 'fit_count': 1,
                        'checkpoint_sha256': saved['checkpoint_sha256'], 'reload_verification': saved['reload_verification']}
        print(f'{name}: fit once on {len(ids)} TRAIN queries; reload equivalence PASSED', flush=True)
    if frozen_phase1_hashes() != phase1:
        raise ValueError('frozen Phase 1 artifacts changed during checkpoint persistence')
    result = {'status': 'CHECKPOINTS_SAVED_CALIBRATION_NOT_RUN', 'models': report,
              'validation_used_for_fitting': False, 'validation_used_for_model_selection': False,
              'test_accessed': False, 'calibration_run': False, 'test_evaluation_run': False,
              'phase1_unchanged': True, 'phase1_sha256': phase1,
              'heuristic': {'G2': 'FAILED', 'G3a': 'NOT_EVALUATED', 'G3b': 'NOT_EVALUATED', 'reconstructed': False}}
    with manifest.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    return result


def main() -> None:
    """Persist only SAGE-SLO and PoolAdapt; no validation/TEST stage exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    persist(args.output_dir)


if __name__ == '__main__':
    main()
