"""Persistence for the two existing fitted sklearn Phase 2 pipelines.

Pickle checkpoints are trusted local artifacts, including fitted preprocessing.
This module performs no fitting, calibration, feature selection, or evaluation.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.pipeline import Pipeline

from src.pooladapt.selector import BUDGET_GRID, FROZEN_FEATURES

CHECKPOINT_FORMAT = 'Python pickle protocol 5; complete fitted sklearn Pipeline'
MODEL_NAMES = ('SAGE-SLO', 'PoolAdapt')


def metadata_path(path: Path) -> Path:
    """Locate the JSON sidecar for a fitted pipeline checkpoint."""
    return path.with_suffix('.metadata.json')


def file_digest(path: Path) -> str:
    """Hash serialized state and source artifacts for provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pipeline_configuration(model: Pipeline) -> dict[str, Any]:
    """Record exact classes and all effective parameters, including defaults."""
    def json_parameter(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return {'float': 'NaN' if math.isnan(value) else str(value)}
        if isinstance(value, dict):
            return {key: json_parameter(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_parameter(item) for item in value]
        return value
    return {name: {'class': f'{type(step).__module__}.{type(step).__name__}',
                   'parameters': json_parameter(step.get_params(deep=False))} for name, step in model.steps}


def checkpoint_metadata(model: Pipeline, name: str, training_ids: Sequence[str],
                        input_hashes: Mapping[str, str]) -> dict[str, Any]:
    """Describe the existing TRAIN-only recipe, feature order, and target."""
    if name not in MODEL_NAMES or not training_ids or len(training_ids) != len(set(training_ids)):
        raise ValueError('expected a supported model and unique TRAIN IDs')
    config = pipeline_configuration(model)
    classifier_name = 'forest' if name == 'SAGE-SLO' else 'classifier'
    return {'model_name': name, 'training_split': 'SciFact Phase 2 TRAIN; seed-42 80/20 original TRAIN partition',
            'n_training_queries': len(training_ids), 'training_query_ids': list(training_ids),
            'feature_names': list(FROZEN_FEATURES), 'feature_order': list(FROZEN_FEATURES),
            'target_definition': {'name': 'oracle-derived M_star', 'tau': .99,
                                  'rule': 'Smallest K in budget grid with nDCG@10(K) >= 0.99 * nDCG@10(Full Rerank); fallback 100',
                                  'source': 'existing mstar_labels_tau099.csv; TRAIN rows only'},
            'budget_grid': list(BUDGET_GRID), 'random_seed': model.named_steps[classifier_name].random_state,
            'model_hyperparameters': config[classifier_name],
            'preprocessing_configuration': {key: value for key, value in config.items() if key != classifier_name},
            'pipeline_step_order': [name for name, _ in model.steps],
            'checkpoint_format': CHECKPOINT_FORMAT, 'pickle_protocol': 5,
            'python_version': platform.python_version(), 'sklearn_version': sklearn.__version__,
            'numpy_version': np.__version__, 'input_sha256': dict(input_hashes),
            'validation_used_for_fitting': False, 'validation_used_for_model_selection': False,
            'test_accessed': False, 'fit_count': 1}


def verify_reload(original: Pipeline, reloaded: Pipeline, features: Sequence[Sequence[float]],
                  sample_ids: Sequence[str], training_ids: Sequence[str]) -> dict[str, Any]:
    """Require identical classes, labels, and probabilities on TRAIN-only rows."""
    if original is reloaded:
        raise ValueError('reload must create a fresh model instance')
    if not sample_ids or len(sample_ids) != len(features) or len(sample_ids) != len(set(sample_ids)) or not set(sample_ids) <= set(training_ids):
        raise ValueError('reload verification must use unique TRAIN sample IDs only')
    x = np.asarray(features, dtype=float)
    np.testing.assert_array_equal(original.classes_, reloaded.classes_, err_msg='checkpoint classes differ')
    np.testing.assert_array_equal(original.predict(x), reloaded.predict(x), err_msg='checkpoint predictions differ')
    before, after = original.predict_proba(x), reloaded.predict_proba(x)
    np.testing.assert_array_equal(before, after, err_msg='checkpoint probabilities differ')
    return {'status': 'PASSED', 'split': 'TRAIN', 'n_queries': len(sample_ids),
            'query_ids': list(sample_ids), 'fresh_model_instance': True,
            'identical_class_order': True, 'identical_predictions': True,
            'identical_probabilities': True, 'max_probability_absolute_difference': float(np.max(np.abs(before - after)))}


def save_checkpoint(model: Pipeline, path: Path, metadata: Mapping[str, Any],
                    sample_features: Sequence[Sequence[float]], sample_ids: Sequence[str]) -> dict[str, Any]:
    """Save once, reload into a fresh pipeline, and fail loudly on any mismatch."""
    sidecar = metadata_path(path)
    if path.exists() or sidecar.exists():
        raise FileExistsError(f'refusing to overwrite existing checkpoint: {path}')
    payload = pickle.dumps(model, protocol=5)
    if payload != pickle.dumps(model, protocol=5):
        raise ValueError('serialization of fitted state is not deterministic')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(payload)
    with path.open('rb') as handle:
        reloaded = pickle.load(handle)
    report = verify_reload(model, reloaded, sample_features, sample_ids, metadata['training_query_ids'])
    saved = {**metadata, 'checkpoint_sha256': file_digest(path), 'reload_verification': report,
             'fitted_classes': [int(label) for label in model.classes_]}
    with sidecar.open('x', encoding='utf-8') as handle:
        json.dump(saved, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    return saved


def load_checkpoint(path: Path, expected_name: str | None = None) -> Pipeline:
    """Load a trusted checkpoint after integrity, feature, and recipe checks."""
    metadata = json.loads(metadata_path(path).read_text(encoding='utf-8'))
    if expected_name is not None and metadata['model_name'] != expected_name:
        raise ValueError('checkpoint model identity differs from the requested method')
    if metadata['checkpoint_sha256'] != file_digest(path):
        raise ValueError('checkpoint checksum mismatch')
    if metadata['feature_order'] != list(FROZEN_FEATURES) or metadata['budget_grid'] != list(BUDGET_GRID):
        raise ValueError('checkpoint features/order/budget grid differ from frozen methodology')
    if metadata['sklearn_version'] != sklearn.__version__ or metadata['numpy_version'] != np.__version__:
        raise ValueError('checkpoint runtime versions differ; refusing incompatible sklearn state')
    if metadata['reload_verification']['status'] != 'PASSED':
        raise ValueError('checkpoint reload verification did not pass')
    with path.open('rb') as handle:
        model = pickle.load(handle)
    config = pipeline_configuration(model)
    classifier_name = 'forest' if metadata['model_name'] == 'SAGE-SLO' else 'classifier'
    if config[classifier_name] != metadata['model_hyperparameters'] or {key: value for key, value in config.items() if key != classifier_name} != metadata['preprocessing_configuration']:
        raise ValueError('checkpoint fitted pipeline differs from metadata recipe')
    return model
