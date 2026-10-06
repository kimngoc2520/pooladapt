"""Checkpoint round-trip, exact recipe, and TRAIN-only persistence tests."""
from __future__ import annotations

import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments import persist_phase2_checkpoints as persistence
from experiments.run_phase2_evaluation import load_model
from experiments.run_pooladapt import build_model as build_pooladapt
from experiments.run_sage_slo import build_model as build_sage
from src.pooladapt.checkpoints import (checkpoint_metadata, load_checkpoint, metadata_path,
    pipeline_configuration, save_checkpoint, verify_reload)
from src.pooladapt.selector import BUDGET_GRID, FROZEN_FEATURES


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.random.default_rng(42).normal(size=(25, 13)).tolist()
        self.x[0][0] = np.nan
        self.y = list(BUDGET_GRID) * 5
        self.ids = [f'train{i}' for i in range(25)]

    def test_exact_existing_sage_recipe(self) -> None:
        original = Pipeline((('imputer', SimpleImputer(strategy='median')),
                             ('forest', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'))))
        self.assertEqual(pipeline_configuration(build_sage()), pipeline_configuration(original))

    def test_exact_existing_pooladapt_recipe(self) -> None:
        original = Pipeline((('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                             ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))))
        self.assertEqual(pipeline_configuration(build_pooladapt()), pipeline_configuration(original))

    def test_both_roundtrips_and_deterministic_state_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            for name, builder in (('SAGE-SLO', build_sage), ('PoolAdapt', build_pooladapt)):
                model = builder().fit(self.x, self.y)
                path = Path(temp) / f'{name}.pkl'
                meta = checkpoint_metadata(model, name, self.ids, {})
                report = save_checkpoint(model, path, meta, self.x[:5], self.ids[:5])
                reloaded = load_checkpoint(path, expected_name=name)
                evaluator_model = load_model(path, expected_name=name)
                self.assertIsNot(model, reloaded)
                np.testing.assert_array_equal(model.predict(self.x), reloaded.predict(self.x))
                np.testing.assert_array_equal(model.predict_proba(self.x), reloaded.predict_proba(self.x))
                np.testing.assert_array_equal(model.predict_proba(self.x), evaluator_model.predict_proba(self.x))
                self.assertEqual(path.read_bytes(), pickle.dumps(model, protocol=5))
                self.assertEqual(report['reload_verification']['status'], 'PASSED')
                self.assertEqual(report['reload_verification']['max_probability_absolute_difference'], 0)
                self.assertEqual(report['feature_order'], list(FROZEN_FEATURES))
                self.assertEqual(report['target_definition']['tau'], .99)
                self.assertFalse(report['validation_used_for_fitting'])
                self.assertFalse(report['test_accessed'])
                with self.assertRaises(FileExistsError):
                    save_checkpoint(model, path, meta, self.x[:5], self.ids[:5])

    def test_reload_failure_is_loud_and_no_verified_metadata_is_written(self) -> None:
        model = build_pooladapt().fit(self.x, self.y)
        different = build_pooladapt().fit(self.x, list(reversed(self.y)))
        with self.assertRaises(AssertionError):
            verify_reload(model, different, self.x, self.ids, self.ids)
        with self.assertRaises(ValueError):
            verify_reload(model, model, self.x, self.ids, self.ids)
        with self.assertRaises(ValueError):
            verify_reload(model, different, self.x[:1], ['validation'], self.ids)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.pkl'
            with patch('src.pooladapt.checkpoints.verify_reload', side_effect=AssertionError('mismatch')):
                with self.assertRaises(AssertionError):
                    save_checkpoint(model, path, checkpoint_metadata(model, 'PoolAdapt', self.ids, {}), self.x[:1], self.ids[:1])
            self.assertFalse(metadata_path(path).exists())

    def test_checkpoint_checksum_and_feature_order_are_enforced(self) -> None:
        model = build_pooladapt().fit(self.x, self.y)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.pkl'
            save_checkpoint(model, path, checkpoint_metadata(model, 'PoolAdapt', self.ids, {}), self.x[:1], self.ids[:1])
            with self.assertRaises(ValueError):
                load_checkpoint(path, expected_name='SAGE-SLO')
            meta_path = metadata_path(path)
            meta = json.loads(meta_path.read_text())
            meta['feature_order'] = list(reversed(FROZEN_FEATURES))
            meta_path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_checkpoint(path)
            meta['feature_order'] = list(FROZEN_FEATURES)
            meta_path.write_text(json.dumps(meta))
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                load_checkpoint(path)


class TrainOnlyPersistenceTests(unittest.TestCase):
    def write_table(self, path: Path, rows: list[dict]) -> None:
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def test_held_out_rows_cannot_enter_features_or_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [dict(query_id='train', **{name: '2' for name in FROZEN_FEATURES}),
                    dict(query_id='validation', **{name: 'INVALID' for name in FROZEN_FEATURES})]
            for name in ('pool_features.csv', 'query_features.csv'):
                self.write_table(root / name, rows)
            self.write_table(root / 'targets.csv', [{'query_id': 'train', 'M_star': '20', 'tau': '.99'},
                                                   {'query_id': 'validation', 'M_star': 'INVALID', 'tau': 'INVALID'}])
            x, y = persistence.training_inputs(root, root / 'targets.csv', ['train'])
            self.assertEqual(x, [[2.] * 13]); self.assertEqual(y, [20])

    def test_missing_train_labels_stop_rather_than_guess(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'labels.csv'
            self.write_table(path, [{'query_id': 'validation', 'M_star': '10'}])
            with self.assertRaises(ValueError):
                persistence.filtered_table(path, ['train'])

    def test_population_recovery_requires_exact_647_train(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'train.json'; lock = Path(temp) / 'ids.txt'
            artifact = {'split': 'train', 'num_queries': 647, 'methods': {'full_rerank': {'predictions': [{'query_id': str(i)} for i in range(647)]}}}
            path.write_text(json.dumps(artifact))
            actual = persistence.train_population(path, lock)
            self.assertEqual(len(actual), 647)
            self.assertEqual(actual, sorted(actual))
            lock.write_text('different\n')
            with self.assertRaises(ValueError):
                persistence.train_population(path, lock)
            artifact['split'] = 'validation'; path.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):
                persistence.train_population(path, Path(temp) / 'absent.txt')

    def test_existing_checkpoint_prevents_any_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'sage_slo.pkl').write_bytes(b'existing')
            with patch.object(persistence, 'build_sage') as sage, patch.object(persistence, 'build_pooladapt') as pool:
                with self.assertRaises(FileExistsError):
                    persistence.persist(root)
                sage.assert_not_called(); pool.assert_not_called()


if __name__ == '__main__':
    unittest.main()
