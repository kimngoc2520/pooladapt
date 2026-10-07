"""Synthetic tests for the Phase 2 provenance amendment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from experiments import run_phase2_evaluation as runner


class ProvenanceAmendmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / 'results/phase2/07_evaluation'
        self.out.mkdir(parents=True)
        self.dataset = self.root / 'data/scifact'
        self.dataset.mkdir(parents=True)
        self.source = self.root / 'src/example.py'
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b'print("reviewed")\n')
        self.paths = {
            'checkpoint': self.root / 'results/phase2/checkpoints/pooladapt.pkl',
            'checkpoint_metadata': self.root / 'results/phase2/checkpoints/pooladapt.metadata.json',
            'calibration': self.out / 'budget_calibration.csv',
            'dataset': self.dataset / 'splits/train_query_ids.txt',
            'gate': self.root / 'results/phase2/07_heuristic/heuristic_config.json',
            'feature': self.root / 'results/phase2/01_candidate_logging/pool_features.csv',
            'oracle': self.root / 'results/phase2/02_oracle/mstar_labels_tau099.csv',
            'phase1': self.root / 'results/phase1/frozen.csv',
        }
        for path in self.paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode())
        self.lock = {
            'locked': True, 'dataset': 'SciFact', 'dataset_dir': str(self.dataset),
            'input_sha256': {
                str(self.paths['checkpoint']): runner.sha256(self.paths['checkpoint']),
                str(self.paths['checkpoint_metadata']): runner.sha256(self.paths['checkpoint_metadata']),
                str(self.paths['calibration']): runner.sha256(self.paths['calibration']),
                str(self.paths['dataset']): runner.sha256(self.paths['dataset']),
                str(self.paths['gate']): runner.sha256(self.paths['gate']),
                str(self.paths['feature']): runner.sha256(self.paths['feature']),
                str(self.paths['oracle']): runner.sha256(self.paths['oracle']),
            },
            'phase1_sha256': {'results/phase1/frozen.csv': runner.sha256(self.paths['phase1'])},
            'checkpoint_information': {'PoolAdapt': {'metadata': {'identity': 'PoolAdapt'}}},
            'models': {'PoolAdapt': 'pooladapt.pkl'},
            'configurations': [{'method': 'Fixed Prefix', 'target_budget': 10, 'locked': True}],
            'candidate_pool_size': 100,
            'budget_grid': [10, 20, 30, 50, 100], 'target_budgets': [10, 20, 30, 50],
            'calibration_tolerance': .05, 'gates': {'G2': 'FAILED'},
            'cross_encoder_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'dense_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'cross_encoder_configuration': {},
            'random_seeds': [0, 1, 2, 3, 4], 'test_ids': ['q1'],
            'statistical_protocol': 'paired', 'test_used_for_calibration': False,
            'environment': {}, 'models': {},
        }
        self.reviewed_commit = 'a' * 40
        self.manifest = {'schema': 'pooladapt.reviewed-source-manifest.v1',
                         'reviewed_commit': self.reviewed_commit,
                         'files': [{'path': 'src/example.py', 'blob': 'blob-id',
                                    'sha256': runner.sha256(self.source)}]}
        self.manifest_path = self.out / 'reviewed_source_manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest) + '\n', encoding='utf-8')
        self.lock_path = self.out / 'locked_configurations.json'
        self.lock_path.write_text('{}\n', encoding='utf-8')
        self.calibration_path = self.out / 'budget_calibration.csv'
        self.calibration_path.write_bytes(b'calibration\n')
        self.lock['input_sha256'][str(self.calibration_path)] = runner.sha256(self.calibration_path)
        self.amendment_path = self.out / 'provenance_amendment.json'
        with patch.object(runner, 'ROOT', self.root), patch.object(runner, 'LOCK', self.lock_path):
            projection = runner.canonical_json_digest(runner.decision_projection(self.lock))
        self.amendment_path.write_text(json.dumps({
            'status': 'finalized / reviewed',
            'original_lock_sha256': 'lock-hash',
            'calibration_sha256': 'calibration-hash',
            'reviewed_execution_commit': self.reviewed_commit,
            'reviewed_source_manifest_sha256': runner.sha256(self.manifest_path),
            'decision_state_projection_sha256': projection,
        }) + '\n', encoding='utf-8')
        self.manifest_digest = runner.sha256(self.manifest_path)
        self.amendment_digest = runner.sha256(self.amendment_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def patched(self):
        stack = ExitStack()
        stack.enter_context(patch.object(runner, 'ROOT', self.root))
        stack.enter_context(patch.object(runner, 'LOCK', self.lock_path))
        stack.enter_context(patch.object(runner, 'PROVENANCE_AMENDMENT', self.amendment_path))
        stack.enter_context(patch.object(runner, 'REVIEWED_SOURCE_MANIFEST', self.manifest_path))
        stack.enter_context(patch.object(runner, 'FROZEN_LOCK_SHA256', 'lock-hash'))
        stack.enter_context(patch.object(runner, 'FROZEN_CALIBRATION_SHA256', 'calibration-hash'))
        stack.enter_context(patch.object(runner, 'git_output', side_effect=[
            'blob-id', 'blob-id', self.reviewed_commit,
            'head ' + self.reviewed_commit, '']))
        return stack

    def validate(self) -> None:
        with self.patched():
            runner.validate_provenance_amendment(self.lock)

    def test_correct_reviewed_source_passes(self):
        self.validate()

    def test_wrong_git_commit_fails(self):
        with self.patched() as stack:
            stack.enter_context(patch.object(runner, 'git_output', side_effect=[
                'blob-id', 'blob-id', 'wrong', 'head wrong']))
            with self.assertRaisesRegex(ValueError, 'provenance freeze commit'):
                runner.validate_provenance_amendment(self.lock)

    def test_modified_source_fails(self):
        self.source.write_bytes(b'changed\n')
        with self.patched() as stack:
            stack.enter_context(patch.object(runner, 'git_output', return_value='different'))
            with self.assertRaisesRegex(ValueError, 'reviewed source bytes differ'):
                runner.validate_provenance_amendment(self.lock)

    def test_missing_source_fails(self):
        self.source.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.validate()

    def test_modified_manifest_fails(self):
        self.manifest_path.write_text('{}\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'source manifest differs'):
            self.validate()

    def test_modified_amendment_fails(self):
        self.amendment_path.write_text('{}\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'pending'):
            self.validate()

    def test_line_endings_do_not_change_git_blob_check(self):
        self.source.write_bytes(b'print("reviewed")\r\n')
        self.validate()

    def test_frozen_input_and_phase1_drift_fail(self):
        cases = ['checkpoint', 'checkpoint_metadata', 'calibration', 'dataset', 'gate',
                 'feature', 'oracle', 'phase1']
        for name in cases:
            with self.subTest(name=name):
                path = self.paths[name]
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                try:
                    with self.patched():
                        with self.assertRaises(ValueError):
                            runner.validate_lock(self.lock)
                finally:
                    path.write_bytes(original)

    def test_decision_projection_drift_fails(self):
        mutated = dict(self.lock)
        mutated['target_budgets'] = [10, 20]
        with self.patched():
            with self.assertRaisesRegex(ValueError, 'projection'):
                runner.validate_provenance_amendment(mutated)

    def test_validation_does_not_require_test_qrels_or_quality(self):
        self.validate()
        self.assertFalse(any(path.name.startswith('test') for path in self.root.rglob('*')))


if __name__ == '__main__':
    unittest.main()
