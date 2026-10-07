"""Readiness checks use synthetic inputs and empty GPU warm-ups, never TEST inference."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments import preflight_phase2_test as preflight


class PreflightTests(unittest.TestCase):
    def test_inventory_includes_indirect_hash_inputs_and_checkpoint_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = dict(dataset_dir=r'D:\project\data\scifact',
                input_sha256={r'D:\project\results\phase2\07_heuristic\heuristic_config.json': 'hash',
                              r'D:\project\src\retrieval\dense.py': 'hash'},
                models={'SAGE-SLO': r'D:\project\results\phase2\checkpoints\sage_slo.pkl'},
                phase1_sha256={r'results\phase1\baseline_evaluation.csv': 'hash'})
            paths = {p.relative_to(root).as_posix() for p in preflight.required_paths(lock, root)}
            self.assertEqual(paths, {'results/phase2/07_heuristic/heuristic_config.json',
                'results/phase2/checkpoints/sage_slo.pkl',
                'results/phase2/checkpoints/sage_slo.metadata.json',
                'results/phase1/baseline_evaluation.csv',
                'results/phase2/07_evaluation/locked_configurations.json',
                'results/phase2/07_evaluation/budget_calibration.csv',
                'results/phase2/07_evaluation/provenance_amendment.json',
                'results/phase2/07_evaluation/reviewed_source_manifest.json'})

    def test_missing_gate_fails_with_exact_filename_before_fitting_or_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root / 'results/phase2/07_evaluation'; directory.mkdir(parents=True)
            lock_path = directory / 'locked_configurations.json'
            lock_path.write_text(json.dumps(dict(dataset_dir=r'D:\project\data\scifact',
                input_sha256={r'D:\project\results\phase2\07_heuristic\heuristic_config.json': 'hash'},
                models={}, phase1_sha256={})))
            (directory / 'budget_calibration.csv').touch()
            with patch.object(preflight, 'ROOT', root), patch.object(preflight.evaluator, 'LOCK', lock_path), \
                 patch.object(preflight.evaluator, 'load_model') as load, patch.object(preflight.evaluator, 'validate_lock') as validate:
                with self.assertRaisesRegex(ValueError, 'heuristic_config.json: missing required file'):
                    preflight.audit_inputs(root / 'data/scifact', root / 'results/phase2/07_evaluation_gpu')
                load.assert_not_called(); validate.assert_not_called()

    def test_duplicate_development_feature_ids_fail_with_artifact_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root / 'results/phase2/01_candidate_logging'; directory.mkdir(parents=True)
            (directory / 'pool_features.csv').write_text('query_id,score\na,1\na,2\n')
            lock = dict(train_ids=['a'], validation_ids=['b'], test_ids=['test'])
            with self.assertRaisesRegex(ValueError, 'pool_features.csv'):
                preflight.check_features(lock, root)

    def test_unavailable_cuda_fails_without_any_pretrained_model_call(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict('sys.modules', {'torch': torch}), patch.object(preflight, 'CrossEncoderReranker') as cross:
            with self.assertRaisesRegex(ValueError, 'CUDA.*CPU fallback forbidden'):
                preflight.check_models({})
            cross.assert_not_called()

    def test_models_use_only_empty_warmups_and_explicit_cuda(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: 'Test GPU'),
                                __version__='test', version=SimpleNamespace(cuda='test'))
        dense = Mock(device='cuda:0'); dense.encode.return_value = np.ones((1, 4))
        dense_constructor = Mock(return_value=dense)
        cross = Mock(effective_device='cuda:0')
        lock = dict(dense_model='frozen-dense', cross_encoder_model='frozen-cross')
        with patch.dict('sys.modules', {'torch': torch, 'sentence_transformers': SimpleNamespace(SentenceTransformer=dense_constructor)}), \
             patch.object(preflight, 'CrossEncoderReranker', return_value=cross) as cross_constructor, \
             patch.object(preflight.evaluator, 'execution_versions'), patch('builtins.print'):
            preflight.check_models(lock)
        dense_constructor.assert_called_once_with('frozen-dense')
        dense.encode.assert_called_once_with([''], convert_to_numpy=True, show_progress_bar=False)
        cross_constructor.assert_called_once_with(model_name='frozen-cross', device='cuda')
        cross.warm_up.assert_called_once_with(); cross.rerank.assert_not_called()

    def test_pass_only_after_all_checks_and_failure_has_no_pass(self):
        with patch('sys.argv', ['preflight']), patch.object(preflight, 'audit_inputs', return_value={}), \
             patch.object(preflight, 'check_models'), redirect_stdout(io.StringIO()) as output:
            preflight.main()
            self.assertEqual(output.getvalue().strip(), 'PHASE 2 PREFLIGHT: PASS')
        with patch('sys.argv', ['preflight']), patch.object(preflight, 'audit_inputs', side_effect=ValueError('exact/artifact: missing')), \
             patch.object(preflight, 'check_models') as models, redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as error: preflight.main()
            self.assertEqual(error.exception.code, 1)
            self.assertIn('PHASE 2 PREFLIGHT: FAIL: exact/artifact: missing', output.getvalue())
            models.assert_not_called()
