"""Execution plumbing tests without weights or real TEST queries."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import ExitStack
import numpy as np

from experiments import run_phase2_evaluation as runner
from src.evaluation.phase2_execution import PredictionJournal, evaluation_key, expected_keys, locked_path
from src.evaluation.phase2 import sha256
from src.reranking.cross_encoder import CrossEncoderReranker


class GPUDeviceTests(unittest.TestCase):
    def torch(self, available=True):
        return SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(return_value=available),
            get_device_name=Mock(return_value='Test GPU'), synchronize=Mock()))

    def test_device_propagation_and_cuda_timing(self):
        torch = self.torch(); model = Mock(device='cuda:0'); model.predict.return_value = [1.]
        constructor = Mock(return_value=model)
        with patch.dict('sys.modules', {'torch': torch, 'sentence_transformers': SimpleNamespace(CrossEncoder=constructor)}):
            reranker = CrossEncoderReranker(device='cuda'); reranker.warm_up()
            torch.cuda.synchronize.reset_mock()
            with patch('src.reranking.cross_encoder.time.perf_counter', side_effect=[1., 2., 5., 8.]):
                reranker.rerank('q', [{'id': 'd', 'text': 'evidence'}])
            constructor.assert_called_once_with(reranker.model_name, device='cuda')
            self.assertEqual(torch.cuda.synchronize.call_count, 3)
            self.assertEqual(reranker.last_latency_ce_seconds, 3.)
            self.assertEqual(reranker.last_latency_rerank_seconds, 7.)

    def test_unavailable_cuda_fails_before_inputs(self):
        with patch.dict('sys.modules', {'torch': self.torch(False)}):
            with self.assertRaisesRegex(RuntimeError, 'CUDA requested'):
                CrossEncoderReranker(device='cuda')
            with patch.object(runner, 'read_json') as read:
                with self.assertRaisesRegex(RuntimeError, 'CUDA requested'):
                    runner.test_run(Namespace(device='cuda'))
                read.assert_not_called()

    def test_cuda_cannot_silently_use_cpu(self):
        with patch.dict('sys.modules', {'torch': self.torch()}):
            reranker = CrossEncoderReranker(device='cuda', model=SimpleNamespace(device='cpu'))
            with self.assertRaisesRegex(RuntimeError, 'effective CrossEncoder device'):
                reranker._get_model()

    def test_none_preserves_defaults_and_cpu_propagates(self):
        for device, kwargs in [(None, {}), ('cpu', {'device': 'cpu'})]:
            constructor = Mock(return_value=SimpleNamespace(device='cpu'))
            with patch.dict('sys.modules', {'sentence_transformers': SimpleNamespace(CrossEncoder=constructor)}):
                reranker = CrossEncoderReranker(device=device); reranker._get_model()
                constructor.assert_called_once_with(reranker.model_name, **kwargs)


class JournalTests(unittest.TestCase):
    def row(self, query='q'):
        return dict(query_id=query, method='Fixed Prefix', target_budget=10, seed=None,
                    reranked_pairs=10, reranking_latency=.01, pool_ids=[str(i) for i in range(100)],
                    selected_ids=[str(i) for i in range(10)], document_ids=[str(i) for i in range(10)])

    def test_resume_and_duplicate_prevention(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test_predictions.jsonl'; manifest = {'device': 'cuda:0'}
            row = self.row(); keys = {evaluation_key(row), evaluation_key(self.row('q2'))}
            first = PredictionJournal(path, manifest, keys); first.append(row)
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'duplicate'): first.append(row)
            self.assertEqual(path.read_bytes(), original)
            resumed = PredictionJournal(path, manifest, keys, resume=True)
            self.assertIn(evaluation_key(row), resumed.rows); resumed.append(self.row('q2'))
            self.assertEqual(len(path.read_text().splitlines()), 2)
            with self.assertRaises(FileExistsError): PredictionJournal(path, manifest, keys)
            with self.assertRaisesRegex(ValueError, 'mixing'):
                PredictionJournal(path, {'device': 'cpu'}, keys, resume=True)

    def test_cpu_records_and_duplicate_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test_predictions.jsonl'; row = self.row(); keys = {evaluation_key(row)}
            journal = PredictionJournal(path, {'device': 'cuda'}, keys); journal.append(row)
            payload = path.read_bytes(); path.write_bytes(payload + payload)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                PredictionJournal(path, {'device': 'cuda'}, keys, resume=True)
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'CPU records'):
                PredictionJournal(path, {'device': 'cuda'}, keys, resume=True)
            path.write_bytes(b'{broken}\n')
            with self.assertRaisesRegex(ValueError, 'invalid completed'):
                PredictionJournal(path, {'device': 'cuda'}, keys, resume=True)

    def test_torn_tail_is_preserved_and_recomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test_predictions.jsonl'; row = self.row(); keys = {evaluation_key(row)}
            PredictionJournal(path, {'device': 'cuda'}, keys); path.write_bytes(b'{"query_id":')
            resumed = PredictionJournal(path, {'device': 'cuda'}, keys, resume=True)
            self.assertEqual((path.parent / 'torn_tail.bin').read_bytes(), b'{"query_id":')
            resumed.append(row); self.assertEqual(len(resumed.rows), 1)

    def test_expected_keys_include_seeds_and_exclude_failures(self):
        lock = {'test_ids': ['a', 'b'], 'configurations': [
            {'method': 'Random', 'target_budget': 20, 'calibration_status': 'NOT_REQUIRED'},
            {'method': 'PoolAdapt', 'target_budget': 10, 'calibration_status': 'FAILED'}]}
        keys = expected_keys(lock); self.assertEqual(len(keys), 10)
        self.assertEqual({k[3] for k in keys}, set(range(5)))



class FrozenInputTests(unittest.TestCase):
    def test_repository_relative_paths_on_windows_and_linux(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(locked_path(r'D:\project\data\scifact', r'D:\project\data\scifact', root), root / 'data/scifact')
            self.assertEqual(locked_path('/old/project/results/model.pkl', '/old/project/data/scifact', root), root / 'results/model.pkl')
            self.assertEqual(locked_path('results/model.pkl', '/old/project/data/scifact', root), root / 'results/model.pkl')
            with self.assertRaises(ValueError):
                locked_path('../outside', '/old/project/data/scifact', root)

    def test_clone_input_provenance_without_old_summary(self):
        """Relocation preserves data/lock/calibration/runtime checks without packaging."""
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            data = root / 'data/scifact/corpus.jsonl'; data.parent.mkdir(parents=True); data.write_bytes(b'frozen data')
            source = root / 'src/pooladapt/features.py'; source.parent.mkdir(parents=True); source.write_bytes(b'feature = 1\r\n')
            source_digest = sha256(source); source.write_bytes(b'feature = 1\n')
            infrastructure = root / 'src/reranking/cross_encoder.py'; infrastructure.parent.mkdir(parents=True); infrastructure.write_bytes(b'authorized device support')
            out = root / 'results/phase2/evaluation'; out.mkdir(parents=True)
            calibration = out / 'budget_calibration.csv'; calibration.write_bytes(b'frozen calibration')
            lock_path = out / 'locked_configurations.json'
            environment = dict(python='3.11.15', torch='2.14.0')
            lock = dict(locked=True, dataset='SciFact', dataset_dir=r'D:\project\data\scifact',
                input_sha256={r'D:\project\data\scifact\corpus.jsonl': sha256(data),
                              r'D:\project\src\pooladapt\features.py': source_digest,
                              r'D:\project\src\reranking\cross_encoder.py': 'original CPU source digest'},
                phase1_sha256={r'results\phase1\frozen': 'digest'}, environment=environment,
                random_seeds=list(range(5)), test_ids=[str(i) for i in range(300)])
            lock_path.write_text(json.dumps(lock))
            for name, value in [('ROOT', root), ('LOCK', lock_path), ('SUMMARY', root / 'absent_summary.json'),
                                ('FROZEN_LOCK_SHA256', sha256(lock_path)), ('FROZEN_CALIBRATION_SHA256', sha256(calibration))]:
                stack.enter_context(patch.object(runner, name, value))
            stack.enter_context(patch.object(runner, 'phase1_hashes', return_value={'results/phase1/frozen': 'digest'}))
            stack.enter_context(patch.object(runner, 'environment_versions', return_value={**environment, 'torch': '2.14.0+cu128'}))
            stack.enter_context(patch.object(runner, 'validate_provenance_amendment'))
            runner.validate_lock(lock)
            data.write_bytes(b'changed data')
            with self.assertRaisesRegex(ValueError, 'locked input changed'): runner.validate_lock(lock)
            data.write_bytes(b'frozen data')
            calibration.write_bytes(b'changed calibration')
            with self.assertRaisesRegex(ValueError, 'budget calibration changed'): runner.validate_lock(lock)
            calibration.write_bytes(b'frozen calibration')
            lock_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'configuration lock was modified'): runner.validate_lock(lock)


class EvaluatorResumeTests(unittest.TestCase):
    def test_interrupted_gpu_run_skips_committed_keys_without_cpu_artifacts(self):
        """Exercise evaluator propagation, interruption and finalization on synthetic data."""
        ids = ['synthetic-a', 'synthetic-b']
        corpus = {str(i): {'text': 'evidence'} for i in range(100)}
        hits = [{'id': d, 'rank': i + 1, 'score': 1.} for i, d in enumerate(corpus)]
        class Retriever:
            def __init__(self, *args, **kwargs): pass
            def retrieve(self, *args): return hits
        class Model:
            classes_ = np.array([10, 20, 30, 50, 100])
            def predict_proba(self, values): return np.array([[.2] * 5])
        class Reranker:
            effective_device = 'cuda:0'
            last_latency_rerank_seconds = .01
            calls = 0
            kwargs = []
            def __init__(self, **kwargs): self.kwargs.append(kwargs)
            def warm_up(self): pass
            def rerank(self, query, selected, top_k):
                Reranker.calls += 1
                if Reranker.calls == 3: raise RuntimeError('simulated interruption')
                return selected[:top_k]
        configs = []
        for method in ('Full Rerank', 'Fixed Prefix', 'PACE-EF', 'Random', 'PoolAdapt', 'SAGE-SLO'):
            adaptive = method in ('PoolAdapt', 'SAGE-SLO')
            for target in ((100,) if method == 'Full Rerank' else (10, 20, 30, 50)):
                configs.append(dict(method=method, target_budget=target, actual_average_budget=10 if adaptive else target,
                    calibration_status='SUCCESS' if adaptive else 'NOT_REQUIRED', temperature=1 if adaptive else None,
                    budget_bias_lambda=1 if adaptive else None, budget_type='adaptive' if adaptive else 'exact'))
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp); old = root / 'results/phase2/evaluation'
            old.mkdir(parents=True); output = old / 'final_gpu'
            cpu_path = old / 'test_predictions.jsonl'; cpu_path.write_bytes(b'old CPU artifact; never read\n')
            lockpath = old / 'locked.json'
            lockpath.write_text(json.dumps(dict(locked=True, dataset_dir=str(root.resolve()), test_ids=ids,
                configurations=configs, dense_model='fake', cross_encoder_model='fake',
                models={'PoolAdapt': 'pool.pkl', 'SAGE-SLO': 'sage.pkl'})))
            for name, value in [('OUT', old), ('LOCK', lockpath), ('BM25Retriever', Retriever), ('DenseRetriever', Retriever),
                                ('CrossEncoderReranker', Reranker)]:
                stack.enter_context(patch.object(runner, name, value))
            stack.enter_context(patch.object(runner, 'validate_lock'))
            stack.enter_context(patch.object(runner, 'decision_projection', return_value={}))
            stack.enter_context(patch.object(runner, 'environment_versions', return_value={'torch': 'test'}))
            stack.enter_context(patch.object(runner, 'load_model', return_value=Model()))
            stack.enter_context(patch.object(runner, 'load_beir_dataset', return_value=(corpus, {q: 'evidence' for q in ids}, {})))
            stack.enter_context(patch.object(runner, 'frozen_features', return_value={name: 1. for name in runner.FROZEN_FEATURES}))
            stack.enter_context(patch.object(runner, 'pace_order', side_effect=lambda q, p, bm: p))
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: 'Test GPU'), version=SimpleNamespace(cuda='test'), Tensor=type('Tensor', (), {}))
            stack.enter_context(patch.dict('sys.modules', {'torch': torch}))
            labels = stack.enter_context(patch.object(runner, 'load_qrels', return_value={q: {'0': 1} for q in ids}))
            stack.enter_context(patch('builtins.print'))
            args = Namespace(dataset_dir=root, output_dir=output, device='cuda', fresh_output=True, resume=False)
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'): runner.test_run(args)
            labels.assert_not_called()
            first = (output / 'test_predictions.jsonl').read_bytes()
            self.assertEqual(len(first.splitlines()), 2)
            args.fresh_output = False; args.resume = True
            runner.test_run(args)
            rows = [json.loads(line) for line in (output / 'test_predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 74)
            self.assertEqual(len({evaluation_key(row) for row in rows}), 74)
            self.assertEqual(Reranker.calls, 75)  # 74 successful calls plus the interrupted call
            self.assertTrue((output / 'test_predictions.jsonl').read_bytes().startswith(first))
            self.assertEqual(cpu_path.read_bytes(), b'old CPU artifact; never read\n')
            self.assertTrue(all(k['device'] == 'cuda' for k in Reranker.kwargs))
            self.assertEqual(json.loads((output / 'phase2_summary.json').read_text())['status'], 'COMPLETE')
            with self.assertRaisesRegex(ValueError, 'already COMPLETE'): runner.test_run(args)
