"""Regression tests for the final calibration/evaluation boundary."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import run_phase2_evaluation as runner
from src.baselines.fixed_prefix import select_fixed_prefix
from src.baselines.pace_ef import select_prefix
from src.baselines.random_selection import select_random
from src.baselines.sage_slo import select_budget
from src.evaluation.phase2 import (CALIBRATION_FIELDS, RESULT_FIELDS, SEEDS, TARGETS,
    aggregate, calibrate, heuristic_gates, lambda_sweep, paired_statistics,
    predict_budgets, random_aggregate, sha256, verify_population, write_csv)
from src.evaluation.phase2_inputs import pace_order, validate_pool
from src.pooladapt.selector import BUDGET_GRID, FROZEN_FEATURES, calibrated_logits
from src.retrieval.bm25 import BM25Retriever


class CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = [str(i) for i in range(162)]
        self.logits = np.random.default_rng(42).normal(size=(162, 5))

    def test_validation_only_and_test_cannot_influence(self) -> None:
        with self.assertRaises(ValueError):
            calibrate('PoolAdapt', self.ids, self.ids, ['0'], self.logits)
        with self.assertRaises(ValueError):
            calibrate('PoolAdapt', self.ids + ['test'], self.ids, ['test'], self.logits)
        with self.assertRaises(ValueError):
            verify_population(['a', 'a'], ['a'])
        with self.assertRaises(ValueError):
            calibrate('PoolAdapt', self.ids, self.ids, [], self.logits[:-1])

    def test_both_existing_controls_and_closest_workload(self) -> None:
        for method in ('PoolAdapt', 'SAGE-SLO'):
            rows, sanity = calibrate(method, self.ids, self.ids, ['test'], self.logits)
            self.assertTrue(sanity['responsive'])
            averages = [predict_budgets(self.logits, 1, lam).mean() for lam in lambda_sweep(self.logits)]
            for row in rows:
                self.assertLessEqual(row['relative_budget_error'], .05)
                self.assertAlmostEqual(abs(row['actual_average_budget'] - row['target_budget']), min(abs(x - row['target_budget']) for x in averages))
                self.assertTrue(row['locked'])
                self.assertIsNone(row['validation_nDCG@10'])

    def test_unreachable_targets_fail_without_relabeling(self) -> None:
        rows, _ = calibrate('SAGE-SLO', ['v'], ['v'], ['test'], np.zeros((1, 5)))
        self.assertEqual(rows[0]['calibration_status'], 'SUCCESS')
        self.assertEqual([r['calibration_status'] for r in rows[1:]], ['FAILED'] * 3)
        self.assertEqual(rows[1]['actual_average_budget'], 10)

    def test_unresponsive_control_stops_method(self) -> None:
        logits = np.tile([0., -1000., -1000., -1000., -1000.], (2, 1))
        rows, sanity = calibrate('SAGE-SLO', ['a', 'b'], ['a', 'b'], [], logits)
        self.assertFalse(sanity['responsive'])
        self.assertTrue(all(r['calibration_status'] == 'FAILED' for r in rows))

    def test_tie_break_and_temperature_equivalence(self) -> None:
        self.assertEqual(int(predict_budgets(np.zeros((1, 5)), 1, 0)[0]), 10)
        for row in self.logits[:10]:
            for lam in (-5, 0, 5):
                self.assertEqual(select_budget(row, 1, lam), int(predict_budgets(row[None, :], 1, lam)[0]))
                np.testing.assert_allclose(calibrated_logits(row, 1, lam), row + lam * __import__('src.pooladapt.selector', fromlist=['budget_bias']).budget_bias())
        np.testing.assert_array_equal(predict_budgets(self.logits, 2, 3), predict_budgets(self.logits, 1, 6))

    def test_invalid_parameters(self) -> None:
        for t, lam in ((0, 0), (-1, 0), (1, float('nan'))):
            with self.assertRaises(ValueError):
                predict_budgets(self.logits, t, lam)


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = [{'id': str(i), 'title': '', 'text': 'evidence term', 'rrf_score': 1 / (60 + i + 1)} for i in range(100)]

    def test_exact_budgets_and_random_reproducibility(self) -> None:
        for target in TARGETS:
            self.assertEqual(select_fixed_prefix(self.pool, target), self.pool[:target])
            self.assertEqual(select_prefix(self.pool[::-1], target), self.pool[::-1][:target])
            for seed in SEEDS:
                a = select_random(self.pool, target, seed=seed, query_id='q')
                self.assertEqual(a, select_random(self.pool, target, seed=seed, query_id='q'))
                self.assertEqual(len(a), target)
                self.assertEqual(len({c['id'] for c in a}), target)
        self.assertNotEqual(select_random(self.pool, 20, seed=0, query_id='q'), select_random(self.pool, 20, seed=1, query_id='q'))

    def test_pace_preserves_pool_and_fixed_k(self) -> None:
        bm25 = BM25Retriever({c['id']: c for c in self.pool})
        ordered = pace_order('evidence', self.pool, bm25)
        self.assertEqual({c['id'] for c in ordered}, {c['id'] for c in self.pool})
        for target in TARGETS:
            self.assertEqual(len(select_prefix(ordered, target)), target)

    def test_relevance_cannot_enter_selector(self) -> None:
        validate_pool(self.pool)
        for forbidden in ('qrels', 'relevance', 'M_star', 'label'):
            polluted = [dict(c, **{forbidden: 1}) for c in self.pool]
            with self.assertRaises(ValueError):
                validate_pool(polluted)
        with self.assertRaises(ValueError):
            validate_pool(self.pool[:-1] + [self.pool[0]])

    def test_heuristic_stays_not_evaluated(self) -> None:
        gates = heuristic_gates({'g2_status': 'FAILED', 'g3_status': 'NOT_EVALUATED'})
        self.assertEqual(gates['G2'], 'FAILED')
        self.assertEqual(gates['G3a'], 'NOT_EVALUATED')
        self.assertEqual(gates['G3b'], 'NOT_EVALUATED')
        with self.assertRaises(ValueError):
            heuristic_gates({'g2_status': 'PASS'})


class ReportingTests(unittest.TestCase):
    def rows(self, budget: int = 20) -> list[dict]:
        return [dict(query_id=q, reranked_pairs=budget, reranking_latency=l,
                     **{'nDCG@10': n, 'Recall@10': n, 'MRR@10': n}) for q, l, n in [('a', .1, 1.), ('b', .3, 0.)]]

    def config(self) -> dict:
        return dict(method='PoolAdapt', target_budget=20, actual_average_budget=20,
                    calibration_status='SUCCESS', budget_type='adaptive', seed=None)

    def test_metrics_latency_and_compression(self) -> None:
        result = aggregate(self.rows(), self.config(), .4)
        self.assertEqual(result['nDCG@10'], .5)
        self.assertEqual(result['total_reranked_pairs'], 40)
        self.assertEqual(result['compression_ratio'], .8)
        self.assertEqual(result['latency_reduction'], .5)
        self.assertTrue(all(result[key] >= 0 for key in ('mean_reranking_latency', 'p50_reranking_latency', 'p95_reranking_latency')))
        self.assertFalse(aggregate(self.rows(30), self.config(), .4)['matched_budget'])
        with self.assertRaises(ValueError):
            aggregate(self.rows() + self.rows(), self.config(), .4)
        bad = self.rows(); bad[0]['reranking_latency'] = -1
        with self.assertRaises(ValueError):
            aggregate(bad, self.config(), .4)

    def test_random_sd_across_seeds(self) -> None:
        seeds = []
        for seed in SEEDS:
            result = aggregate(self.rows(), dict(self.config(), seed=seed), .4)
            result['nDCG@10'] = seed / 10
            seeds.append(result)
        aggregate_seed = random_aggregate(seeds)
        self.assertAlmostEqual(aggregate_seed['nDCG@10'], .2)
        self.assertAlmostEqual(aggregate_seed['nDCG@10_seed_std'], np.std(np.arange(5) / 10, ddof=1))
        with self.assertRaises(ValueError):
            random_aggregate(seeds[:-1])

    def test_schemas_and_empty_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'results.csv'
            write_csv(path, [aggregate(self.rows(), self.config(), .4)], RESULT_FIELDS)
            with path.open(newline='') as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames), RESULT_FIELDS)
                self.assertEqual(len(list(reader)), 1)
            write_csv(path, [], CALIBRATION_FIELDS)
            self.assertEqual(path.read_text().strip(), ','.join(CALIBRATION_FIELDS))

    def test_paired_queries_holm_and_random_structure(self) -> None:
        groups = {('Full Rerank', 100, None): self.rows(100)}
        for target in TARGETS:
            groups[('Fixed Prefix', target, None)] = self.rows(target)
            for seed in SEEDS:
                groups[('Random', target, seed)] = self.rows(target)
        matched = {(name, b) for b in TARGETS for name in ('Fixed Prefix', 'Random')}
        stats = paired_statistics(groups, matched, ['a', 'b'])
        self.assertEqual(len(stats), 24)
        self.assertTrue(all(r['p_Holm'] == 1 and r['query_count'] == 2 and r['paired'] for r in stats))
        self.assertTrue(all(r['random_construction'] == 'query mean over 5 seeds' for r in stats if r['method_a'] == 'Random'))
        groups[('Fixed Prefix', 10, None)][0]['query_id'] = 'test'
        with self.assertRaises(ValueError):
            paired_statistics(groups, matched, ['a', 'b'])

    def test_lock_reuse_tampering_and_phase1_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / 'model.pkl'; input_path.write_bytes(b'frozen')
            lock_path = root / 'lock.json'; lock_path.write_text('{}')
            calibration_path = root / 'budget_calibration.csv'; calibration_path.write_bytes(b'calibration')
            summary_path = root / 'summary.json'
            lock = dict(locked=True, input_sha256={str(input_path): sha256(input_path)}, phase1_sha256={'frozen': 'digest'}, random_seeds=list(SEEDS), test_ids=[str(i) for i in range(300)])
            with patch.object(runner, 'LOCK', lock_path), patch.object(runner, 'SUMMARY', summary_path), \
                 patch.object(runner, 'FROZEN_LOCK_SHA256', sha256(lock_path)), \
                 patch.object(runner, 'FROZEN_CALIBRATION_SHA256', sha256(calibration_path)), \
                 patch.object(runner, 'validate_provenance_amendment'), \
                 patch.object(runner, 'phase1_hashes', return_value={'frozen': 'digest'}):
                runner.validate_lock(lock)
                self.assertEqual(input_path.read_bytes(), b'frozen')
                input_path.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    runner.validate_lock(lock)
                input_path.write_bytes(b'frozen')
                lock_path.write_text('{"lambda": 9}')
                with self.assertRaises(ValueError):
                    runner.validate_lock(lock)
            with patch.object(runner, 'LOCK', lock_path), patch.object(runner, 'SUMMARY', summary_path), \
                 patch.object(runner, 'FROZEN_LOCK_SHA256', sha256(lock_path)), \
                 patch.object(runner, 'FROZEN_CALIBRATION_SHA256', sha256(calibration_path)), \
                 patch.object(runner, 'validate_provenance_amendment'), \
                 patch.object(runner, 'phase1_hashes', return_value={'frozen': 'changed'}):
                with self.assertRaises(ValueError):
                    runner.validate_lock(lock)

    def test_complete_test_stage_uses_locks_before_labels_and_preserves_failures(self) -> None:
        """Exercise all output writers on synthetic 300 queries, without real TEST."""
        ids = [str(i) for i in range(300)]
        corpus = {f'd{i}': {'text': 'evidence', 'title': ''} for i in range(100)}
        hits = [{'id': d, 'rank': i + 1, 'score': 1.} for i, d in enumerate(corpus)]
        class Retriever:
            def __init__(self, *args, **kwargs): pass
            def retrieve(self, query, top_n): return hits
        class Model:
            classes_ = np.asarray(BUDGET_GRID)
            def predict_proba(self, values): return np.asarray([[.2] * 5])
        class Reranker:
            calls = 0
            warmed = False
            last_latency_rerank_seconds = .01
            def __init__(self, *args, **kwargs): pass
            def warm_up(self): self.warmed = True
            def rerank(self, query, selected, top_k):
                assert self.warmed
                Reranker.calls += 1
                return list(selected[:top_k])
        configs = []
        for method in ('Full Rerank', 'Fixed Prefix', 'PACE-EF', 'Random', 'PoolAdapt', 'SAGE-SLO'):
            for target in ((100,) if method == 'Full Rerank' else TARGETS):
                adaptive = method in ('PoolAdapt', 'SAGE-SLO')
                configs.append(dict(method=method, target_budget=target, actual_average_budget=10 if adaptive else target,
                                    calibration_status=('SUCCESS' if target == 10 else 'FAILED') if adaptive else 'NOT_REQUIRED',
                                    temperature=1 if adaptive else None, budget_bias_lambda=1 if adaptive else None,
                                    locked=True, budget_type='adaptive' if adaptive else 'exact'))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); out = root / 'results/phase2/evaluation'; out.mkdir(parents=True)
            lock_path, summary_path = out / 'locked.json', out / 'stage_summary.json'
            lock = dict(locked=True, dataset_dir=str(root.resolve()), test_ids=ids, configurations=configs,
                        dense_model='fake', cross_encoder_model='fake', models={'PoolAdapt': 'pool.pkl', 'SAGE-SLO': 'sage.pkl'})
            lock_path.write_text(json.dumps(lock))
            self.assertFalse(summary_path.exists())  # TEST must not require prior stage status.
            def evaluation_labels(path):
                self.assertEqual(Reranker.calls, 300 * 31)
                return {qid: {'d0': 1} for qid in ids}
            with patch.object(runner, 'OUT', out), patch.object(runner, 'LOCK', lock_path), patch.object(runner, 'SUMMARY', summary_path), \
                 patch.object(runner, 'validate_lock') as validate, patch.object(runner, 'load_model', return_value=Model()), \
                 patch.object(runner, 'load_beir_dataset', return_value=(corpus, {q: 'evidence' for q in ids}, {})), \
                 patch.object(runner, 'BM25Retriever', Retriever), patch.object(runner, 'DenseRetriever', Retriever), \
                 patch.object(runner, 'CrossEncoderReranker', Reranker), patch.object(runner, 'pace_order', side_effect=lambda query, pool, bm: list(reversed(pool))), \
                 patch.object(runner, 'frozen_features', return_value={name: 1. for name in FROZEN_FEATURES}), \
                 patch.object(runner, 'load_qrels', side_effect=evaluation_labels), patch('builtins.print'):
                runner.test_run(Namespace(dataset_dir=root))
                self.assertEqual(validate.call_count, 2)
            self.assertEqual(json.loads(lock_path.read_text()), lock)
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary['status'], 'COMPLETE')
            self.assertEqual(len(summary['final_test_results']), 21)
            self.assertEqual(len(summary['random_seed_results']), 20)
            failed = [r for r in summary['final_test_results'] if r['calibration_status'] == 'FAILED']
            self.assertEqual(len(failed), 6)
            self.assertTrue(all(r['actual_average_budget'] is None and r['validation_average_budget'] == 10 for r in failed))
            for name in ('comparative_results.csv', 'quality_cost_latency.csv', 'random_seed_results.csv', 'statistical_analysis.csv'):
                self.assertTrue((out / name).is_file())


if __name__ == '__main__':
    unittest.main()
