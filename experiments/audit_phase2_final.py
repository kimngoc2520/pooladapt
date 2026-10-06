"""Independently audit frozen final Phase 2 artifacts without model inference."""
from __future__ import annotations

import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.phase2 import METRICS, SEEDS, TARGETS, sha256

OUT = ROOT / 'results/phase2/07_evaluation'


def audit() -> dict[str, Any]:
    """Check populations, selection provenance, locks, aggregation, and timing."""
    lock_path = OUT / 'locked_configurations.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    summary = json.loads((ROOT / 'results/phase2/phase2_summary.json').read_text(encoding='utf-8'))
    if summary['status'] != 'COMPLETE' or summary['locked_configuration_sha256'] != sha256(lock_path):
        raise ValueError('final evaluation incomplete or configuration lock changed')
    for path, digest in lock['input_sha256'].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f'locked input changed: {path}')
    for path, digest in lock['phase1_sha256'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f'frozen Phase 1 artifact changed: {path}')
    tree = ast.parse((ROOT / 'experiments/run_phase2_evaluation.py').read_text(encoding='utf-8'))
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'fit' for node in ast.walk(tree)):
        raise ValueError('final evaluator contains a fitting call')
    train, validation, test = map(set, (lock['train_ids'], lock['validation_ids'], lock['test_ids']))
    if (len(train), len(validation), len(test)) != (647, 162, 300) or train & validation or train & test or validation & test:
        raise ValueError('invalid or overlapping split populations')
    for info in lock['checkpoint_information'].values():
        meta = info['metadata']
        if set(meta['training_query_ids']) != train or meta['n_training_queries'] != 647 or meta['fit_count'] != 1:
            raise ValueError('checkpoint TRAIN provenance mismatch')
    configs = {(r['method'], r['target_budget']): r for r in lock['configurations']}
    groups = defaultdict(dict)
    pools, orders = {}, {}
    calls = 0
    with (OUT / 'test_predictions.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            qid, method, target, seed = row['query_id'], row['method'], row['target_budget'], row['seed']
            config = configs[(method, target)]
            if config['calibration_status'] == 'FAILED' or qid not in test:
                raise ValueError('failed/unexpected operating point received TEST inference')
            key = (method, target, seed)
            if qid in groups[key]:
                raise ValueError('duplicate prediction key')
            pool, order, selected = row['pool_ids'], row['frontloaded_ids'], row['selected_ids']
            if len(pool) != 100 or len(set(pool)) != 100 or len(order) != 100 or set(order) != set(pool):
                raise ValueError('pool or PACE permutation mismatch')
            if qid in pools and (pools[qid] != pool or orders[qid] != order):
                raise ValueError('methods received different candidate pools/order')
            pools[qid], orders[qid] = pool, order
            k = row['reranked_pairs']
            if len(selected) != k or len(set(selected)) != k or not set(selected) <= set(pool):
                raise ValueError('selection workload/uniqueness mismatch')
            if config['budget_type'] == 'exact' and k != target:
                raise ValueError('exact K mismatch')
            if method in ('PoolAdapt', 'SAGE-SLO', 'Fixed Prefix', 'Full Rerank') and selected != pool[:k]:
                raise ValueError('RRF-prefix selection mismatch')
            if method == 'PACE-EF' and selected != order[:target]:
                raise ValueError('PACE-EF did not use exact frontloaded prefix')
            if method == 'Random' and seed not in SEEDS:
                raise ValueError('unexpected Random seed')
            if row['reranking_latency'] < 0 or not np.isfinite(row['reranking_latency']):
                raise ValueError('invalid measured latency')
            if len(row['document_ids']) != 10 or len(set(row['document_ids'])) != 10 or not set(row['document_ids']) <= set(selected):
                raise ValueError('Top-10 output mismatch')
            groups[key][qid] = row
            calls += 1
    expected = {(r['method'], r['target_budget'], seed) for r in configs.values() if r['calibration_status'] != 'FAILED' for seed in (SEEDS if r['method'] == 'Random' else (None,))}
    if set(groups) != expected or any(set(rows) != test for rows in groups.values()):
        raise ValueError('operating-point populations differ from the 300 locked TEST IDs')
    with (OUT / 'test_query_metrics.csv').open(encoding='utf-8', newline='') as handle:
        metrics = list(csv.DictReader(handle))
    keys = {(r['method'], int(r['target_budget']), r['seed'], r['query_id']) for r in metrics}
    if len(metrics) != calls or len(keys) != calls:
        raise ValueError('metric row cardinality/uniqueness mismatch')
    for row in summary['final_test_results']:
        method, target = row['method'], row['target_budget']
        if row['calibration_status'] == 'FAILED':
            if any(row[m] is not None for m in METRICS):
                raise ValueError('failed point has fabricated TEST metrics')
            continue
        query_rows = [r for r in metrics if r['method'] == method and int(r['target_budget']) == target]
        average = float(np.mean([float(r['reranked_pairs']) for r in query_rows]))
        if not np.isclose(average, row['actual_average_budget']) or not np.isclose(row['compression_ratio'], 1 - average / 100):
            raise ValueError('aggregate budget/compression mismatch')
        for metric in METRICS:
            if not np.isclose(row[metric], np.mean([float(r[metric]) for r in query_rows])):
                raise ValueError('aggregate quality mismatch')
    result = {'status': 'PASSED', 'training_queries': len(train), 'validation_queries': len(validation), 'test_queries': len(test),
              'test_inference_calls': calls, 'query_metric_rows': len(metrics), 'operating_point_seed_groups': len(groups),
              'same_test_ids_for_every_group': True, 'every_pool_100_unique_documents': True,
              'shared_pools_across_methods': True, 'pace_pool_permutation_preserved': True,
              'exact_fixed_pace_random_budgets': True, 'full_rerank_K_100': True,
              'random_seeds': list(SEEDS), 'locked_parameters_unchanged': True, 'checkpoint_recipes_and_hashes_unchanged': True,
              'no_fit_call_in_final_evaluator': True, 'phase1_unchanged': True,
              'qrels_usage': 'TEST qrels loaded only after all inference; selector interfaces accept no relevance inputs',
              'timing': lock['timing'], 'failed_targets': summary['calibration_failures'],
              'test_workload_drift': summary['test_budget_drift'], 'gates': lock['gates']}
    (OUT / 'validation_checks.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    audit()
