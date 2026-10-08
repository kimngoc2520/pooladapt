"""Frozen input path resolution and durable, device-isolated TEST journals."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from src.evaluation.phase2 import SEEDS, sha256

def locked_path(original: str, dataset_dir: str, root: Path) -> Path:
    """Resolve a frozen repository path after a Windows/Linux repository move."""
    kind = PureWindowsPath if PureWindowsPath(dataset_dir).drive or "\\" in dataset_dir else PurePosixPath
    old_root = kind(dataset_dir).parent.parent
    source = kind(original)
    relative = source.relative_to(old_root) if source.is_absolute() else source
    path = (root / relative.as_posix()).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('locked input escapes repository root')
    return path


def evaluation_key(row: dict[str, Any]) -> tuple[str, str, int, int | None]:
    """Canonical query/method/target/seed key, independent of file order."""
    seed = row['seed']
    target = row['target_budget']
    if type(target) is not int or (seed is not None and type(seed) is not int):
        raise ValueError('invalid evaluation key types')
    return str(row['query_id']), str(row['method']), target, seed


def expected_keys(lock: dict[str, Any]) -> set[tuple[str, str, int, int | None]]:
    """Enumerate only frozen successful/reference operating points."""
    return {(str(q), c['method'], c['target_budget'], seed)
            for q in lock['test_ids'] for c in lock['configurations']
            if c['calibration_status'] != 'FAILED'
            for seed in (SEEDS if c['method'] == 'Random' else (None,))}


class PredictionJournal:
    """Append flushed records once; fail closed on foreign runs or duplicate keys."""

    def __init__(self, path: Path, manifest: dict[str, Any], allowed: set,
                 resume: bool = False) -> None:
        self.path, self.allowed = path, allowed
        self.manifest_path = path.with_name('execution_manifest.json')
        self.rows: dict[tuple, dict[str, Any]] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            saved = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            if saved != manifest:
                raise ValueError('resume runtime/device/input manifest differs; CPU/GPU mixing is forbidden')
            if not path.exists():
                path.touch(exist_ok=False)  # interruption after manifest commit, before first row
        else:
            if path.exists() or self.manifest_path.exists():
                raise FileExistsError('fresh output must be unused; use --resume only for this GPU run')
            temporary = self.manifest_path.with_suffix('.json.tmp')
            with temporary.open('w', encoding='utf-8') as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.manifest_path)
            path.touch(exist_ok=False)
        self.run_digest = sha256(self.manifest_path)
        with path.open('rb') as handle:
            offset = 0
            while line := handle.readline():
                if not line.endswith(b'\n'):
                    # A crash during write can leave a torn final record. Preserve its bytes.
                    backup = path.with_name('torn_tail.bin')
                    number = 1
                    while backup.exists():
                        backup = path.with_name(f'torn_tail.{number}.bin')
                        number += 1
                    with backup.open('xb') as saved_tail:
                        saved_tail.write(line)
                    with path.open('r+b') as recovery:
                        recovery.truncate(offset)
                        recovery.flush()
                        os.fsync(recovery.fileno())
                    break
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError) as error:
                    raise ValueError('invalid completed prediction record') from error
                self._validate(row)
                self.rows[evaluation_key(row)] = row
                offset += len(line)

    def _validate(self, row: dict[str, Any]) -> None:
        key = evaluation_key(row)
        if key not in self.allowed or key in self.rows:
            raise ValueError(f'unknown or duplicate evaluation key: {key}')
        if row.get('execution_manifest_sha256') != self.run_digest:
            raise ValueError('prediction belongs to another execution; CPU records are forbidden')
        if not math.isfinite(row['reranking_latency']) or row['reranking_latency'] < 0:
            raise ValueError('invalid latency')
        budget = row['reranked_pairs']
        if type(budget) is not int or budget not in (10, 20, 30, 50, 100):
            raise ValueError('invalid saved budget')
        if row['method'] not in ('PoolAdapt', 'SAGE-SLO') and budget != row['target_budget']:
            raise ValueError('exact-budget method changed its frozen budget')
        pool, selected, ranked = row['pool_ids'], row['selected_ids'], row['document_ids']
        if len(pool) != 100 or len(set(pool)) != 100 or len(selected) != budget or len(set(selected)) != budget:
            raise ValueError('invalid saved candidate population/budget')
        if not set(selected) <= set(pool) or len(ranked) != min(10, budget) or len(set(ranked)) != len(ranked) or not set(ranked) <= set(selected):
            raise ValueError('invalid saved ranking')

    def append(self, row: dict[str, Any]) -> None:
        """Commit one complete record durably, then add its key to the completed set."""
        row = {**row, 'execution_manifest_sha256': self.run_digest}
        self._validate(row)
        with self.path.open('a', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[evaluation_key(row)] = row
