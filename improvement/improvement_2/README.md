# Improvement 2

This package contains development-only research artifacts for candidate-pool-aware reranking: audit evidence, protocol documents, reproducible executions, and their reports. It preserves historical work alongside the current implementation; execution identity is recorded independently of readable directory names.

## Layout

- `audit/` contains source-data, feature, oracle, and implementation audits. `pre_execution/` and `post_execution/` retain dated readiness and reconciliation records.
- `docs/` holds research documents. Historical document names and contents are preserved.
- `manifests/` contains retained manifests and `migration_mapping.json`.
- `reports/` contains the main protocol-complement report and stage-level historical reports.
- `scripts/` contains the current runner and shared methodology module. `scripts/legacy/` retains prior runners.
- `tests/` contains deterministic methodology regression tests.
- `results/` contains readable execution directories and `execution_index.json`, which maps each to its original ID.

## Current implementation

`scripts/run.py` is the current execution entry point. `scripts/protocol.py` contains reusable probability alignment, Hard-score, lambda-shift, bisection, closed-form-null, and Holm-correction functions. The files in `scripts/legacy/` are preserved only for historical reproduction.

## Run

The runner requires an unused, human-readable output directory and the authoritative protocol source. It performs a development-only analysis using persisted reranking outputs:

```powershell
python improvement/improvement_2/scripts/run.py --output improvement/improvement_2/results/development_run --protocol "C:\Users\User\Downloads\Cải tiến 2.docx"
```

Do not reuse an existing execution directory. Consult `results/execution_index.json` before selecting a name.

## Test

```powershell
python -m pytest improvement/improvement_2/tests/test_methodology.py -q -p no:cacheprovider
```

## Execution history and reports

`results/latest_development/` is the latest completed development execution. `results/preflight_attempt/` preserves an empty preflight attempt. `results/legacy_run/` and `results/historical_baseline/` preserve earlier work. Each retained manifest keeps original timestamps and run IDs; `results/execution_index.json` and `manifests/migration_mapping.json` provide the path mapping.

Read `reports/protocol_complement_report.md` for current traceability and limitations. The oracle-headroom, feature-ablation, and budget-allocation reports are historical records and retain their original statements and references.
