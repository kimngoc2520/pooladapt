# Locked SciFact Phase 2 development split

The original SciFact dataset provides TRAIN and TEST splits only. For Phase 2
development and calibration, the original TRAIN query set is partitioned into
disjoint TRAIN and VALIDATION subsets using a fixed query-level random seed of
42 and an 80/20 split. The original SciFact TEST split remains locked and is
reserved for final evaluation.

This is a project-specific development split derived from original TRAIN, not
an official SciFact validation split. The lock files, split-specific qrels, and
development reranking outputs are produced by:

```powershell
python experiments/prepare_phase2_scifact.py
```

The command uses the existing full-corpus BM25, dense, RRF, and Cross-Encoder
implementations. It generates both `full_rerank` and `fixed_prefix_100`, checks
their actual per-query outputs, and keeps both method identities.
