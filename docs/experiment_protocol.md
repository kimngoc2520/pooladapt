# Experiment Protocol

This is the current protocol for review before implementation. Items not fixed by the initialization brief are marked `TO BE FINALIZED`.

## Research Question

Can candidate-pool characteristics derived from sparse-dense retrieval signals be used to selectively rerank candidates in large hybrid candidate pools while reducing inference cost with minimal retrieval-quality degradation?

## Hypothesis

Candidates with high sparse-dense disagreement may have higher reranking utility. Selectively reranking such candidates may reduce cost with limited quality degradation. This is a hypothesis, not an established conclusion, and experiments must be able to falsify it.

## Dataset Progression

1. SciFact: pilot, debugging, candidate analysis, and budget sweep
2. FiQA: validation and generalization
3. HotpotQA: stress test and generalization
4. MS MARCO: optional, only if time remains

Datasets must not be downloaded during repository initialization.

## Query Split

A fixed query split will be used early. The example split is 70% train, 10% validation, and 20% test. The exact split is `TO BE FINALIZED`.

Test data must remain locked.

## Retrieval and Candidate Pool

- Candidate pool size: `N = 100`
- Sparse retrieval: BM25 Top-100
- Dense retrieval: Dense Top-100
- Fusion: Reciprocal Rank Fusion (RRF)
- RRF parameter: `k = 60`
- A missing document contribution from a retriever is zero.
- No artificial rank 0 or rank 101/106 is assigned.

The RRF score is the sum over retrievers of `1 / (k + rank(d))` for retrievers where the document is present.

## Reranking Budgets

Planned fixed budgets are `M in {10, 20, 30, 50, 70}` for a candidate pool of `N = 100`. The exact final budget protocol is `TO BE FINALIZED`.

The primary algorithm-level cost proxy is the number of Cross-Encoder reranked pairs.

## Baselines

- Full Rerank
- Fixed Prefix
- Random Selection
- PACE
- SAGE-SLO
- PoolAdapt

PACE is the method described in "Less can be More: Relieving RAG Bottlenecks via Evidence Frontloading and Pressure-Adaptive Budgeting" (arXiv:2608.25115). If only Evidence Frontloading can be reproduced, it must be called PACE-EF.

SAGE-SLO is the method described in "SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems" (arXiv:2608.08237). Other papers using the acronym SAGE are not this baseline.

The baseline distinctions are:

- SAGE-SLO: query-level adaptation of how many passages a query needs.
- PACE: evidence ordering plus adaptive reranking budget.
- PoolAdapt: candidate-level selection of which candidates inside a large pool are worth reranking.

## Calibration

Adaptive baselines such as SAGE-SLO and PACE must be calibrated separately so comparisons can be made at comparable average reranking budgets. Adaptive baselines must not be assumed to naturally produce the same average budget. The detailed calibration procedure is `TO BE FINALIZED`.

## Metrics

### Quality

- nDCG@10
- Recall@10
- MRR@10

### Efficiency

- Rerank pairs per query
- Compression ratio
- Reranker input tokens per query, when measurable

### Latency

Measure retrieval latency, selector latency, reranking latency, and end-to-end latency. Report P50 and P95.

### Trade-off

- Quality retention
- Quality at comparable budget
- Quality-cost curves across multiple budget levels

Compression alone is not evidence of success.

## Leakage Rules

Full-reranker outputs may be used to create candidate-utility labels only for train and validation during selector development.

Test data must not be used to:

- Create training labels
- Tune thresholds
- Tune features
- Choose selector models
- Choose checkpoints
- Make implementation decisions

## Random Seed Policy

`TO BE FINALIZED`.

## Hardware and Environment

`TO BE FINALIZED`.

## Protocol Status

The protocol is a reviewable initialization document. Implementation details, selector model choice, exact splits, calibration, random seeds, and hardware/environment specifications must be finalized before experiments.
