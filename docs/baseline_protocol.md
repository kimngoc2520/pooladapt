# Baseline Protocol

This document separates the methods described by the original papers from the planned PoolAdapt reproduction. Details not explicitly fixed in the initialization protocol remain `TO BE FINALIZED`.

## Baseline Matrix

| Baseline name | Full paper title | arXiv ID | Purpose | Original method | Our implementation | Retrieval backbone | Candidate pool | Reranker | Budget | Calibration protocol | Known differences | Reproduction status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Full Rerank | TO BE FINALIZED | TO BE FINALIZED | Reference quality with all candidates reranked | Hybrid retrieval, N=100, rerank all 100 | Hybrid retrieval -> N=100 -> rerank all 100 | BM25 Top-100 + Dense Top-100 -> RRF | N=100 | Shared Cross-Encoder reranker | 100 | Not applicable; fixed full-pool budget | TO BE FINALIZED | Not started | Full reranking is the quality reference baseline. |
| Fixed Prefix | TO BE FINALIZED | TO BE FINALIZED | Fixed-budget candidate selection using the first M candidates | Hybrid retrieval, N=100, take first M, rerank M | Hybrid retrieval -> N=100 -> take first M -> rerank M | BM25 Top-100 + Dense Top-100 -> RRF | N=100 | Shared Cross-Encoder reranker | M in {10, 20, 30, 50, 70} | Fixed budget | TO BE FINALIZED | Not started | Candidate ordering is the fused pool ordering. |
| Random Selection | TO BE FINALIZED | TO BE FINALIZED | Fixed-budget candidate selection independent of candidate characteristics | Hybrid retrieval, N=100, randomly select M, rerank M | Hybrid retrieval -> N=100 -> randomly select M -> rerank M | BM25 Top-100 + Dense Top-100 -> RRF | N=100 | Shared Cross-Encoder reranker | M in {10, 20, 30, 50, 70} | Fixed budget; random seed policy is TO BE FINALIZED | TO BE FINALIZED | Not started | Keep separate from candidate-level adaptive methods. |
| PACE | Less can be More: Relieving RAG Bottlenecks via Evidence Frontloading and Pressure-Adaptive Budgeting | arXiv:2608.25115 | Adaptive evidence ordering and reranking-budget comparison | Evidence Frontloading + Pressure-Adaptive Budgeting | TO BE FINALIZED; if only Evidence Frontloading is reproduced, label it PACE-EF | TO BE FINALIZED | TO BE FINALIZED | Shared Cross-Encoder reranker | Adaptive; comparable average budgets required | Must be calibrated separately to comparable average reranking budgets | A partial implementation must not be labeled full PACE | Not started | PACE and PoolAdapt must not be merged. |
| SAGE-SLO | SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems | arXiv:2608.08237 | Query-level adaptive retrieval baseline | Query-level adaptation: how many passages a query needs | TO BE FINALIZED | TO BE FINALIZED | TO BE FINALIZED | Shared Cross-Encoder reranker | Adaptive; comparable average budgets required | Must be calibrated separately to comparable average reranking budgets | This refers specifically to SAGE-SLO, not other papers using SAGE | Not started | Do not document this baseline as ambiguous "SAGE". |
| PoolAdapt | TO BE FINALIZED | TO BE FINALIZED | Proposed candidate-level selective reranking method | TO BE FINALIZED; PoolAdapt is the original research contribution | TO BE FINALIZED after protocol review | BM25 Top-100 + Dense Top-100 -> RRF | N=100 | Shared Cross-Encoder reranker | M in {10, 20, 30, 50, 70}; final protocol TO BE FINALIZED | TO BE FINALIZED | Must remain distinct from SAGE-SLO and PACE mechanisms | Not started | Do not decide the final selector model before candidate analysis. |

## Common Protocol Elements

- The planned common retrieval backbone is BM25 Top-100 plus Dense Top-100 followed by RRF.
- RRF uses the standard rank-based formulation with default `k = 60`.
- A document absent from a retriever contributes zero from that retriever; no artificial rank 0 or rank 101/106 is assigned.
- All baselines and PoolAdapt are planned to use the same Cross-Encoder reranker.
- Adaptive baselines must be calibrated separately before comparisons at comparable average reranking budgets.

## Reproduction Status

No baseline has been implemented or evaluated during repository initialization.
