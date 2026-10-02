# SAGE-inspired / adapted SAGE-SLO

Verified source (2026-10-02): Raza, Yang, and Srinivasan, *SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems*, [arXiv:2608.08237](https://arxiv.org/abs/2608.08237), CoDIT 2026, DOI 10.1109/CoDIT70676.2026.11631166. SAGE is a learned SLO-aware retrieval policy: it selects a per-query passage budget from initial-retrieval features, imitating a latency-quality oracle. Its original quality criterion is answer Exact Match under end-to-end tail-latency SLO and retrieval-cost constraints. It is not the unrelated Selective Attention-Guided Extraction work.

This is a **SAGE-inspired / adapted SAGE-SLO** baseline, not a faithful reproduction. It maps the policy to the existing BM25 + Dense + RRF pool of 100 and `{10,20,30,50,100}`. Its train-only oracle is the project M* nDCG@10-retention target at tau=0.99; it does not evaluate generation, EM, an end-to-end SLO, or LLM calls.

`z_k(q)=log(max(P_model(M*=k | x(q)), 1e-12))` for every budget. The model is a deterministic-seeded RandomForestClassifier (100 trees, depth 10, balanced classes) trained only on locked TRAIN labels; a TRAIN-fitted median imputer precedes it. `x(q)` contains only the existing 13 pool/query fields from `results/phase2/01_candidate_logging/{pool_features,query_features}.csv`. qrels, relevance, M*, IDs, Cross-Encoder outputs, reranked positions, and TEST data are excluded at inference.

The exposed non-final calibration is `z'_k(q)=z_k(q)/T + lambda*g(k)`, where `g(k)=-(k-mean(G))/std(G)`. Prompt 6 owns final validation calibration and matched-budget comparisons.
