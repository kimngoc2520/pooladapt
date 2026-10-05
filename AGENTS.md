# Project Overview: PoolAdapt
- **Full Title**: PoolAdapt: Candidate-Pool-Aware Selective Reranking for Large Candidate Pools in Agentic Hybrid RAG under Inference Cost Constraints
- **Core Research Contribution**: A candidate-pool-aware selective reranking method. Instead of running an expensive Cross-Encoder reranker on all candidates in a large pool ($N=100$), PoolAdapt analyzes candidate pool characteristics and adaptively selects an optimal subset $S$ of size $K$ ($\vert{}S\vert{}=K$) to pass to the reranker, minimizing inference cost while preserving retrieval quality (`nDCG@10`).

# Architecture & Responsibilities
1. **Agentic Workflow / Router (`src/agent/`)**:
   - System-level orchestration. Handles workflow routing and tool calling.
   - IT DOES NOT select candidates itself. It invokes Hybrid Retrieval, passes the pool to PoolAdapt, receives selected candidates, calls the Reranker, and feeds Top-K to the LLM.
2. **Hybrid Retrieval (`src/retrieval/`)**:
   - BM25 Top-100 + Dense Top-100 merged via Reciprocal Rank Fusion (RRF with $k_{rrf}=60$) to form a Candidate Pool of $N=100$.
3. **PoolAdapt (`src/pooladapt/` - Core Research Module)**:
   - Observes pool characteristics: BM25/Dense/RRF ranks and scores, sparse-dense agreement/disagreement, score/rank distributions.
   - Determines adaptive $K$ and selects the candidate subset $S$.
4. **Reranking (`src/reranking/`)**:
   - Heavy Cross-Encoder Reranker operating ONLY on the selected subset $S$ ($\vert{}S\vert{}=K$).
5. **Baselines (`src/baselines/`)**:
   - Full Reranking ($N=100$), Fixed Prefix, Random Selection, PACE, and SAGE-SLO.
6. **Evaluation (`src/evaluation/`)**:
   - Quality: `nDCG@10`, `Recall@10`, `MRR@10`.
   - Efficiency: Reranked pairs/query, compression ratio, P50/P95 latency, $K$ distribution, quality-cost trade-off curves.

# Directory Structure Constraints
Do not alter this structure unless explicitly requested:
src/
├── agent/            # System-level orchestration (router, state, workflow)
├── baselines/        # Comparison methods (Full, Fixed, Random, PACE, SAGE)
├── data/             # Dataset loaders (SciFact, FiQA, HotpotQA via BEIR)
├── evaluation/       # Metrics and evaluation scripts
├── pooladapt/        # CORE RESEARCH CONTRIBUTION (features, policy, selector)
├── reranking/        # Cross-Encoder implementation
└── retrieval/        # BM25, Dense, RRF implementation

# Coding Guidelines
- Write clean, modular, and well-documented Python code.
- Always include type hints and docstrings for core functions.
- Prioritize reproducibility and clean logging for metrics during experiments.