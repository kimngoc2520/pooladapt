# PoolAdapt

PoolAdapt studies candidate-level selective reranking for large candidate pools in agentic hybrid RAG under inference cost constraints. The project investigates whether sparse-dense retrieval characteristics can guide reranking allocation while preserving retrieval quality and reducing Cross-Encoder inference cost.

## Research Question

Can candidate-pool characteristics derived from sparse-dense retrieval signals be used to selectively rerank candidates in large hybrid candidate pools while reducing inference cost with minimal retrieval-quality degradation?

## Architecture

```text
User Query
    |
Agentic Router / Workflow
    |
Hybrid Retrieval
    |-- BM25 Top-100
    `-- Dense Top-100
            |
        RRF Fusion (k=60)
            |
    Large Candidate Pool (N=100)
            |
Candidate-Pool Characterization
            |
    PoolAdapt Candidate Selector
            |
    Selected M candidates
            |
    Cross-Encoder Reranker
            |
          Top-K
            |
       LLM / Agent Generation
            |
          Answer
```

The terms in this project have deliberately narrow roles:

- **Agentic Hybrid RAG** is the system architecture and its lightweight workflow layer.
- **PoolAdapt** is the research contribution: candidate-level selection within the fused pool.
- **Selective reranking** is the mechanism used to allocate Cross-Encoder work.
- **Inference cost** is the central constraint, measured primarily through reranked pairs and latency.

The agent layer coordinates a fixed pipeline only. It does not add query rewriting, web search, multi-hop reasoning, memory, or planning.

## Baselines

- Full Rerank
- Fixed Prefix
- Random Selection
- PACE: "Less can be More: Relieving RAG Bottlenecks via Evidence Frontloading and Pressure-Adaptive Budgeting"
- SAGE-SLO: "SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems"
- PoolAdapt

## Datasets

- SciFact: pilot, debugging, candidate analysis, and budget sweep
- FiQA: validation and generalization
- HotpotQA: stress test and generalization
- MS MARCO: optional, only if time remains

## Evaluation

- Quality: nDCG@10, Recall@10, MRR@10
- Efficiency: rerank pairs per query, compression ratio, and reranker input tokens per query when measurable
- Latency: retrieval, selector, reranking, and end-to-end latency, including P50 and P95
- Trade-off: quality retention, quality at comparable budget, and quality-cost curves

## Project Status

Repository skeleton and research protocol initialization.
