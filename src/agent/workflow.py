"""Dependency-injected orchestration for Agentic Hybrid RAG."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .state import PipelineState

Candidates = Sequence[dict[str, Any]]


class HybridRAGWorkflow:
    """Run retrieve → characterize → select → rerank → generate.

    Each stage is injected so this layer remains independent of PoolAdapt and model
    implementations. The workflow deliberately contains no planning or query rewriting.
    """

    def __init__(
        self,
        retrieve: Callable[[str], Candidates],
        characterize: Callable[[Candidates], Candidates],
        select: Callable[[Candidates], Candidates],
        rerank: Callable[[str, Candidates], Candidates],
        generate: Callable[[str, Candidates], str] | None = None,
        top_k: int = 10,
    ) -> None:
        self.retrieve, self.characterize, self.select = retrieve, characterize, select
        self.rerank, self.generate, self.top_k = rerank, generate, top_k

    def run(self, state: PipelineState) -> PipelineState:
        state.retrieved_candidates = list(self.retrieve(state.query))
        state.candidate_features = list(self.characterize(state.retrieved_candidates))
        state.selected_candidates = list(self.select(state.candidate_features))
        state.reranked_documents = list(self.rerank(state.query, state.selected_candidates))
        state.final_context = state.reranked_documents[: self.top_k]
        if self.generate is not None:
            state.answer = self.generate(state.query, state.final_context)
        return state
