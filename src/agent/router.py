"""Minimal controller for the fixed Agentic Hybrid RAG stage sequence."""

from __future__ import annotations

from .state import PipelineState
from .workflow import HybridRAGWorkflow


class AgenticRouter:
    """Routes a query through one fixed workflow; it does not transform queries."""

    def __init__(self, workflow: HybridRAGWorkflow) -> None:
        self.workflow = workflow

    def run(self, query: str) -> PipelineState:
        return self.workflow.run(PipelineState(query=query))
