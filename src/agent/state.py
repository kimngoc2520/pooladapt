"""State passed through the lightweight RAG workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineState:
    query: str
    retrieved_candidates: list[dict[str, Any]] = field(default_factory=list)
    candidate_features: list[dict[str, Any]] = field(default_factory=list)
    selected_candidates: list[dict[str, Any]] = field(default_factory=list)
    reranked_documents: list[dict[str, Any]] = field(default_factory=list)
    final_context: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None
