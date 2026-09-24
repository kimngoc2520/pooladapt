"""Cross-Encoder reranking shared by the Phase 1 baselines."""

from __future__ import annotations

from collections.abc import Sequence
import time
from typing import Any, Protocol


DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        """Score candidates and return them in descending cross-encoder order."""


class CrossEncoderReranker:
    """Score query-document pairs with a lazily loaded Sentence Transformers model.

    ``model`` may be supplied by tests or callers that manage model loading. It must
    expose ``predict(pairs)`` and return one score per pair.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        top_k: int | None = 10,
        model: Any | None = None,
    ) -> None:
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        self.model_name = model_name
        self.top_k = top_k
        self._model = model
        self.last_latency_rerank_seconds = 0.0
        self.last_latency_ce_seconds = 0.0

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise ImportError(
                    "CrossEncoderReranker requires sentence-transformers; install the project dependencies "
                    "or pass a compatible model for testing."
                ) from error
            self._model = CrossEncoder(self.model_name)
        return self._model

    def warm_up(self) -> None:
        """Load the model once and perform an untimed inference warm-up."""
        self._get_model().predict([("", "")])

    @staticmethod
    def _document_text(candidate: dict[str, Any]) -> str:
        if "text" in candidate:
            return str(candidate["text"])
        if "contents" in candidate:
            return str(candidate["contents"])
        title = str(candidate.get("title", ""))
        return f"{title}\n{candidate.get('document', '')}".strip()

    @staticmethod
    def _document_id(candidate: dict[str, Any]) -> Any:
        if "id" in candidate:
            return candidate["id"]
        if "_id" in candidate:
            return candidate["_id"]
        raise KeyError("each candidate must contain 'id' or '_id'")

    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        # Model construction/download is deliberately outside the timed region.
        # The experiment runner calls ``warm_up`` before processing any query.
        model = self._get_model()
        rerank_started = time.perf_counter()
        candidates = list(candidates)
        limit = self.top_k if top_k is None else top_k
        if limit is not None and limit < 0:
            raise ValueError("top_k must be non-negative or None")
        if not candidates or limit == 0:
            self.last_latency_ce_seconds = 0.0
            self.last_latency_rerank_seconds = time.perf_counter() - rerank_started
            return []

        pairs = [(query, self._document_text(candidate)) for candidate in candidates]
        ce_started = time.perf_counter()
        scores = model.predict(pairs)
        self.last_latency_ce_seconds = time.perf_counter() - ce_started
        if len(scores) != len(candidates):
            raise ValueError("Cross-Encoder returned a score count different from the candidate count")

        ranked = []
        for candidate, score in zip(candidates, scores):
            result = dict(candidate)
            numeric_score = float(score)
            result["id"] = self._document_id(candidate)
            result["score"] = numeric_score
            result["reranker_score"] = numeric_score
            ranked.append(result)
        ranked.sort(key=lambda candidate: candidate["reranker_score"], reverse=True)
        self.last_latency_rerank_seconds = time.perf_counter() - rerank_started
        return ranked if limit is None else ranked[:limit]
