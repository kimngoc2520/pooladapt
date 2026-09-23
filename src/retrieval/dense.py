"""Sentence-transformer dense retrieval for BEIR-style corpora."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    """Encode a corpus once and retrieve by cosine similarity.

    ``model`` can be injected in tests or experiments.  When omitted, the
    standard compact sentence-transformer is loaded lazily.
    """

    def __init__(
        self,
        corpus: Mapping[str, Mapping[str, Any]],
        model_name: str = DEFAULT_MODEL,
        model: Any | None = None,
        batch_size: int = 32,
    ) -> None:
        self.document_ids = list(corpus)
        self.model_name = model_name
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
        self._model = model
        documents = [f"{document.get('title', '')} {document.get('text', '')}" for document in corpus.values()]
        self._document_embeddings = self._normalize(self._encode(documents, batch_size=batch_size))

    def _encode(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        kwargs: dict[str, Any] = {"convert_to_numpy": True, "show_progress_bar": False}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        return np.asarray(self._model.encode(texts, **kwargs), dtype=np.float32)

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.atleast_2d(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def retrieve(self, query: str, top_n: int = 100) -> list[dict[str, Any]]:
        if top_n < 0:
            raise ValueError("top_n must be non-negative")
        query_embedding = self._normalize(self._encode([query]))[0]
        scores = self._document_embeddings @ query_embedding
        ranked_indices = sorted(range(len(self.document_ids)), key=lambda index: (-scores[index], self.document_ids[index]))
        return [
            {"id": self.document_ids[index], "score": float(scores[index]), "rank": rank}
            for rank, index in enumerate(ranked_indices[:top_n], start=1)
        ]
