"""In-memory BM25 retrieval for BEIR-style corpora."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Use a small deterministic tokenizer suitable for the baseline."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Retrieve documents from a BEIR corpus using BM25Okapi."""

    def __init__(self, corpus: Mapping[str, Mapping[str, Any]]) -> None:
        self.document_ids = list(corpus)
        self._tokenized_corpus = [
            tokenize(f"{document.get('title', '')} {document.get('text', '')}")
            for document in corpus.values()
        ]
        self._index = BM25Okapi(self._tokenized_corpus)

    def retrieve(self, query: str, top_n: int = 100) -> list[dict[str, Any]]:
        if top_n < 0:
            raise ValueError("top_n must be non-negative")
        scores = self._index.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(self.document_ids)), key=lambda index: (-scores[index], self.document_ids[index]))
        return [
            {"id": self.document_ids[index], "score": float(scores[index]), "rank": rank}
            for rank, index in enumerate(ranked_indices[:top_n], start=1)
        ]
