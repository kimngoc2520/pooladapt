"""Small ranking-quality metrics used by the Phase 1 baselines."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _relevance(relevance: Mapping[Any, float] | Iterable[Any], document_id: Any) -> float:
	if isinstance(relevance, Mapping):
		return float(relevance.get(document_id, 0.0))
	return 1.0 if document_id in set(relevance) else 0.0


def ndcg_at_k(ranked_ids: Sequence[Any], relevance: Mapping[Any, float] | Iterable[Any], k: int = 10) -> float:
	if k <= 0:
		raise ValueError("k must be positive")
	relevance = relevance if isinstance(relevance, Mapping) else {document_id: 1.0 for document_id in relevance}
	gains = [_relevance(relevance, document_id) for document_id in ranked_ids[:k]]
	dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
	ideal = sorted((_relevance(relevance, document_id) for document_id in relevance), reverse=True)[:k]
	idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
	return dcg / idcg if idcg else 0.0


def recall_at_k(ranked_ids: Sequence[Any], relevant_ids: Iterable[Any], k: int = 10) -> float:
	relevant = set(relevant_ids)
	return len(set(ranked_ids[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked_ids: Sequence[Any], relevant_ids: Iterable[Any], k: int = 10) -> float:
	relevant = set(relevant_ids)
	for rank, document_id in enumerate(ranked_ids[:k], start=1):
		if document_id in relevant:
			return 1.0 / rank
	return 0.0
