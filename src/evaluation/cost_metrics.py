"""Cross-Encoder pair-count and candidate-pool cost metrics."""

from __future__ import annotations

from collections.abc import Iterable


def reranked_pairs_per_query(selected_candidates: int | Iterable[object]) -> int:
	"""Return the number of query-document pairs sent to the Cross-Encoder."""
	if isinstance(selected_candidates, int):
		count = selected_candidates
	else:
		count = sum(1 for _ in selected_candidates)
	if count < 0:
		raise ValueError("reranked pair count must be non-negative")
	return count


def average_reranked_pairs(pair_counts: Iterable[int]) -> float:
	counts = list(pair_counts)
	if not counts:
		return 0.0
	if any(count < 0 for count in counts):
		raise ValueError("reranked pair counts must be non-negative")
	return sum(counts) / len(counts)


def compression_ratio(candidate_pool_size: int, reranked_pairs: int) -> float:
	if candidate_pool_size <= 0:
		raise ValueError("candidate_pool_size must be positive")
	if not 0 <= reranked_pairs <= candidate_pool_size:
		raise ValueError("reranked_pairs must be within the candidate-pool size")
	return 1.0 - reranked_pairs / candidate_pool_size


def total_reranking_pairs(pair_counts: Iterable[int]) -> int:
	counts = list(pair_counts)
	if any(count < 0 for count in counts):
		raise ValueError("reranked pair counts must be non-negative")
	return sum(counts)
