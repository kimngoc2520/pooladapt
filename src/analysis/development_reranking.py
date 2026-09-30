"""Validation helpers for Phase 2 SciFact development reranking artifacts."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


METHOD_BUDGETS = {"full_rerank": 100, "fixed_prefix_10": 10, "fixed_prefix_20": 20, "fixed_prefix_30": 30, "fixed_prefix_50": 50, "fixed_prefix_100": 100}
PREDICTION_FIELDS = {"query_id", "document_ids", "reranked_pairs", "latency_rerank_seconds", "latency_ce_seconds"}


def validate_artifact(artifact: Mapping[str, Any], expected_ids: set[str], split: str) -> dict[str, Any]:
    """Reject missing methods, ID misalignment, duplicates, or absent timing fields."""
    if artifact.get("dataset") != "scifact" or artifact.get("split") != split:
        raise ValueError("artifact dataset/split metadata is incorrect")
    if artifact.get("candidate_pool_size") != 100 or artifact.get("top_k") != 10:
        raise ValueError("artifact must use candidate_pool_size=100 and top_k=10")
    methods = artifact.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(METHOD_BUDGETS):
        raise ValueError("artifact does not contain exactly the required reranking methods")
    rankings: dict[str, dict[str, list[str]]] = {}
    for name, budget in METHOD_BUDGETS.items():
        predictions = methods[name].get("predictions") if isinstance(methods[name], Mapping) else None
        if not isinstance(predictions, list):
            raise ValueError(f"{name} lacks predictions")
        indexed: dict[str, list[str]] = {}
        for record in predictions:
            if not isinstance(record, Mapping) or not PREDICTION_FIELDS <= record.keys():
                raise ValueError(f"{name} has an incomplete prediction record")
            query_id = str(record["query_id"])
            if query_id in indexed:
                raise ValueError(f"{name} contains duplicate query ID {query_id!r}")
            if not isinstance(record["document_ids"], list) or len(record["document_ids"]) > 10:
                raise ValueError(f"{name}, {query_id}: invalid document_ids")
            if not isinstance(record["reranked_pairs"], int) or record["reranked_pairs"] != budget:
                raise ValueError(f"{name}, {query_id}: incorrect reranked_pairs")
            if not all(isinstance(record[field], float) for field in ("latency_rerank_seconds", "latency_ce_seconds")):
                raise ValueError(f"{name}, {query_id}: latency fields must be floats")
            indexed[query_id] = [str(document_id) for document_id in record["document_ids"]]
        if set(indexed) != expected_ids:
            raise ValueError(f"{name} query IDs do not exactly match the {split} split")
        rankings[name] = indexed
    equal_ids = [query_id for query_id in sorted(expected_ids) if rankings["full_rerank"][query_id] == rankings["fixed_prefix_100"][query_id]]
    return {"query_count": len(expected_ids), "full_rerank_equals_fixed_prefix_100": len(equal_ids) == len(expected_ids), "equal_query_count": len(equal_ids), "different_query_ids": sorted(expected_ids - set(equal_ids))}
