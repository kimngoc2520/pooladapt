"""Aggregate reranking-stage and cross-encoder latency samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .efficiency_metrics import latency_summary


SUMMARY_KEYS = ("mean", "p50", "p95")


def summarize_latency_samples(
    rerank_samples: Sequence[float], ce_samples: Sequence[float]
) -> dict[str, dict[str, float]]:
    """Summarize paired L_rerank and diagnostic L_CE samples for one method."""
    if len(rerank_samples) != len(ce_samples):
        raise ValueError("rerank and cross-encoder latency sample counts must match")
    if not rerank_samples:
        raise ValueError("at least one latency sample is required")
    return {
        "rerank_latency_seconds": latency_summary(rerank_samples),
        "ce_latency_seconds": latency_summary(ce_samples),
    }


def latency_reduction(
    method_rerank_latency: Mapping[str, float], full_rerank_latency: Mapping[str, float]
) -> dict[str, float]:
    """Calculate like-for-like L_rerank reductions against full reranking."""
    reduction: dict[str, float] = {}
    for statistic in SUMMARY_KEYS:
        try:
            method_value = float(method_rerank_latency[statistic])
            full_value = float(full_rerank_latency[statistic])
        except KeyError as error:
            raise ValueError(f"latency summary is missing {statistic!r}") from error
        if full_value == 0:
            raise ValueError(f"full_rerank {statistic} latency must be non-zero for reduction")
        reduction[statistic] = 1.0 - method_value / full_value
    return reduction


def method_latency_samples(predictions: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    """Extract paired latency fields from one method's prediction records."""
    rerank_samples: list[float] = []
    ce_samples: list[float] = []
    for prediction in predictions:
        if "latency_rerank_seconds" not in prediction or "latency_ce_seconds" not in prediction:
            raise ValueError("each prediction must contain rerank and cross-encoder latency fields")
        rerank_samples.append(float(prediction["latency_rerank_seconds"]))
        ce_samples.append(float(prediction["latency_ce_seconds"]))
    return rerank_samples, ce_samples


def aggregate_latency_methods(methods: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate all methods and calculate non-reference L_rerank reductions."""
    if "full_rerank" not in methods:
        raise ValueError("full_rerank is required as the latency-reduction reference")

    summaries: dict[str, dict[str, Any]] = {}
    for method_name, method_data in methods.items():
        predictions = method_data.get("predictions")
        if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
            raise ValueError(f"Method {method_name!r} must contain a predictions sequence")
        rerank_samples, ce_samples = method_latency_samples(predictions)
        summaries[method_name] = summarize_latency_samples(rerank_samples, ce_samples)

    full_rerank_latency = summaries["full_rerank"]["rerank_latency_seconds"]
    for method_name, summary in summaries.items():
        summary["latency_reduction"] = (
            None
            if method_name == "full_rerank"
            else latency_reduction(summary["rerank_latency_seconds"], full_rerank_latency)
        )
    return summaries


def build_latency_report(prediction_data: Mapping[str, Any]) -> dict[str, Any]:
    """Build the JSON-ready latency report from baseline prediction data."""
    methods = prediction_data.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("baseline predictions must contain a methods mapping")
    return {
        "dataset": prediction_data.get("dataset"),
        "candidate_pool_size": prediction_data.get("candidate_pool_size"),
        "top_k": prediction_data.get("top_k"),
        "methods": aggregate_latency_methods(methods),
    }
