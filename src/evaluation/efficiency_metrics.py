"""Timing summaries for reranking experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import time


def percentile(values: Iterable[float], percentile_rank: float) -> float:
	samples = sorted(float(value) for value in values)
	if not samples:
		raise ValueError("at least one latency sample is required")
	if not 0 <= percentile_rank <= 100:
		raise ValueError("percentile rank must be between 0 and 100")
	position = (len(samples) - 1) * percentile_rank / 100
	lower = int(position)
	upper = min(lower + 1, len(samples) - 1)
	return samples[lower] + (samples[upper] - samples[lower]) * (position - lower)


def latency_summary(latencies_seconds: Iterable[float]) -> dict[str, float]:
	samples = [float(value) for value in latencies_seconds]
	if not samples:
		raise ValueError("at least one latency sample is required")
	return {
		"mean": sum(samples) / len(samples),
		"p50": percentile(samples, 50),
		"p95": percentile(samples, 95),
	}


def measure_latency(operation: Callable[[], object], repetitions: int = 1) -> dict[str, float]:
	"""Measure wall-clock duration around each operation call using ``perf_counter``."""
	if repetitions <= 0:
		raise ValueError("repetitions must be positive")
	samples = []
	for _ in range(repetitions):
		started = time.perf_counter()
		operation()
		samples.append(time.perf_counter() - started)
	return latency_summary(samples)
