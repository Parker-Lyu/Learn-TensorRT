from __future__ import annotations

import math
import statistics


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("latency samples must not be empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(latency_ms: list[float], elapsed_seconds: float) -> dict:
    if elapsed_seconds <= 0:
        raise ValueError("elapsed time must be positive")
    return {
        "requests": len(latency_ms),
        "latency_ms": {"mean": statistics.fmean(latency_ms),
                       "p50": percentile(latency_ms, 0.50),
                       "p90": percentile(latency_ms, 0.90),
                       "p99": percentile(latency_ms, 0.99)},
        "throughput_requests_per_second": len(latency_ms) / elapsed_seconds,
    }
