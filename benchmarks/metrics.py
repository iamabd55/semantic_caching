from dataclasses import dataclass
import math
from typing import List


@dataclass
class LatencySummary:
    """Statistical summary of request latencies in milliseconds."""
    samples: int
    avg_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class CostEfficiencySummary:
    """Summary of cost and token savings achieved by caching."""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate_pct: float
    calls_avoided: int
    calls_avoided_pct: float
    tokens_avoided: int
    tokens_total_without_cache: int
    tokens_total_with_cache: int
    estimated_cost_without_cache_usd: float
    estimated_cost_with_cache_usd: float
    estimated_cost_savings_usd: float
    estimated_cost_savings_pct: float


def calculate_percentile(sorted_values: List[float], percentile: float) -> float:
    """
    Computes a percentile value using linear interpolation between nearest ranks.
    percentile should be between 0.0 and 100.0.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    k = (len(sorted_values) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return d0 + d1


def calculate_latency_summary(latencies: List[float]) -> LatencySummary:
    """
    Computes average, min, max, P50, P90, P95, and P99 for a list of latency values.
    """
    if not latencies:
        return LatencySummary(
            samples=0,
            avg_ms=0.0,
            p50_ms=0.0,
            p90_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
        )

    sorted_lats = sorted(latencies)
    avg_ms = sum(sorted_lats) / len(sorted_lats)

    return LatencySummary(
        samples=len(sorted_lats),
        avg_ms=round(avg_ms, 2),
        p50_ms=round(calculate_percentile(sorted_lats, 50.0), 2),
        p90_ms=round(calculate_percentile(sorted_lats, 90.0), 2),
        p95_ms=round(calculate_percentile(sorted_lats, 95.0), 2),
        p99_ms=round(calculate_percentile(sorted_lats, 99.0), 2),
        min_ms=round(sorted_lats[0], 2),
        max_ms=round(sorted_lats[-1], 2),
    )
