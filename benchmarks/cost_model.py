from dataclasses import dataclass
from typing import List
from benchmarks.metrics import CostEfficiencySummary


@dataclass
class ModelPricing:
    """Pricing configuration in USD per 1,000,000 tokens."""
    prompt_price_per_m: float = 0.50       # $0.50 per 1M input tokens
    completion_price_per_m: float = 1.50   # $1.50 per 1M output tokens


class CostModel:
    """
    Computes economic savings, token avoidance, and dollar cost reductions (PRD Section 20).
    """

    def __init__(self, pricing: ModelPricing | None = None) -> None:
        self.pricing = pricing or ModelPricing()

    def estimate_tokens(self, text: str) -> int:
        """
        Approximate token count (rule of thumb: ~4 characters per token in English).
        """
        return max(1, len(text.strip()) // 4)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate USD cost for given token counts."""
        prompt_cost = (prompt_tokens / 1_000_000.0) * self.pricing.prompt_price_per_m
        completion_cost = (completion_tokens / 1_000_000.0) * self.pricing.completion_price_per_m
        return prompt_cost + completion_cost

    def summarize_workload(
        self,
        total_requests: int,
        cache_hits: int,
        cache_misses: int,
        total_prompt_tokens: int,
        total_completion_tokens: int,
        avoided_prompt_tokens: int,
        avoided_completion_tokens: int,
    ) -> CostEfficiencySummary:
        """
        Produce a CostEfficiencySummary comparing total expenditure without caching vs with caching.
        """
        hit_rate_pct = (cache_hits / total_requests * 100.0) if total_requests > 0 else 0.0
        calls_avoided = cache_hits
        calls_avoided_pct = (calls_avoided / total_requests * 100.0) if total_requests > 0 else 0.0

        # Baseline: All requests go to LLM without cache
        cost_without_cache = self.calculate_cost(total_prompt_tokens, total_completion_tokens)

        # Actual: Only cache misses incur LLM token costs
        actual_prompt_tokens = total_prompt_tokens - avoided_prompt_tokens
        actual_completion_tokens = total_completion_tokens - avoided_completion_tokens
        cost_with_cache = self.calculate_cost(actual_prompt_tokens, actual_completion_tokens)

        savings_usd = max(0.0, cost_without_cache - cost_with_cache)
        savings_pct = (savings_usd / cost_without_cache * 100.0) if cost_without_cache > 0 else 0.0

        tokens_avoided = avoided_prompt_tokens + avoided_completion_tokens
        tokens_total_without = total_prompt_tokens + total_completion_tokens
        tokens_total_with = actual_prompt_tokens + actual_completion_tokens

        return CostEfficiencySummary(
            total_requests=total_requests,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            hit_rate_pct=round(hit_rate_pct, 2),
            calls_avoided=calls_avoided,
            calls_avoided_pct=round(calls_avoided_pct, 2),
            tokens_avoided=tokens_avoided,
            tokens_total_without_cache=tokens_total_without,
            tokens_total_with_cache=tokens_total_with,
            estimated_cost_without_cache_usd=round(cost_without_cache, 6),
            estimated_cost_with_cache_usd=round(cost_with_cache, 6),
            estimated_cost_savings_usd=round(savings_usd, 6),
            estimated_cost_savings_pct=round(savings_pct, 2),
        )
