import pytest
from benchmarks.cost_model import CostModel, ModelPricing
from benchmarks.metrics import calculate_latency_summary, calculate_percentile
from benchmarks.runner import BenchmarkRunner, SimulatedLLMProvider


def test_calculate_percentiles_math():
    # Empty case
    empty_summary = calculate_latency_summary([])
    assert empty_summary.samples == 0
    assert empty_summary.avg_ms == 0.0
    assert empty_summary.p50_ms == 0.0

    # Single element
    single = [42.0]
    s_summary = calculate_latency_summary(single)
    assert s_summary.samples == 1
    assert s_summary.p50_ms == 42.0
    assert s_summary.p95_ms == 42.0
    assert s_summary.avg_ms == 42.0

    # Ordered values 1 to 100
    values = [float(i) for i in range(1, 101)]
    summary = calculate_latency_summary(values)
    assert summary.samples == 100
    assert summary.min_ms == 1.0
    assert summary.max_ms == 100.0
    assert summary.avg_ms == 50.5
    # P50 (median) should be around 50.5
    assert pytest.approx(summary.p50_ms, abs=0.5) == 50.5
    # P95 should be around 95.05
    assert pytest.approx(summary.p95_ms, abs=1.0) == 95.05


def test_cost_model_calculations():
    pricing = ModelPricing(prompt_price_per_m=1.0, completion_price_per_m=2.0)
    model = CostModel(pricing=pricing)

    # 1,000,000 prompt tokens -> $1.0; 500,000 completion tokens -> $1.0. Total = $2.0
    cost = model.calculate_cost(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert pytest.approx(cost) == 2.0

    # Workload summary
    summary = model.summarize_workload(
        total_requests=10,
        cache_hits=4,
        cache_misses=6,
        total_prompt_tokens=1000,
        total_completion_tokens=2000,
        avoided_prompt_tokens=400,
        avoided_completion_tokens=800,
    )
    assert summary.total_requests == 10
    assert summary.cache_hits == 4
    assert summary.cache_misses == 6
    assert summary.hit_rate_pct == 40.0
    assert summary.calls_avoided == 4
    assert summary.calls_avoided_pct == 40.0
    assert summary.tokens_avoided == 1200
    assert summary.estimated_cost_savings_pct == 40.0


@pytest.mark.asyncio
async def test_benchmark_runner_mini_run():
    # Fast simulated provider for unit test (mean 20ms)
    fast_provider = SimulatedLLMProvider(mean_latency_ms=25.0, std_latency_ms=2.0, min_latency_ms=10.0)
    runner = BenchmarkRunner(provider=fast_provider, threshold=0.75)

    base = ["What is python?", "Explain SQL."]
    paraphrases = ["Tell me what python is.", "Explain SQL databases."]

    results = await runner.run_full_benchmark(
        base_queries=base,
        paraphrases=paraphrases,
        sample_count=4,
    )

    assert results.direct_llm.samples > 0
    assert results.cache_miss.samples > 0
    assert results.cache_hit.samples > 0

    # Cache hit should return fast (< 100ms)
    assert results.cache_hit.avg_ms > 0
    # Direct LLM calls should register simulated delay
    assert results.direct_llm.avg_ms >= 10.0
    # Cost savings should be recorded for workload
    assert results.cost_efficiency.total_requests > 0
    assert results.cost_efficiency.hit_rate_pct >= 0.0
