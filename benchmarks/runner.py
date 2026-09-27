import asyncio
from dataclasses import dataclass
import random
import time
from typing import Dict, List, Optional, Tuple

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    Usage,
)
from app.cache.exact_cache import ExactCache, InMemoryExactCache
from app.cache.semantic_cache import SemanticCache
from app.db.qdrant import QdrantService
from app.embeddings.service import EmbeddingService, get_embedding_service
from app.providers.base import LLMProvider
from benchmarks.cost_model import CostModel
from benchmarks.metrics import (
    CostEfficiencySummary,
    LatencySummary,
    calculate_latency_summary,
)


class SimulatedLLMProvider(LLMProvider):
    """
    Simulated upstream LLM provider with realistic token generation timing and payload.
    Emulates upstream network + generation latency (normal distribution: mean=850ms, std=120ms).
    """

    def __init__(
        self,
        mean_latency_ms: float = 850.0,
        std_latency_ms: float = 120.0,
        min_latency_ms: float = 400.0,
    ) -> None:
        self.mean_latency_ms = mean_latency_ms
        self.std_latency_ms = std_latency_ms
        self.min_latency_ms = min_latency_ms

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        # Simulate upstream latency
        sampled_latency_ms = max(
            self.min_latency_ms,
            random.gauss(self.mean_latency_ms, self.std_latency_ms),
        )
        await asyncio.sleep(sampled_latency_ms / 1000.0)

        prompt_text = " ".join(m.content for m in request.messages)
        prompt_tokens = max(10, len(prompt_text.split()) * 2)
        completion_text = f"Simulated authoritative response explaining: {prompt_text[:40]}..."
        completion_tokens = max(25, len(completion_text.split()) * 2)

        return ChatCompletionResponse(
            id=f"chatcmpl-sim-{int(time.time()*1000)}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content=completion_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


@dataclass
class BenchmarkResults:
    """Full benchmark outcome across all scenarios."""
    direct_llm: LatencySummary
    cache_miss: LatencySummary
    cache_hit: LatencySummary
    cost_efficiency: CostEfficiencySummary
    workload_latencies_with_cache: LatencySummary
    workload_latencies_without_cache: LatencySummary
    speedup_ratio: float


class BenchmarkRunner:
    """
    Orchestrates empirical latency and cost benchmarks per PRD Section 20.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        embedding_service: Optional[EmbeddingService] = None,
        threshold: float = 0.75,
    ) -> None:
        self.provider = provider or SimulatedLLMProvider()
        self.embedding_service = embedding_service or get_embedding_service()
        self.threshold = threshold
        self.cost_model = CostModel()

    def _create_isolated_cache(self) -> SemanticCache:
        """Create an isolated, in-memory cache instance for clean benchmark runs."""
        exact_cache = InMemoryExactCache()
        qdrant_service = QdrantService(location=":memory:")
        qdrant_service.ensure_collection()
        return SemanticCache(
            exact_cache=exact_cache,
            embedding_service=self.embedding_service,
            qdrant_service=qdrant_service,
            threshold=self.threshold,
        )

    def _make_request(self, prompt: str, model: str = "openai/gpt-4o-mini") -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
        )

    async def benchmark_direct_llm(self, queries: List[str]) -> Tuple[List[float], List[ChatCompletionResponse]]:
        """Scenario A: Direct LLM (Client -> LLM)."""
        latencies = []
        responses = []
        for q in queries:
            req = self._make_request(q)
            start = time.perf_counter()
            resp = await self.provider.complete(req)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            responses.append(resp)
        return latencies, responses

    async def benchmark_cache_miss(self, queries: List[str]) -> List[float]:
        """Scenario B: Cache Miss (Client -> Gateway -> Embedding -> Qdrant -> LLM)."""
        latencies = []
        # Use fresh cache per query to guarantee cache misses
        for q in queries:
            cache = self._create_isolated_cache()
            req = self._make_request(q)
            start = time.perf_counter()
            cache_res = await cache.lookup(req)
            assert not cache_res.hit
            llm_resp = await self.provider.complete(req)
            await cache.store(req, llm_resp)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
        return latencies

    async def benchmark_cache_hit(
        self,
        base_queries: List[str],
        paraphrases: List[str],
    ) -> List[float]:
        """Scenario C: Cache Hit (Client -> Gateway -> Embedding -> Qdrant -> Cached Response)."""
        cache = self._create_isolated_cache()
        # Seed cache with base queries
        for q in base_queries:
            req = self._make_request(q)
            llm_resp = await self.provider.complete(req)
            await cache.store(req, llm_resp)

        latencies = []
        # Measure hits on paraphrases and exact repeats
        for p in paraphrases:
            req = self._make_request(p)
            start = time.perf_counter()
            cache_res = await cache.lookup(req)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if cache_res.hit:
                latencies.append(elapsed_ms)
        return latencies

    async def run_workload_simulation(
        self,
        workload: List[str],
    ) -> Tuple[List[float], List[float], CostEfficiencySummary]:
        """
        Runs a mixed stream of queries through:
        1. Gateway with caching enabled
        2. Direct LLM without caching
        Calculates end-to-end latency distributions and economic token/cost savings.
        """
        cache = self._create_isolated_cache()

        latencies_with_cache: List[float] = []
        latencies_without_cache: List[float] = []

        hits = 0
        misses = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        avoided_prompt_tokens = 0
        avoided_completion_tokens = 0

        for query_text in workload:
            req = self._make_request(query_text)

            # Baseline without cache
            t0 = time.perf_counter()
            direct_resp = await self.provider.complete(req)
            latencies_without_cache.append((time.perf_counter() - t0) * 1000.0)

            p_toks = direct_resp.usage.prompt_tokens if direct_resp.usage else 15
            c_toks = direct_resp.usage.completion_tokens if direct_resp.usage else 35
            total_prompt_tokens += p_toks
            total_completion_tokens += c_toks

            # With cache
            t1 = time.perf_counter()
            cache_res = await cache.lookup(req)
            if cache_res.hit:
                hits += 1
                avoided_prompt_tokens += p_toks
                avoided_completion_tokens += c_toks
            else:
                misses += 1
                llm_resp = await self.provider.complete(req)
                await cache.store(req, llm_resp)

            latencies_with_cache.append((time.perf_counter() - t1) * 1000.0)

        cost_summary = self.cost_model.summarize_workload(
            total_requests=len(workload),
            cache_hits=hits,
            cache_misses=misses,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            avoided_prompt_tokens=avoided_prompt_tokens,
            avoided_completion_tokens=avoided_completion_tokens,
        )

        return latencies_with_cache, latencies_without_cache, cost_summary

    async def run_full_benchmark(
        self,
        base_queries: List[str],
        paraphrases: List[str],
        sample_count: int = 30,
    ) -> BenchmarkResults:
        """
        Executes the complete benchmark suite across Scenarios A, B, C and mixed workload.
        """
        # Warmup embedding model
        self.embedding_service.embed("warmup query")

        # 1. Benchmark Scenario A: Direct LLM
        sample_queries = (base_queries * ((sample_count // len(base_queries)) + 1))[:sample_count]
        lats_direct, _ = await self.benchmark_direct_llm(sample_queries)
        summary_direct = calculate_latency_summary(lats_direct)

        # 2. Benchmark Scenario B: Cache Miss
        lats_miss = await self.benchmark_cache_miss(sample_queries)
        summary_miss = calculate_latency_summary(lats_miss)

        # 3. Benchmark Scenario C: Cache Hit
        lats_hit = await self.benchmark_cache_hit(base_queries, paraphrases)
        summary_hit = calculate_latency_summary(lats_hit)

        # 4. Mixed Realistic Workload Simulation
        # Mix consists of original base queries (misses then hits), paraphrases (semantic hits), repeats
        workload = []
        for b, p in zip(base_queries, paraphrases):
            workload.extend([b, p, b])  # base (miss) -> paraphrase (semantic hit) -> base (exact hit)
        random.seed(42)
        random.shuffle(workload)
        workload = workload[:sample_count * 2]

        lats_with_cache, lats_without_cache, cost_summary = await self.run_workload_simulation(workload)
        summary_workload_with = calculate_latency_summary(lats_with_cache)
        summary_workload_without = calculate_latency_summary(lats_without_cache)

        speedup = (
            round(summary_direct.p50_ms / summary_hit.p50_ms, 1)
            if summary_hit.p50_ms > 0
            else 0.0
        )

        return BenchmarkResults(
            direct_llm=summary_direct,
            cache_miss=summary_miss,
            cache_hit=summary_hit,
            cost_efficiency=cost_summary,
            workload_latencies_with_cache=summary_workload_with,
            workload_latencies_without_cache=summary_workload_without,
            speedup_ratio=speedup,
        )
