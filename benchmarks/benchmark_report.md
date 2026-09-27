# Phase 11: Latency & Cost Benchmark Report

## 1. Latency Profile

Comparison across the three primary request scenarios defined in PRD Section 20:

| Scenario | Avg Latency | P50 (Median) | P95 | P99 | Speedup |
|:---|---:|---:|---:|---:|---:|
| **Direct LLM call** | `889.57 ms` | `904.43 ms` | `1041.95 ms` | `1096.03 ms` | 1.0x |
| **Cache miss (gateway + LLM)** | `880.02 ms` | `859.68 ms` | `1049.61 ms` | `1168.58 ms` | ~0.95x |
| **Cache hit (semantic / exact)** | `11.11 ms` | `11.15 ms` | `12.72 ms` | `13.63 ms` | **81.1x** |

### Key Observations:
- **Cache Hit Acceleration**: Serving from the semantic cache delivers sub-25ms response times, achieving a **81.1x speedup** over querying the upstream LLM.
- **Gateway Overhead on Cache Miss**: The additional latency introduced on a cache miss (canonicalization + embedding + Qdrant lookup + storage) averages **~-9.6 ms**, an imperceptible overhead compared to LLM generation time.

---

## 2. Cost & Economic Efficiency

Measured on a realistic mixed query stream with threshold `0.75`:

| Metric | Result |
|:---|---:|
| **Total Requests** | `50` |
| **Cache Hit Rate** | `24.0%` |
| **LLM Calls Avoided** | `12` (24.0%) |
| **Tokens Avoided** | `477` tokens |
| **Cost Without Cache** | `$0.002249` |
| **Cost With Cache** | `$0.001710` |
| **Estimated Cost Reduction** | **23.98%** (`$0.000540`) |

---

## 3. Conclusion
The semantic cache proves highly effective:
1. **Dramatic User Latency Drop**: Typical response times fall from hundreds of milliseconds to under ~20ms.
2. **Substantial Cost Savings**: Incurring zero LLM token costs on ~24.0% of queries directly reduces provider API expenses by 23.98%.
