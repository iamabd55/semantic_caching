from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.cache.polarity_guard import has_polarity_conflict
from app.embeddings.service import EmbeddingService, get_embedding_service


@dataclass
class ThresholdMetrics:
    """Classification metrics for a specific similarity threshold."""
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    total: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    category_breakdown: Dict[str, Dict[str, int]]


@dataclass
class PairEvaluation:
    """Evaluation result for a single query pair."""
    id: str
    query_a: str
    query_b: str
    expected_match: bool
    similarity: float
    category: str
    domain: str
    has_polarity_conflict: bool


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Computes cosine similarity between two 1D float vectors:
    dot(a, b) / (norm(a) * norm(b))
    """
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0

    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for a, b in zip(vec_a, vec_b):
        dot_product += a * b
        norm_a_sq += a * a
        norm_b_sq += b * b

    norm_a = math.sqrt(norm_a_sq)
    norm_b = math.sqrt(norm_b_sq)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def calculate_metrics(
    pairs: List[PairEvaluation],
    threshold: float,
    use_polarity_guard: bool = True,
) -> ThresholdMetrics:
    """
    Computes confusion matrix and classification metrics (precision, recall, F1, accuracy)
    for a given threshold.
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    category_breakdown: Dict[str, Dict[str, int]] = {}

    for pair in pairs:
        # Determine prediction: meets threshold and (optionally) passes polarity guard
        raw_match = pair.similarity >= threshold
        if use_polarity_guard and raw_match and pair.has_polarity_conflict:
            pred_match = False
        else:
            pred_match = raw_match

        actual_match = pair.expected_match
        cat = pair.category
        if cat not in category_breakdown:
            category_breakdown[cat] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        if pred_match and actual_match:
            tp += 1
            category_breakdown[cat]["tp"] += 1
        elif pred_match and not actual_match:
            fp += 1
            category_breakdown[cat]["fp"] += 1
        elif not pred_match and not actual_match:
            tn += 1
            category_breakdown[cat]["tn"] += 1
        else:
            fn += 1
            category_breakdown[cat]["fn"] += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return ThresholdMetrics(
        threshold=round(threshold, 4),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        total=total,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        category_breakdown=category_breakdown,
    )


class ThresholdEvaluator:
    """
    Threshold evaluation engine for semantic cache.
    Loads dataset, embeds query pairs, sweeps thresholds, and evaluates metrics.
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self.dataset_path = (
            dataset_path
            or Path(__file__).parent / "dataset" / "cache_eval.json"
        )
        self.embedding_service = embedding_service or get_embedding_service()
        self.dataset: List[Dict[str, Any]] = []
        self.evaluated_pairs: List[PairEvaluation] = []
        self.average_embedding_latency_ms: float = 0.0

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load labeled evaluation pairs from disk."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        return self.dataset

    def compute_pair_similarities(self) -> List[PairEvaluation]:
        """
        Embed all query pairs in the dataset and pre-compute cosine similarities
        and polarity conflict status.
        """
        import time

        if not self.dataset:
            self.load_dataset()

        # Batch encode all unique queries to minimize inference passes
        all_queries = set()
        for item in self.dataset:
            all_queries.add(item["query_a"])
            all_queries.add(item["query_b"])

        query_list = list(all_queries)
        start_time = time.perf_counter()
        embeddings = self.embedding_service.embed_batch(query_list)
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.average_embedding_latency_ms = (
            total_time_ms / len(query_list) if query_list else 0.0
        )
        embedding_map = {q: emb for q, emb in zip(query_list, embeddings)}

        evaluated = []
        for item in self.dataset:
            vec_a = embedding_map[item["query_a"]]
            vec_b = embedding_map[item["query_b"]]
            sim = cosine_similarity(vec_a, vec_b)
            polarity_conflict = has_polarity_conflict(item["query_a"], item["query_b"])

            evaluated.append(
                PairEvaluation(
                    id=item["id"],
                    query_a=item["query_a"],
                    query_b=item["query_b"],
                    expected_match=item["expected_match"],
                    similarity=sim,
                    category=item.get("category", "default"),
                    domain=item.get("domain", "default"),
                    has_polarity_conflict=polarity_conflict,
                )
            )

        self.evaluated_pairs = evaluated
        return self.evaluated_pairs

    def evaluate_threshold_sweep(
        self,
        thresholds: Optional[List[float]] = None,
        use_polarity_guard: bool = True,
    ) -> List[ThresholdMetrics]:
        """
        Sweeps through candidate thresholds and computes metrics for each.
        """
        if not self.evaluated_pairs:
            self.compute_pair_similarities()

        if thresholds is None:
            thresholds = [0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.95]

        results = []
        for t in thresholds:
            metrics = calculate_metrics(
                self.evaluated_pairs,
                threshold=t,
                use_polarity_guard=use_polarity_guard,
            )
            results.append(metrics)
        return results

    def find_optimal_threshold(
        self,
        metrics_list: List[ThresholdMetrics],
        min_precision: float = 0.85,
    ) -> ThresholdMetrics:
        """
        Selects the optimal threshold.
        Prioritizes maximizing F1 score subject to a minimum precision constraint
        (since serving wrong cached responses is worse than a cache miss).
        """
        eligible = [m for m in metrics_list if m.precision >= min_precision]
        candidates = eligible if eligible else metrics_list

        # Pick candidate with highest F1; on tie, highest precision
        optimal = max(candidates, key=lambda m: (m.f1, m.precision))
        return optimal
