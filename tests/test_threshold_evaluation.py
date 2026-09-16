import pytest
from evaluation.evaluator import (
    PairEvaluation,
    ThresholdEvaluator,
    ThresholdMetrics,
    calculate_metrics,
    cosine_similarity,
)


def test_cosine_similarity():
    # Identical vectors
    assert pytest.approx(cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])) == 1.0
    # Orthogonal vectors
    assert pytest.approx(cosine_similarity([1.0, 0.0], [0.0, 1.0])) == 0.0
    # Opposite vectors
    assert pytest.approx(cosine_similarity([2.0, 0.0], [-2.0, 0.0])) == -1.0
    # Proportional vectors
    assert pytest.approx(cosine_similarity([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])) == 1.0
    # Zero or empty vectors
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_calculate_metrics_formulas_and_edge_cases():
    # Test zero cases (no positives)
    empty_metrics = calculate_metrics([], threshold=0.85)
    assert empty_metrics.total == 0
    assert empty_metrics.precision == 0.0
    assert empty_metrics.recall == 0.0
    assert empty_metrics.f1 == 0.0
    assert empty_metrics.accuracy == 0.0

    # Synthetic test pairs:
    # Pair 1: sim 0.90, expected True  -> at 0.85: TP
    # Pair 2: sim 0.80, expected True  -> at 0.85: FN
    # Pair 3: sim 0.90, expected False -> at 0.85: FP
    # Pair 4: sim 0.70, expected False -> at 0.85: TN
    pairs = [
        PairEvaluation("1", "q1", "q2", True, 0.90, "test", "test", False),
        PairEvaluation("2", "q3", "q4", True, 0.80, "test", "test", False),
        PairEvaluation("3", "q5", "q6", False, 0.90, "test", "test", False),
        PairEvaluation("4", "q7", "q8", False, 0.70, "test", "test", False),
    ]

    metrics = calculate_metrics(pairs, threshold=0.85, use_polarity_guard=False)
    assert metrics.tp == 1
    assert metrics.fn == 1
    assert metrics.fp == 1
    assert metrics.tn == 1
    assert metrics.total == 4
    # Precision: 1 / (1 + 1) = 0.5
    assert pytest.approx(metrics.precision) == 0.5
    # Recall: 1 / (1 + 1) = 0.5
    assert pytest.approx(metrics.recall) == 0.5
    # F1: 2 * (0.5 * 0.5) / 1.0 = 0.5
    assert pytest.approx(metrics.f1) == 0.5
    # Accuracy: (1 + 1) / 4 = 0.5
    assert pytest.approx(metrics.accuracy) == 0.5


def test_calculate_metrics_polarity_guard_behavior():
    # Pair with high similarity (0.92) but negation conflict (expected False)
    pair_negation = PairEvaluation(
        id="neg_1",
        query_a="Is it safe?",
        query_b="Is it not safe?",
        expected_match=False,
        similarity=0.92,
        category="negation_conflict",
        domain="safety",
        has_polarity_conflict=True,
    )

    # Without guard -> False Positive at threshold 0.85
    metrics_unguarded = calculate_metrics([pair_negation], threshold=0.85, use_polarity_guard=False)
    assert metrics_unguarded.fp == 1
    assert metrics_unguarded.tn == 0

    # With guard -> Blocked from matching -> True Negative!
    metrics_guarded = calculate_metrics([pair_negation], threshold=0.85, use_polarity_guard=True)
    assert metrics_guarded.fp == 0
    assert metrics_guarded.tn == 1


def test_prd_phase9_evaluation_sweep_on_real_dataset():
    """
    PRD Section 18 Checkpoint:
    - Tests thresholds: 0.75, 0.80, 0.85, 0.90, 0.95
    - Computes TP, FP, TN, FN, Precision, Recall, F1 Score
    - Validates that measured results are populated and quality meets target
    """
    evaluator = ThresholdEvaluator()
    evaluator.load_dataset()
    assert len(evaluator.dataset) >= 100

    evaluator.compute_pair_similarities()
    assert len(evaluator.evaluated_pairs) == len(evaluator.dataset)

    prd_thresholds = [0.75, 0.80, 0.85, 0.90, 0.95]
    sweep_results = evaluator.evaluate_threshold_sweep(
        thresholds=prd_thresholds,
        use_polarity_guard=True,
    )

    assert len(sweep_results) == 5

    for m in sweep_results:
        assert m.threshold in prd_thresholds
        assert m.tp + m.fp + m.tn + m.fn == len(evaluator.dataset)
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.f1 <= 1.0
        assert 0.0 <= m.accuracy <= 1.0

    optimal = evaluator.find_optimal_threshold(sweep_results, min_precision=0.80)
    # The evaluation dataset and model should yield strong discrimination
    assert optimal.f1 >= 0.80
    assert optimal.precision >= 0.80
