from pathlib import Path
import pytest
from mlflow.tracking import MlflowClient

from evaluation.evaluator import ThresholdMetrics
from evaluation.mlflow_tracker import MLflowTracker, compute_dataset_version


@pytest.fixture
def temp_mlflow_tracker(tmp_path):
    """Provides an isolated MLflow tracker instance with a temporary SQLite database."""
    db_path = tmp_path / "test_mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    tracker = MLflowTracker(
        experiment_name="test-threshold-evaluation",
        tracking_uri=tracking_uri,
    )
    return tracker


def test_compute_dataset_version(tmp_path):
    # Test valid file
    sample_file = tmp_path / "data.json"
    sample_file.write_text('{"sample": "data"}', encoding="utf-8")

    version_1 = compute_dataset_version(sample_file)
    version_2 = compute_dataset_version(sample_file)

    assert version_1.startswith("sha256:")
    assert version_1 == version_2

    # Test non-existent file
    missing = tmp_path / "missing.json"
    assert compute_dataset_version(missing) == "unknown"


def test_log_threshold_run_prd_compliance(temp_mlflow_tracker):
    """
    PRD Section 19 Verification:
    Each MLflow run should record:
    - Parameters: threshold, embedding_model, dataset_version
    - Metrics: precision, recall, f1, hit_rate, average_latency
    """
    tracker = temp_mlflow_tracker

    sample_metrics = ThresholdMetrics(
        threshold=0.80,
        tp=35,
        fp=5,
        tn=50,
        fn=10,
        total=100,
        precision=0.875,
        recall=0.7778,
        f1=0.8235,
        accuracy=0.85,
        category_breakdown={"category_a": {"tp": 20, "fp": 2, "tn": 25, "fn": 5}},
    )

    run_id = tracker.log_threshold_run(
        metrics=sample_metrics,
        embedding_model="all-MiniLM-L6-v2",
        dataset_version="sha256:abcd1234efgh",
        average_latency_ms=12.45,
        use_polarity_guard=True,
    )

    assert run_id is not None

    # Retrieve run from MLflow Client
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    run = client.get_run(run_id)

    # 1. Verify PRD Mandated Parameters
    params = run.data.params
    assert "threshold" in params and float(params["threshold"]) == 0.80
    assert params["embedding_model"] == "all-MiniLM-L6-v2"
    assert params["dataset_version"] == "sha256:abcd1234efgh"
    assert params["use_polarity_guard"] == "True"

    # 2. Verify PRD Mandated Metrics
    metrics = run.data.metrics
    assert "precision" in metrics and pytest.approx(metrics["precision"]) == 0.875
    assert "recall" in metrics and pytest.approx(metrics["recall"]) == 0.7778
    assert "f1" in metrics and pytest.approx(metrics["f1"]) == 0.8235
    # Hit rate = (TP + FP) / Total = (35 + 5) / 100 = 0.40
    assert "hit_rate" in metrics and pytest.approx(metrics["hit_rate"]) == 0.40
    assert "average_latency" in metrics and pytest.approx(metrics["average_latency"]) == 12.45
    assert "accuracy" in metrics and pytest.approx(metrics["accuracy"]) == 0.85
    assert metrics["tp"] == 35
    assert metrics["fp"] == 5
    assert metrics["tn"] == 50
    assert metrics["fn"] == 10

    # 3. Verify logged artifacts
    artifacts = client.list_artifacts(run_id, path="evaluation_details")
    assert len(artifacts) >= 1
    assert any(a.path.endswith("category_breakdown.json") for a in artifacts)


def test_log_evaluation_sweep(temp_mlflow_tracker):
    tracker = temp_mlflow_tracker

    sweep = [
        ThresholdMetrics(0.75, 40, 5, 45, 10, 100, 0.88, 0.80, 0.84, 0.85, {}),
        ThresholdMetrics(0.80, 35, 3, 47, 15, 100, 0.92, 0.70, 0.80, 0.82, {}),
        ThresholdMetrics(0.85, 25, 2, 48, 25, 100, 0.93, 0.50, 0.65, 0.73, {}),
    ]

    run_ids = tracker.log_evaluation_sweep(
        sweep_metrics=sweep,
        embedding_model="all-MiniLM-L6-v2",
        dataset_version="sha256:testversion",
        average_latency_ms=10.0,
        use_polarity_guard=True,
    )

    assert len(run_ids) == 3

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    experiment_runs = client.search_runs(experiment_ids=[tracker.experiment_id])
    assert len(experiment_runs) == 3
