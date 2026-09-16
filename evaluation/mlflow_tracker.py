import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import mlflow
from mlflow.tracking import MlflowClient

from evaluation.evaluator import ThresholdMetrics

# Silence agent hint and allow file store if needed
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def compute_dataset_version(dataset_path: Path) -> str:
    """Compute a deterministic hash/version of the evaluation dataset file."""
    if not dataset_path.exists():
        return "unknown"
    hasher = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()[:12]}"


class MLflowTracker:
    """
    Handles logging of threshold evaluation runs and metrics into MLflow (Phase 10).
    Ensures runs are reproducible, comparable, and trackable.
    """

    def __init__(
        self,
        experiment_name: str = "semantic-cache-threshold-evaluation",
        tracking_uri: Optional[str] = None,
    ) -> None:
        self.experiment_name = experiment_name
        default_db = Path(__file__).parent.parent / "mlflow.db"
        self.tracking_uri = tracking_uri or f"sqlite:///{default_db.as_posix()}"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        # Set or create experiment
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_id=self.experiment_id)

    def log_threshold_run(
        self,
        metrics: ThresholdMetrics,
        embedding_model: str,
        dataset_version: str,
        average_latency_ms: float,
        use_polarity_guard: bool = True,
        run_name: Optional[str] = None,
    ) -> str:
        """
        Log a single threshold evaluation run to MLflow per PRD Section 19.
        Returns the run ID.
        """
        effective_run_name = run_name or f"threshold_{metrics.threshold:.2f}_{'guarded' if use_polarity_guard else 'raw'}"

        hit_rate = (metrics.tp + metrics.fp) / metrics.total if metrics.total > 0 else 0.0

        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=effective_run_name,
        ) as run:
            # 1. Log Mandated Parameters (PRD Section 19)
            mlflow.log_params({
                "threshold": metrics.threshold,
                "embedding_model": embedding_model,
                "dataset_version": dataset_version,
                "use_polarity_guard": use_polarity_guard,
            })

            # 2. Log Mandated Metrics (PRD Section 19)
            mlflow.log_metrics({
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "hit_rate": round(hit_rate, 4),
                "average_latency": round(average_latency_ms, 2),
                "accuracy": metrics.accuracy,
                "tp": metrics.tp,
                "fp": metrics.fp,
                "tn": metrics.tn,
                "fn": metrics.fn,
                "total_queries": metrics.total,
            })

            # 3. Log Category Breakdown as Artifact
            breakdown_path = Path("category_breakdown.json")
            try:
                with open(breakdown_path, "w", encoding="utf-8") as f:
                    json.dump(metrics.category_breakdown, f, indent=2)
                mlflow.log_artifact(str(breakdown_path), artifact_path="evaluation_details")
            finally:
                if breakdown_path.exists():
                    breakdown_path.unlink()

            return run.info.run_id

    def log_evaluation_sweep(
        self,
        sweep_metrics: List[ThresholdMetrics],
        embedding_model: str,
        dataset_version: str,
        average_latency_ms: float,
        use_polarity_guard: bool = True,
    ) -> List[str]:
        """
        Logs an entire threshold sweep to MLflow, returning list of run IDs.
        """
        run_ids = []
        for m in sweep_metrics:
            run_id = self.log_threshold_run(
                metrics=m,
                embedding_model=embedding_model,
                dataset_version=dataset_version,
                average_latency_ms=average_latency_ms,
                use_polarity_guard=use_polarity_guard,
            )
            run_ids.append(run_id)
        return run_ids
