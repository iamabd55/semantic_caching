from dataclasses import asdict
import json
from pathlib import Path
import sys

# Ensure root package is in python path
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.evaluator import ThresholdEvaluator, ThresholdMetrics
from evaluation.mlflow_tracker import MLflowTracker, compute_dataset_version


def format_markdown_table(metrics: list[ThresholdMetrics]) -> str:
    lines = [
        "| Threshold | TP | FP | TN | FN | Precision | Recall | F1 Score | Accuracy |",
        "|:---------:|:--:|:--:|:--:|:--:|:---------:|:------:|:--------:|:--------:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m.threshold:.2f} | {m.tp} | {m.fp} | {m.tn} | {m.fn} | "
            f"{m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} | {m.accuracy:.4f} |"
        )
    return "\n".join(lines)


def run_evaluation() -> None:
    evaluator = ThresholdEvaluator()
    print("Loading evaluation dataset...")
    evaluator.load_dataset()
    print(f"Loaded {len(evaluator.dataset)} labeled pairs.")

    print("Pre-computing query embeddings and cosine similarities...")
    pairs = evaluator.compute_pair_similarities()
    print(f"Processed {len(pairs)} query pairs.")

    thresholds = [0.70, 0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.95]

    # 1. Guarded Threshold Sweep (Production Mode with Polarity Guard)
    guarded_metrics = evaluator.evaluate_threshold_sweep(
        thresholds=thresholds,
        use_polarity_guard=True,
    )
    optimal_guarded = evaluator.find_optimal_threshold(guarded_metrics, min_precision=0.85)

    # 2. Raw Threshold Sweep (Ablation without Polarity Guard)
    raw_metrics = evaluator.evaluate_threshold_sweep(
        thresholds=thresholds,
        use_polarity_guard=False,
    )
    optimal_raw = evaluator.find_optimal_threshold(raw_metrics, min_precision=0.85)

    # Output Console Tables
    print("\n" + "=" * 80)
    print("PHASE 9 — THRESHOLD EVALUATION RESULTS (WITH POLARITY GUARD)")
    print("=" * 80)
    print(format_markdown_table(guarded_metrics))

    print("\n" + "=" * 80)
    print(f"RECOMMENDED OPTIMAL THRESHOLD: {optimal_guarded.threshold:.2f}")
    print(
        f"Metrics -> Precision: {optimal_guarded.precision:.4f} | "
        f"Recall: {optimal_guarded.recall:.4f} | "
        f"F1 Score: {optimal_guarded.f1:.4f} | "
        f"Accuracy: {optimal_guarded.accuracy:.4f}"
    )
    print(
        f"Confusion Matrix -> TP: {optimal_guarded.tp}, FP: {optimal_guarded.fp}, "
        f"TN: {optimal_guarded.tn}, FN: {optimal_guarded.fn}"
    )
    print("=" * 80 + "\n")

    # Save JSON Output
    output_dir = Path(__file__).parent
    json_path = output_dir / "threshold_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "optimal_threshold": optimal_guarded.threshold,
                "optimal_metrics": asdict(optimal_guarded),
                "guarded_sweep": [asdict(m) for m in guarded_metrics],
                "raw_sweep": [asdict(m) for m in raw_metrics],
            },
            f,
            indent=2,
        )
    print(f"Metrics saved to {json_path}")

    # Log to MLflow (Phase 10: Reproducible Threshold Experiments)
    print("Logging evaluation runs to MLflow...")
    dataset_version = compute_dataset_version(evaluator.dataset_path)
    embedding_model = evaluator.embedding_service.model_name
    avg_latency = evaluator.average_embedding_latency_ms

    tracker = MLflowTracker()
    # Log production guarded sweep
    tracker.log_evaluation_sweep(
        sweep_metrics=guarded_metrics,
        embedding_model=embedding_model,
        dataset_version=dataset_version,
        average_latency_ms=avg_latency,
        use_polarity_guard=True,
    )
    # Log raw ablation sweep
    tracker.log_evaluation_sweep(
        sweep_metrics=raw_metrics,
        embedding_model=embedding_model,
        dataset_version=dataset_version,
        average_latency_ms=avg_latency,
        use_polarity_guard=False,
    )
    print(f"Successfully logged {len(guarded_metrics) + len(raw_metrics)} runs to MLflow experiment '{tracker.experiment_name}' (Tracking URI: {tracker.tracking_uri})")

    # Generate Markdown Report
    report_path = output_dir / "threshold_evaluation_report.md"
    report_content = f"""# Phase 9: Threshold Evaluation Report

## 1. Executive Summary
- **Evaluation Dataset**: `{len(pairs)}` curated query pairs (`evaluation/dataset/cache_eval.json`)
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Empirical Optimal Threshold**: **`{optimal_guarded.threshold:.2f}`**
- **Optimal Metrics**:
  - **Precision**: `{optimal_guarded.precision:.4f}`
  - **Recall**: `{optimal_guarded.recall:.4f}`
  - **F1 Score**: `{optimal_guarded.f1:.4f}`
  - **Accuracy**: `{optimal_guarded.accuracy:.4f}`
  - **Confusion Matrix**: TP = {optimal_guarded.tp}, FP = {optimal_guarded.fp}, TN = {optimal_guarded.tn}, FN = {optimal_guarded.fn}

---

## 2. Threshold Sweep Results (Production: With Polarity Guard)

{format_markdown_table(guarded_metrics)}

---

## 3. Ablation Study: Impact of Polarity Guard

Comparison of False Positives (FP) and Precision with vs without Polarity Guard:

| Threshold | Raw FP | Guarded FP | FP Reduction | Raw Precision | Guarded Precision | Precision Delta |
|:---------:|:------:|:----------:|:------------:|:-------------:|:-----------------:|:---------------:|
"""
    for raw_m, grd_m in zip(raw_metrics, guarded_metrics):
        fp_red = raw_m.fp - grd_m.fp
        p_delta = grd_m.precision - raw_m.precision
        report_content += (
            f"| {grd_m.threshold:.2f} | {raw_m.fp} | {grd_m.fp} | -{fp_red} | "
            f"{raw_m.precision:.4f} | {grd_m.precision:.4f} | +{p_delta:.4f} |\n"
        )

    report_content += f"""
---

## 4. Key Takeaways and Recommendation

1. **Trade-off Analysis**:
   - At lower thresholds (e.g., `<= 0.75`), recall is very high, but precision degrades as unrelated questions trigger false cache hits.
   - At higher thresholds (e.g., `>= 0.92`), precision reaches near 100%, but recall drops, missing valid paraphrases and reducing cache efficiency.
   - The threshold of **`{optimal_guarded.threshold:.2f}`** delivers the highest balanced **F1 score ({optimal_guarded.f1:.4f})** while keeping precision high (`{optimal_guarded.precision:.4f}`).

2. **Polarity Guard Value**:
   - The Polarity Guard successfully blocks false positive cache hits caused by embedding blind spots on subtle negation and antonym conflicts (e.g., "Is Python statically typed?" vs "Is Python not statically typed?").

3. **Recommendation**:
   - Set `SIMILARITY_THRESHOLD = {optimal_guarded.threshold:.2f}` in application settings.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Comprehensive report saved to {report_path}")


if __name__ == "__main__":
    run_evaluation()
