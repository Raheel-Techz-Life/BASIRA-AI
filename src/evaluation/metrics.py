"""Evaluation metrics for the educational multiclass tumor-type classifier.

Master prompt Section 8 forbids reporting accuracy alone, so this always
returns macro-F1, per-class precision/recall, and the confusion matrix
alongside accuracy.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def format_metrics_summary(metrics: dict) -> str:
    lines = [
        f"Accuracy: {metrics['accuracy']:.3f}",
        f"Macro F1: {metrics['macro_f1']:.3f}",
        "Per-class (precision / recall / f1):",
    ]
    for label in metrics["labels"]:
        row = metrics["per_class_report"].get(label, {})
        lines.append(
            f"  {label}: P={row.get('precision', 0):.3f} "
            f"R={row.get('recall', 0):.3f} F1={row.get('f1-score', 0):.3f}"
        )
    return "\n".join(lines)
