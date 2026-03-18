"""Metric metadata shared across tasks and comparison logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    higher_is_better: bool
    description: str


METRIC_SPECS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("accuracy", True, "Fraction of predictions that match the target."),
    "precision": MetricSpec("precision", True, "Weighted precision across predicted classes."),
    "recall": MetricSpec("recall", True, "Weighted recall across predicted classes."),
    "f1": MetricSpec("f1", True, "Weighted F1 score across predicted classes."),
    "roc_auc": MetricSpec("roc_auc", True, "Area under the ROC curve for binary classification."),
    "mae": MetricSpec("mae", False, "Mean absolute error."),
    "rmse": MetricSpec("rmse", False, "Root mean squared error."),
    "r2": MetricSpec("r2", True, "Coefficient of determination."),
    "exact_match": MetricSpec("exact_match", True, "Normalized exact match rate."),
    "contains_match": MetricSpec(
        "contains_match",
        True,
        "Rate where one normalized string contains the other.",
    ),
    "token_f1": MetricSpec("token_f1", True, "Token-level F1 after text normalization."),
    "normalized_similarity": MetricSpec(
        "normalized_similarity",
        True,
        "Average normalized string similarity ratio.",
    ),
}

TASK_METRICS: dict[str, set[str]] = {
    "classification": {"accuracy", "precision", "recall", "f1", "roc_auc"},
    "regression": {"mae", "rmse", "r2"},
    "text": {"exact_match", "contains_match", "token_f1", "normalized_similarity"},
}


def get_metric_spec(metric: str) -> MetricSpec:
    """Return metric metadata or raise a clear error for unknown metrics."""
    try:
        return METRIC_SPECS[metric]
    except KeyError as exc:
        supported = ", ".join(sorted(METRIC_SPECS))
        raise ValueError(f"Unknown metric '{metric}'. Supported metrics: {supported}.") from exc


def get_task_metrics(task_type: str) -> set[str]:
    """Return the supported metrics for a task type."""
    try:
        return TASK_METRICS[task_type]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_METRICS))
        raise ValueError(
            f"Unknown task type '{task_type}'. Supported task types: {supported}."
        ) from exc
