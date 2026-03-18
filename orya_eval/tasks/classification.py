"""Classification evaluation logic."""

from __future__ import annotations

from typing import cast

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from orya_eval.config import ClassificationConfig
from orya_eval.exceptions import DataError
from orya_eval.tasks._common import validate_required_columns

DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1"]


def evaluate_classification(frame: pd.DataFrame, config: ClassificationConfig) -> dict[str, float]:
    """Compute classification metrics from configured columns."""
    columns = config.columns
    required = [columns.target, columns.prediction]
    if columns.probability:
        required.append(columns.probability)
    validate_required_columns(frame, required)

    y_true = frame[columns.target]
    y_pred = frame[columns.prediction]

    selected_metrics = list(config.metrics or DEFAULT_METRICS)
    if config.metrics is None and columns.probability and y_true.nunique(dropna=False) == 2:
        selected_metrics.append("roc_auc")

    results: dict[str, float] = {}
    for metric in selected_metrics:
        if metric == "accuracy":
            results[metric] = float(accuracy_score(y_true, y_pred))
        elif metric == "precision":
            results[metric] = float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            )
        elif metric == "recall":
            results[metric] = float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            )
        elif metric == "f1":
            results[metric] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        elif metric == "roc_auc":
            if not columns.probability:
                raise DataError(
                    "Metric 'roc_auc' requires a probability column. "
                    "Add 'columns.probability' to the config or remove "
                    "'roc_auc' from metrics."
                )
            if y_true.nunique(dropna=False) != 2:
                raise DataError(
                    "Metric 'roc_auc' currently supports only binary classification data."
                )
            y_score = cast(pd.Series, frame[columns.probability])
            results[metric] = float(roc_auc_score(y_true, y_score))
        else:
            raise DataError(f"Unsupported classification metric '{metric}'.")

    return results
