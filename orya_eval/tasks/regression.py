"""Regression evaluation logic."""

from __future__ import annotations

from math import sqrt

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from orya_eval.config import RegressionConfig
from orya_eval.exceptions import DataError
from orya_eval.tasks._common import validate_required_columns

DEFAULT_METRICS = ["mae", "rmse", "r2"]


def evaluate_regression(frame: pd.DataFrame, config: RegressionConfig) -> dict[str, float]:
    """Compute regression metrics from configured columns."""
    columns = config.columns
    required = [columns.target, columns.prediction]
    validate_required_columns(frame, required)

    y_true = frame[columns.target]
    y_pred = frame[columns.prediction]
    selected_metrics = list(config.metrics or DEFAULT_METRICS)

    results: dict[str, float] = {}
    for metric in selected_metrics:
        if metric == "mae":
            results[metric] = float(mean_absolute_error(y_true, y_pred))
        elif metric == "rmse":
            results[metric] = float(sqrt(mean_squared_error(y_true, y_pred)))
        elif metric == "r2":
            results[metric] = float(r2_score(y_true, y_pred))
        else:
            raise DataError(f"Unsupported regression metric '{metric}'.")
    return results
