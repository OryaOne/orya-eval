"""String-output evaluation logic."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from orya_eval.config import TextConfig
from orya_eval.exceptions import DataError
from orya_eval.metrics.text import (
    contains_match,
    normalize_text,
    normalized_similarity,
    token_f1,
)
from orya_eval.tasks._common import validate_required_columns

DEFAULT_METRICS = ["exact_match", "contains_match", "token_f1", "normalized_similarity"]


def evaluate_text(frame: pd.DataFrame, config: TextConfig) -> dict[str, float]:
    """Compute simple string-output evaluation metrics."""
    columns = config.columns
    required = [columns.reference, columns.prediction]
    validate_required_columns(frame, required)

    references = frame[columns.reference].tolist()
    predictions = frame[columns.prediction].tolist()
    selected_metrics = list(config.metrics or DEFAULT_METRICS)

    results: dict[str, float] = {}
    for metric in selected_metrics:
        if metric == "exact_match":
            results[metric] = _average(
                float(normalize_text(reference) == normalize_text(prediction))
                for reference, prediction in zip(references, predictions, strict=True)
            )
        elif metric == "contains_match":
            results[metric] = _average(
                contains_match(reference, prediction)
                for reference, prediction in zip(references, predictions, strict=True)
            )
        elif metric == "token_f1":
            results[metric] = _average(
                token_f1(reference, prediction)
                for reference, prediction in zip(references, predictions, strict=True)
            )
        elif metric == "normalized_similarity":
            results[metric] = _average(
                normalized_similarity(reference, prediction)
                for reference, prediction in zip(references, predictions, strict=True)
            )
        else:
            raise DataError(f"Unsupported text metric '{metric}'.")

    return results


def _average(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        raise DataError("Cannot compute text metrics on an empty dataset.")
    return float(sum(collected) / len(collected))
