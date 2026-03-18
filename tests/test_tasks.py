from __future__ import annotations

import pandas as pd
import pytest

from orya_eval.config import ClassificationConfig, RegressionConfig, TextConfig
from orya_eval.exceptions import DataError
from orya_eval.metrics.text import contains_match, normalize_text, normalized_similarity, token_f1
from orya_eval.tasks.classification import evaluate_classification
from orya_eval.tasks.regression import evaluate_regression
from orya_eval.tasks.text import evaluate_text


def test_classification_metrics_are_computed_from_configured_columns(
    classification_frame: pd.DataFrame,
) -> None:
    config = ClassificationConfig(
        task_type="classification",
        data_path="unused.csv",
        columns={
            "target": "target",
            "prediction": "prediction",
            "probability": "probability",
        },
        metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )

    results = evaluate_classification(classification_frame, config)

    assert results["accuracy"] == pytest.approx(5 / 6)
    assert results["precision"] == pytest.approx(0.875)
    assert results["recall"] == pytest.approx(5 / 6)
    assert results["f1"] == pytest.approx(0.8285714286)
    assert results["roc_auc"] == pytest.approx(1.0)


def test_classification_defaults_include_roc_auc_for_binary_probabilities(
    classification_frame: pd.DataFrame,
) -> None:
    config = ClassificationConfig(
        task_type="classification",
        data_path="unused.csv",
        columns={
            "target": "target",
            "prediction": "prediction",
            "probability": "probability",
        },
    )

    results = evaluate_classification(classification_frame, config)

    assert list(results) == ["accuracy", "precision", "recall", "f1", "roc_auc"]


def test_classification_rejects_roc_auc_without_probability_column(
    classification_frame: pd.DataFrame,
) -> None:
    config = ClassificationConfig(
        task_type="classification",
        data_path="unused.csv",
        columns={"target": "target", "prediction": "prediction"},
        metrics=["roc_auc"],
    )

    with pytest.raises(DataError, match="requires a probability column"):
        evaluate_classification(classification_frame[["target", "prediction"]], config)


def test_classification_rejects_multiclass_roc_auc() -> None:
    frame = pd.DataFrame(
        {
            "target": ["cat", "dog", "bird"],
            "prediction": ["cat", "dog", "bird"],
            "probability": [0.9, 0.7, 0.8],
        }
    )
    config = ClassificationConfig(
        task_type="classification",
        data_path="unused.csv",
        columns={
            "target": "target",
            "prediction": "prediction",
            "probability": "probability",
        },
        metrics=["roc_auc"],
    )

    with pytest.raises(DataError, match="only binary classification"):
        evaluate_classification(frame, config)


def test_regression_metrics_are_computed_from_configured_columns(
    regression_frame: pd.DataFrame,
) -> None:
    config = RegressionConfig(
        task_type="regression",
        data_path="unused.csv",
        columns={"target": "target", "prediction": "prediction"},
    )

    results = evaluate_regression(regression_frame, config)

    assert results["mae"] == pytest.approx(0.22)
    assert results["rmse"] == pytest.approx(0.2323790008)
    assert results["r2"] == pytest.approx(0.9717927288)


def test_text_metrics_are_computed_from_normalized_strings(text_frame: pd.DataFrame) -> None:
    config = TextConfig(
        task_type="text",
        data_path="unused.jsonl",
        columns={"reference": "reference", "prediction": "prediction"},
    )

    results = evaluate_text(text_frame, config)

    assert results["exact_match"] == pytest.approx(0.25)
    assert results["contains_match"] == pytest.approx(0.75)
    assert results["token_f1"] == pytest.approx(0.35)
    assert results["normalized_similarity"] == pytest.approx(0.6386956522)


def test_text_metric_helpers_handle_normalization_and_overlap() -> None:
    assert normalize_text("  The  Answer ") == "the answer"
    assert contains_match("Paris", "The answer is Paris") == 1.0
    assert token_f1("red apple", "fresh red apples") == pytest.approx(0.4)
    assert normalized_similarity("Athens", "athens") == pytest.approx(1.0)


def test_text_evaluation_rejects_missing_columns(text_frame: pd.DataFrame) -> None:
    config = TextConfig(
        task_type="text",
        data_path="unused.jsonl",
        columns={"reference": "reference", "prediction": "prediction"},
    )

    with pytest.raises(DataError, match="missing required columns"):
        evaluate_text(text_frame[["reference"]], config)
