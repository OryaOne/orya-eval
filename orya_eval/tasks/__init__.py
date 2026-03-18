"""Task runners."""

from collections.abc import Callable

import pandas as pd

from orya_eval.config import EvalConfig

from .classification import evaluate_classification
from .regression import evaluate_regression
from .text import evaluate_text

TaskEvaluator = Callable[[pd.DataFrame, EvalConfig], dict[str, float]]

TASK_EVALUATORS: dict[str, TaskEvaluator] = {
    "classification": evaluate_classification,
    "regression": evaluate_regression,
    "text": evaluate_text,
}

__all__ = [
    "TASK_EVALUATORS",
    "TaskEvaluator",
    "evaluate_classification",
    "evaluate_regression",
    "evaluate_text",
]
