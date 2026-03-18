"""Typed models for evaluation and comparison results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from orya_eval.metrics.registry import get_metric_spec, get_task_metrics


class ThresholdCheck(BaseModel):
    metric: str
    actual: float
    target: float
    operator: Literal[">=", "<="]
    passed: bool


class EvaluationResult(BaseModel):
    task_type: Literal["classification", "regression", "text"]
    run_name: str | None = None
    description: str | None = None
    data_path: Path
    config_path: Path
    row_count: int
    selected_metrics: list[str]
    metrics: dict[str, float]
    thresholds: list[ThresholdCheck] = Field(default_factory=list)
    passed: bool
    metadata: dict[str, str] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_metrics_for_task(self) -> EvaluationResult:
        allowed = get_task_metrics(self.task_type)
        metric_names = list(self.metrics)
        invalid = [metric for metric in metric_names if metric not in allowed]
        if invalid:
            invalid_text = ", ".join(invalid)
            allowed_text = ", ".join(sorted(allowed))
            raise ValueError(
                f"Unsupported metrics for task '{self.task_type}': "
                f"{invalid_text}. Allowed: {allowed_text}."
            )

        if self.selected_metrics != metric_names:
            raise ValueError(
                "`selected_metrics` must list the same metrics, in the same order, as `metrics`."
            )

        threshold_metrics = [check.metric for check in self.thresholds]
        unknown_thresholds = [metric for metric in threshold_metrics if metric not in self.metrics]
        if unknown_thresholds:
            unknown_text = ", ".join(unknown_thresholds)
            raise ValueError(
                f"Threshold checks reference metrics that are not present in `metrics`: "
                f"{unknown_text}."
            )
        return self


class ComparisonThresholdCheck(BaseModel):
    metric: str
    delta: float
    target_delta: float
    operator: Literal[">=", "<="]
    passed: bool


class MetricDelta(BaseModel):
    metric: str
    baseline: float
    candidate: float
    delta: float
    higher_is_better: bool
    is_regression: bool

    @model_validator(mode="after")
    def validate_direction_metadata(self) -> MetricDelta:
        spec = get_metric_spec(self.metric)
        if self.higher_is_better != spec.higher_is_better:
            raise ValueError(
                f"Metric delta for '{self.metric}' has inconsistent `higher_is_better` metadata."
            )
        return self


class ComparisonResult(BaseModel):
    baseline_path: Path
    candidate_path: Path
    task_type: Literal["classification", "regression", "text"]
    metric_deltas: list[MetricDelta]
    thresholds: list[ComparisonThresholdCheck] = Field(default_factory=list)
    passed: bool
    compared_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_threshold_metrics(self) -> ComparisonResult:
        delta_metrics = {delta.metric for delta in self.metric_deltas}
        unknown_thresholds = [
            check.metric for check in self.thresholds if check.metric not in delta_metrics
        ]
        if unknown_thresholds:
            unknown_text = ", ".join(unknown_thresholds)
            raise ValueError(
                "Comparison threshold checks reference metrics that are not present in "
                f"`metric_deltas`: {unknown_text}."
            )
        return self
