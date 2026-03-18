"""Evaluation orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from orya_eval.config import EvalConfig
from orya_eval.exceptions import ConfigError
from orya_eval.io.data import load_dataframe
from orya_eval.io.files import write_json, write_text
from orya_eval.metrics.registry import get_metric_spec
from orya_eval.models import EvaluationResult, ThresholdCheck
from orya_eval.reporting import render_run_report
from orya_eval.tasks import TASK_EVALUATORS


def run_evaluation(config: EvalConfig, config_path: str | Path) -> EvaluationResult:
    """Run one evaluation from a validated config model."""
    frame = load_dataframe(config.data_path)
    evaluator = TASK_EVALUATORS[config.task_type]
    metrics = evaluator(frame, config)

    threshold_checks = evaluate_thresholds(metrics, config.thresholds)
    result = EvaluationResult(
        task_type=config.task_type,
        run_name=config.run_name,
        description=config.description,
        data_path=config.data_path,
        config_path=Path(config_path).resolve(),
        row_count=len(frame),
        selected_metrics=list(metrics.keys()),
        metrics=metrics,
        thresholds=threshold_checks,
        passed=all(check.passed for check in threshold_checks) if threshold_checks else True,
        metadata=config.metadata,
    )

    write_json(config.reports.json_output, result.model_dump(mode="json"))
    if config.reports.markdown_output:
        write_text(config.reports.markdown_output, render_run_report(result))
    return result


def evaluate_thresholds(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> list[ThresholdCheck]:
    """Evaluate metric thresholds using metric direction metadata."""
    checks: list[ThresholdCheck] = []
    for metric, target in thresholds.items():
        if metric not in metrics:
            available = ", ".join(sorted(metrics))
            raise ConfigError(
                "Threshold configured for "
                f"'{metric}', but that metric was not computed. "
                f"Available metrics: {available}. Add it under `metrics` or remove it from "
                "`thresholds`."
            )
        spec = get_metric_spec(metric)
        actual = metrics[metric]
        operator = ">=" if spec.higher_is_better else "<="
        passed = actual >= target if spec.higher_is_better else actual <= target
        checks.append(
            ThresholdCheck(
                metric=metric,
                actual=actual,
                target=target,
                operator=operator,
                passed=passed,
            )
        )
    return checks
