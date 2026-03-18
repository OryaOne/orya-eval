"""Result comparison and regression checks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from orya_eval.exceptions import ComparisonError
from orya_eval.io.files import write_text
from orya_eval.metrics.registry import get_metric_spec
from orya_eval.models import (
    ComparisonResult,
    ComparisonThresholdCheck,
    EvaluationResult,
    MetricDelta,
)
from orya_eval.reporting import render_comparison_report


def load_result(result_path: str | Path) -> EvaluationResult:
    """Load a result JSON file into a typed model."""
    path = Path(result_path).resolve()
    if not path.exists():
        raise ComparisonError(
            f"Result file not found: {path}. Provide a JSON results file created by "
            "`orya-eval run`."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ComparisonError(
                f"Invalid result file '{path}'. Expected a JSON object created by `orya-eval run`."
            )
        return EvaluationResult.model_validate(payload)
    except json.JSONDecodeError as exc:
        raise ComparisonError(
            f"Malformed JSON in result file '{path}' at line {exc.lineno}, column "
            f"{exc.colno}. Fix the JSON or regenerate the file with `orya-eval run`."
        ) from exc
    except ValidationError as exc:
        raise ComparisonError(_format_result_validation_error(path, exc)) from exc


def compare_results(
    baseline_path: str | Path,
    candidate_path: str | Path,
    thresholds: dict[str, float] | None = None,
    markdown_output: str | Path | None = None,
    fail_on_regression: bool = False,
) -> ComparisonResult:
    """Compare two evaluation results and optionally enforce delta thresholds."""
    baseline = load_result(baseline_path)
    candidate = load_result(candidate_path)
    if baseline.task_type != candidate.task_type:
        raise ComparisonError(
            "Cannot compare results with different task types: "
            f"'{baseline.task_type}' vs '{candidate.task_type}'. Compare runs from the same "
            "evaluation task."
        )

    common_metrics = sorted(set(baseline.metrics) & set(candidate.metrics))
    if not common_metrics:
        raise ComparisonError(
            "The two result files do not share any metrics. Compare runs that were evaluated "
            "with at least one common metric."
        )

    deltas: list[MetricDelta] = []
    for metric in common_metrics:
        spec = get_metric_spec(metric)
        baseline_value = baseline.metrics[metric]
        candidate_value = candidate.metrics[metric]
        delta = candidate_value - baseline_value
        is_regression = delta < 0 if spec.higher_is_better else delta > 0
        deltas.append(
            MetricDelta(
                metric=metric,
                baseline=baseline_value,
                candidate=candidate_value,
                delta=delta,
                higher_is_better=spec.higher_is_better,
                is_regression=is_regression,
            )
        )

    threshold_checks = evaluate_comparison_thresholds(deltas, thresholds or {})
    passed = True
    if threshold_checks:
        passed = all(check.passed for check in threshold_checks)
    elif fail_on_regression:
        passed = not any(delta.is_regression for delta in deltas)

    result = ComparisonResult(
        baseline_path=Path(baseline_path).resolve(),
        candidate_path=Path(candidate_path).resolve(),
        task_type=baseline.task_type,
        metric_deltas=deltas,
        thresholds=threshold_checks,
        passed=passed,
    )
    if markdown_output:
        write_text(markdown_output, render_comparison_report(result))
    return result


def evaluate_comparison_thresholds(
    deltas: list[MetricDelta],
    thresholds: Mapping[str, float],
) -> list[ComparisonThresholdCheck]:
    """Check metric deltas against allowed degradation thresholds."""
    checks: list[ComparisonThresholdCheck] = []
    by_metric = {delta.metric: delta for delta in deltas}
    for metric, target_delta in thresholds.items():
        if metric not in by_metric:
            raise ComparisonError(
                "Comparison threshold requested for metric "
                f"'{metric}', but it is not present in both result files. "
                "Use only metrics that appear in both files."
            )
        delta = by_metric[metric]
        operator = ">=" if delta.higher_is_better else "<="
        if delta.higher_is_better:
            passed = delta.delta >= target_delta
        else:
            passed = delta.delta <= target_delta
        checks.append(
            ComparisonThresholdCheck(
                metric=metric,
                delta=delta.delta,
                target_delta=target_delta,
                operator=operator,
                passed=passed,
            )
        )
    return checks


def _format_result_validation_error(path: Path, exc: ValidationError) -> str:
    details: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error["loc"])
        details.append(f"- {location}: {error['msg']}")
    joined = "\n".join(details)
    return (
        f"Invalid result file '{path}'. Expected the JSON structure produced by "
        "`orya-eval run`.\n"
        f"{joined}"
    )
