"""Terminal summary rendering for CLI commands."""

from __future__ import annotations

from pathlib import Path

from orya_eval.models import ComparisonResult, EvaluationResult


def render_init_summary(template: str, config_path: Path, data_path: Path) -> str:
    return "\n".join(
        [
            _status_line("ready"),
            "",
            "Summary:",
            _summary_row("Command", "init"),
            _summary_row("Template", template),
            _summary_row("Config", str(config_path)),
            _summary_row("Sample data", str(data_path)),
            "",
            "Next step:",
            "  Run your starter evaluation with:",
            f"  orya-eval run {config_path}",
        ]
    )


def render_run_summary(
    result: EvaluationResult,
    *,
    json_results_path: Path,
    markdown_report_path: Path | None,
) -> str:
    lines = [
        _status_line("pass" if result.passed else "fail"),
        "",
        "Summary:",
        _summary_row("Command", "run"),
        _summary_row("Task", result.task_type),
        _summary_row("Run name", result.run_name or "-"),
        _summary_row("Rows", str(result.row_count)),
        _summary_row("Data", str(result.data_path)),
        "",
        "Metrics:",
    ]
    lines.extend(_metric_row(metric, value) for metric, value in sorted(result.metrics.items()))

    if result.thresholds:
        lines.append("")
        lines.append("Threshold checks:")
        lines.extend(
            _threshold_row(
                "pass" if check.passed else "fail",
                check.metric,
                f"{check.actual:.6f} {check.operator} {check.target:.6f}",
            )
            for check in result.thresholds
        )

    lines.append("")
    lines.append("Artifacts:")
    lines.append(_summary_row("JSON results", str(json_results_path)))
    if markdown_report_path is not None:
        lines.append(_summary_row("Markdown report", str(markdown_report_path)))

    lines.append("")
    lines.append("Next step:")
    lines.extend(_next_step_lines_for_run(result, json_results_path))
    return "\n".join(lines)


def render_comparison_summary(
    result: ComparisonResult,
    *,
    markdown_report: Path | None,
) -> str:
    regression_count = sum(1 for delta in result.metric_deltas if delta.is_regression)
    lines = [
        _status_line("pass" if result.passed else "fail"),
        "",
        "Summary:",
        _summary_row("Command", "compare"),
        _summary_row("Task", result.task_type),
        _summary_row("Baseline", str(result.baseline_path)),
        _summary_row("Candidate", str(result.candidate_path)),
        _summary_row("Shared metrics", str(len(result.metric_deltas))),
        _summary_row("Regressions", str(regression_count)),
        "",
        "Metric deltas:",
    ]
    lines.extend(
        _comparison_delta_row(
            delta.metric,
            delta.baseline,
            delta.candidate,
            delta.delta,
            "regression" if delta.is_regression else "ok",
        )
        for delta in result.metric_deltas
    )

    if result.thresholds:
        lines.append("")
        lines.append("Delta thresholds:")
        lines.extend(
            _threshold_row(
                "pass" if check.passed else "fail",
                check.metric,
                f"{check.delta:+.6f} {check.operator} {check.target_delta:+.6f}",
            )
            for check in result.thresholds
        )

    if markdown_report is not None:
        lines.append("")
        lines.append("Artifacts:")
        lines.append(_summary_row("Markdown report", str(markdown_report)))

    lines.append("")
    lines.append("Next step:")
    if result.passed:
        lines.append("  Use these delta checks in CI to block regressions on shared metrics.")
    else:
        lines.append(
            "  Inspect the regressing metrics above or relax the delta thresholds "
            "if the change is expected."
        )
    return "\n".join(lines)


def _status_line(status: str) -> str:
    return f"Status: {status.upper()}"


def _summary_row(label: str, value: str) -> str:
    return f"  {label:<15} {value}"


def _metric_row(metric: str, value: float) -> str:
    return f"  {metric:<22}{value:>10.6f}"


def _comparison_delta_row(
    metric: str,
    baseline: float,
    candidate: float,
    delta: float,
    verdict: str,
) -> str:
    return (
        f"  {metric:<22}"
        f"{baseline:>10.6f} -> {candidate:>10.6f}  "
        f"delta {delta:+.6f}  {verdict}"
    )


def _threshold_row(verdict: str, metric: str, expression: str) -> str:
    return f"  {verdict:<5}{metric:<17}{expression}"


def _next_step_lines_for_run(result: EvaluationResult, json_results_path: Path) -> list[str]:
    if result.thresholds:
        if result.passed:
            return [
                "  Compare this results file to a baseline with:",
                f"  orya-eval compare BASELINE_RESULTS {json_results_path}",
            ]
        return [
            "  Review the threshold checks above, then update the config thresholds",
            "  or model output before running again.",
        ]
    return [
        "  Add `thresholds` to the config if you want this command to act as a",
        "  CI quality gate.",
    ]
