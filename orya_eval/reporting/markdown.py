"""Markdown report rendering."""

from __future__ import annotations

from orya_eval.models import ComparisonResult, EvaluationResult


def render_run_report(result: EvaluationResult) -> str:
    """Render an evaluation result to Markdown."""
    lines = [
        f"# {result.run_name or 'orya-eval report'}",
        "",
        f"- Task: `{result.task_type}`",
        f"- Data: `{result.data_path}`",
        f"- Rows: `{result.row_count}`",
        f"- Status: `{'passed' if result.passed else 'failed'}`",
        f"- Generated: `{result.generated_at.isoformat()}`",
    ]
    if result.description:
        lines.extend(["", result.description])

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for metric, value in sorted(result.metrics.items()):
        lines.append(f"| `{metric}` | {value:.6f} |")

    if result.thresholds:
        lines.extend(
            [
                "",
                "## Thresholds",
                "",
                "| Metric | Operator | Target | Actual | Passed |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for check in result.thresholds:
            status = "yes" if check.passed else "no"
            lines.append(
                f"| `{check.metric}` | `{check.operator}` | "
                f"{check.target:.6f} | {check.actual:.6f} | {status} |"
            )

    return "\n".join(lines) + "\n"


def render_comparison_report(result: ComparisonResult) -> str:
    """Render a comparison result to Markdown."""
    lines = [
        "# orya-eval comparison",
        "",
        f"- Task: `{result.task_type}`",
        f"- Baseline: `{result.baseline_path}`",
        f"- Candidate: `{result.candidate_path}`",
        f"- Status: `{'passed' if result.passed else 'failed'}`",
        f"- Compared: `{result.compared_at.isoformat()}`",
        "",
        "## Metric Deltas",
        "",
        "| Metric | Baseline | Candidate | Delta | Regression |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for delta in result.metric_deltas:
        regression = "yes" if delta.is_regression else "no"
        lines.append(
            f"| `{delta.metric}` | {delta.baseline:.6f} | "
            f"{delta.candidate:.6f} | {delta.delta:+.6f} | {regression} |"
        )

    if result.thresholds:
        lines.extend(
            [
                "",
                "## Comparison Thresholds",
                "",
                "| Metric | Operator | Target Delta | Actual Delta | Passed |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for check in result.thresholds:
            status = "yes" if check.passed else "no"
            lines.append(
                f"| `{check.metric}` | `{check.operator}` | "
                f"{check.target_delta:+.6f} | {check.delta:+.6f} | {status} |"
            )

    return "\n".join(lines) + "\n"
