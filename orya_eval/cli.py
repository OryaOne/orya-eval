"""Typer CLI for orya-eval."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from orya_eval.exceptions import ComparisonError, ConfigError, DataError, OryaEvalError
from orya_eval.metrics.registry import METRIC_SPECS
from orya_eval.reporting import (
    render_comparison_summary,
    render_init_summary,
    render_run_summary,
)

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Run repeatable evaluations and regression checks for ML and AI systems.\n\n"
        "Use `orya-eval init` to create a starter config, `orya-eval run` to evaluate "
        "a dataset, and `orya-eval compare` to compare two result files."
    ),
)


@app.command("init")
def init_command(
    template: Annotated[
        str,
        typer.Option(
            ...,
            "--template",
            metavar="TASK",
            help="Starter template to create: classification, regression, or text.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory where the starter config and sample data should be written.",
        ),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing starter files in the output directory."),
    ] = False,
) -> None:
    """Create a starter evaluation config and matching sample data."""
    try:
        from orya_eval.templates import create_starter_template

        files = create_starter_template(template=template, output_dir=output_dir, force=force)
    except OryaEvalError as exc:
        _exit_with_error(str(exc))
    typer.echo(render_init_summary(template, files.config_path, files.data_path))


@app.command("run")
def run_command(
    config_path: Annotated[
        Path,
        typer.Argument(help="Path to an evaluation YAML config file."),
    ],
) -> None:
    """Run an evaluation from a YAML config file."""
    try:
        from orya_eval.config import load_config
        from orya_eval.runner import run_evaluation

        config = load_config(config_path)
        result = run_evaluation(config, config_path=config_path)
    except (ConfigError, DataError) as exc:
        _exit_with_error(str(exc))

    typer.echo(
        render_run_summary(
            result,
            json_results_path=config.reports.json_output,
            markdown_report_path=config.reports.markdown_output,
        )
    )

    if not result.passed:
        _exit_with_error("Run failed because one or more threshold checks did not pass.", code=1)


@app.command("compare")
def compare_command(
    baseline_result: Annotated[
        Path,
        typer.Argument(help="Path to the baseline JSON results file."),
    ],
    candidate_result: Annotated[
        Path,
        typer.Argument(help="Path to the candidate JSON results file."),
    ],
    delta_threshold: Annotated[
        list[str] | None,
        typer.Option(
            "--delta-threshold",
            "--threshold",
            metavar="METRIC=VALUE",
            help=(
                "Allowed metric delta as METRIC=VALUE. For higher-is-better metrics this "
                "is a minimum delta; for lower-is-better metrics it is a maximum delta."
            ),
        ),
    ] = None,
    markdown_report: Annotated[
        Path | None,
        typer.Option(
            "--markdown-report",
            "--markdown-output",
            help="Optional path for a Markdown comparison report.",
        ),
    ] = None,
    fail_on_regression: Annotated[
        bool,
        typer.Option(
            "--fail-on-regression",
            help="Exit non-zero if any shared metric regresses, even without delta thresholds.",
        ),
    ] = False,
) -> None:
    """Compare baseline and candidate result JSON files."""
    try:
        from orya_eval.comparison import compare_results

        threshold_map = _parse_delta_threshold_options(delta_threshold or [])
        comparison = compare_results(
            baseline_path=baseline_result,
            candidate_path=candidate_result,
            thresholds=threshold_map,
            markdown_output=markdown_report,
            fail_on_regression=fail_on_regression,
        )
    except (ComparisonError, ConfigError) as exc:
        _exit_with_error(str(exc))

    typer.echo(render_comparison_summary(comparison, markdown_report=markdown_report))

    if not comparison.passed:
        _exit_with_error(
            "Comparison failed because a regression or delta threshold check did not pass.",
            code=1,
        )


def _parse_delta_threshold_options(items: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(
                "Invalid delta threshold "
                f"'{item}'. Use METRIC=VALUE, for example accuracy=-0.01 or mae=0.05."
            )
        metric, raw_value = item.split("=", 1)
        metric = metric.strip()
        if metric not in METRIC_SPECS:
            supported = ", ".join(sorted(METRIC_SPECS))
            raise ConfigError(
                f"Unknown metric '{metric}' in delta threshold '{item}'. "
                f"Supported metrics: {supported}."
            )
        try:
            thresholds[metric] = float(raw_value)
        except ValueError as exc:
            raise ConfigError(
                f"Invalid numeric value in delta threshold '{item}'. "
                "Use a number such as -0.01 or 0.05."
            ) from exc
    return thresholds


def _exit_with_error(message: str, code: int = 2) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(code=code)


if __name__ == "__main__":
    app()
