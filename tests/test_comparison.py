from __future__ import annotations

import json
from pathlib import Path

import pytest

from orya_eval.comparison import compare_results, load_result
from orya_eval.exceptions import ComparisonError
from orya_eval.models import EvaluationResult
from orya_eval.reporting.markdown import render_comparison_report


def test_compare_results_writes_markdown_and_computes_deltas(
    tmp_path: Path,
    read_json,
) -> None:
    baseline_path = _write_result(
        tmp_path / "baseline.json",
        task_type="classification",
        metrics={"accuracy": 0.90, "f1": 0.88},
    )
    candidate_path = _write_result(
        tmp_path / "candidate.json",
        task_type="classification",
        metrics={"accuracy": 0.87, "f1": 0.89},
    )
    markdown_path = tmp_path / "comparison.md"

    result = compare_results(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        thresholds={"accuracy": -0.05},
        markdown_output=markdown_path,
    )

    payload = read_json(baseline_path)
    assert payload["task_type"] == "classification"
    assert result.passed is True
    assert result.metric_deltas[0].metric == "accuracy"
    assert markdown_path.exists()
    assert "## Metric Deltas" in markdown_path.read_text(encoding="utf-8")


def test_compare_results_uses_lower_is_better_delta_thresholds(tmp_path: Path) -> None:
    baseline_path = _write_result(
        tmp_path / "baseline.json",
        task_type="regression",
        metrics={"mae": 0.20, "rmse": 0.25},
    )
    candidate_path = _write_result(
        tmp_path / "candidate.json",
        task_type="regression",
        metrics={"mae": 0.24, "rmse": 0.27},
    )

    passing = compare_results(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        thresholds={"mae": 0.05},
    )
    failing = compare_results(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        thresholds={"mae": 0.03},
    )

    assert passing.passed is True
    assert failing.passed is False


def test_compare_results_can_fail_on_any_regression(tmp_path: Path) -> None:
    baseline_path = _write_result(
        tmp_path / "baseline.json",
        task_type="classification",
        metrics={"accuracy": 0.90},
    )
    candidate_path = _write_result(
        tmp_path / "candidate.json",
        task_type="classification",
        metrics={"accuracy": 0.89},
    )

    result = compare_results(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        fail_on_regression=True,
    )

    assert result.passed is False
    assert result.metric_deltas[0].is_regression is True


def test_compare_results_rejects_mismatched_task_types(tmp_path: Path) -> None:
    baseline_path = _write_result(
        tmp_path / "baseline.json",
        task_type="classification",
        metrics={"accuracy": 0.90},
    )
    candidate_path = _write_result(
        tmp_path / "candidate.json",
        task_type="regression",
        metrics={"mae": 0.20},
    )

    with pytest.raises(ComparisonError, match="different task types"):
        compare_results(baseline_path=baseline_path, candidate_path=candidate_path)


def test_compare_results_rejects_when_shared_metrics_are_missing(tmp_path: Path) -> None:
    baseline_path = _write_result(
        tmp_path / "baseline.json",
        task_type="classification",
        metrics={"accuracy": 0.90},
    )
    candidate_path = _write_result(
        tmp_path / "candidate.json",
        task_type="classification",
        metrics={"f1": 0.88},
    )

    with pytest.raises(ComparisonError, match="do not share any metrics"):
        compare_results(baseline_path=baseline_path, candidate_path=candidate_path)


def test_load_result_rejects_invalid_structure(tmp_path: Path) -> None:
    result_path = tmp_path / "invalid.json"
    result_path.write_text(json.dumps({"task_type": "classification"}), encoding="utf-8")

    with pytest.raises(ComparisonError, match="Expected the JSON structure produced by"):
        load_result(result_path)


def test_load_result_rejects_unknown_metric_for_task(tmp_path: Path) -> None:
    result = EvaluationResult(
        task_type="classification",
        run_name="classification-fixture",
        data_path=Path("data.csv"),
        config_path=Path("config.yaml"),
        row_count=3,
        selected_metrics=["accuracy"],
        metrics={"accuracy": 0.9},
        passed=True,
    ).model_dump(mode="json")
    result["selected_metrics"] = ["accuracy", "mae"]
    result["metrics"]["mae"] = 0.1
    result_path = tmp_path / "invalid_metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with pytest.raises(ComparisonError, match="Unsupported metrics for task 'classification'"):
        load_result(result_path)


def test_load_result_rejects_mismatched_selected_metrics(tmp_path: Path) -> None:
    result = EvaluationResult(
        task_type="classification",
        run_name="classification-fixture",
        data_path=Path("data.csv"),
        config_path=Path("config.yaml"),
        row_count=3,
        selected_metrics=["accuracy"],
        metrics={"accuracy": 0.9},
        passed=True,
    ).model_dump(mode="json")
    result["selected_metrics"] = ["f1"]
    result_path = tmp_path / "mismatched_selected_metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with pytest.raises(ComparisonError, match="selected_metrics"):
        load_result(result_path)


def test_render_comparison_report_includes_threshold_table(tmp_path: Path) -> None:
    baseline_path = _write_result(
        tmp_path / "baseline.json",
        task_type="classification",
        metrics={"accuracy": 0.90},
    )
    candidate_path = _write_result(
        tmp_path / "candidate.json",
        task_type="classification",
        metrics={"accuracy": 0.88},
    )
    result = compare_results(
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        thresholds={"accuracy": -0.03},
    )

    markdown = render_comparison_report(result)

    assert "## Comparison Thresholds" in markdown
    assert "| `accuracy` | `>=` |" in markdown


def _write_result(path: Path, task_type: str, metrics: dict[str, float]) -> Path:
    result = EvaluationResult(
        task_type=task_type,
        run_name=f"{task_type}-fixture",
        data_path=Path("data.csv"),
        config_path=Path("config.yaml"),
        row_count=3,
        selected_metrics=list(metrics),
        metrics=metrics,
        passed=True,
    )
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path
