from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from orya_eval.config import ClassificationConfig
from orya_eval.exceptions import ConfigError, DataError
from orya_eval.io.data import load_dataframe
from orya_eval.reporting.markdown import render_run_report
from orya_eval.runner import evaluate_thresholds, run_evaluation


def test_load_dataframe_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("target,prediction\n1,1\n0,0\n", encoding="utf-8")

    frame = load_dataframe(csv_path)

    assert list(frame.columns) == ["target", "prediction"]
    assert len(frame) == 2


def test_load_dataframe_reads_jsonl(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        '{"reference":"Paris","prediction":"Paris"}\n'
        '{"reference":"Athens","prediction":"athens"}\n',
        encoding="utf-8",
    )

    frame = load_dataframe(jsonl_path)

    assert list(frame.columns) == ["reference", "prediction"]
    assert len(frame) == 2


def test_load_dataframe_rejects_unsupported_suffix(tmp_path: Path) -> None:
    data_path = tmp_path / "data.txt"
    data_path.write_text("not supported", encoding="utf-8")

    with pytest.raises(DataError, match="Unsupported data file format"):
        load_dataframe(data_path)


def test_load_dataframe_rejects_empty_csv(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("target,prediction\n", encoding="utf-8")

    with pytest.raises(DataError, match="contains no rows"):
        load_dataframe(data_path)


def test_run_evaluation_writes_json_and_markdown_reports(
    tmp_path: Path,
    classification_frame: pd.DataFrame,
    read_json,
) -> None:
    data_path = tmp_path / "data.csv"
    classification_frame.to_csv(data_path, index=False)
    config = ClassificationConfig(
        task_type="classification",
        run_name="test run",
        description="A small classification regression check.",
        data_path=data_path,
        columns={
            "target": "target",
            "prediction": "prediction",
            "probability": "probability",
        },
        metrics=["accuracy", "roc_auc"],
        thresholds={"accuracy": 0.8},
        reports={
            "json": tmp_path / "reports" / "results.json",
            "markdown": tmp_path / "reports" / "report.md",
        },
        metadata={"dataset": "fixture"},
    )

    result = run_evaluation(config, config_path=tmp_path / "config.yaml")
    payload = read_json(tmp_path / "reports" / "results.json")
    markdown = (tmp_path / "reports" / "report.md").read_text(encoding="utf-8")

    assert result.passed is True
    assert payload["run_name"] == "test run"
    assert payload["metrics"]["accuracy"] == pytest.approx(5 / 6)
    assert payload["thresholds"][0]["passed"] is True
    assert "# test run" in markdown
    assert "## Thresholds" in markdown


def test_run_evaluation_rejects_threshold_for_metric_not_computed(
    tmp_path: Path,
    classification_frame: pd.DataFrame,
) -> None:
    data_path = tmp_path / "data.csv"
    classification_frame.to_csv(data_path, index=False)
    config = ClassificationConfig(
        task_type="classification",
        data_path=data_path,
        columns={"target": "target", "prediction": "prediction"},
        metrics=["accuracy"],
        thresholds={"f1": 0.8},
        reports={"json": tmp_path / "results.json"},
    )

    with pytest.raises(ConfigError, match="but that metric was not computed"):
        run_evaluation(config, config_path=tmp_path / "config.yaml")


def test_evaluate_thresholds_respects_metric_direction() -> None:
    checks = evaluate_thresholds(
        metrics={"accuracy": 0.82, "mae": 0.21},
        thresholds={"accuracy": 0.8, "mae": 0.2},
    )

    by_metric = {check.metric: check for check in checks}
    assert by_metric["accuracy"].passed is True
    assert by_metric["accuracy"].operator == ">="
    assert by_metric["mae"].passed is False
    assert by_metric["mae"].operator == "<="


def test_render_run_report_includes_metrics_and_threshold_rows(
    tmp_path: Path,
    classification_frame: pd.DataFrame,
) -> None:
    data_path = tmp_path / "data.csv"
    classification_frame.to_csv(data_path, index=False)
    config = ClassificationConfig(
        task_type="classification",
        run_name="report fixture",
        data_path=data_path,
        columns={"target": "target", "prediction": "prediction"},
        metrics=["accuracy"],
        thresholds={"accuracy": 0.9},
        reports={"json": tmp_path / "results.json"},
    )

    result = run_evaluation(config, config_path=tmp_path / "config.yaml")
    markdown = render_run_report(result)

    assert "| `accuracy` |" in markdown
    assert "## Thresholds" in markdown
    assert "| `accuracy` | `>=` |" in markdown
