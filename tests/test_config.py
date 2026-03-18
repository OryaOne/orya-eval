from __future__ import annotations

from pathlib import Path

import pytest

from orya_eval.config import ClassificationConfig, load_config
from orya_eval.exceptions import ConfigError


def test_load_config_resolves_relative_paths(tmp_path: Path, write_yaml) -> None:
    data_path = tmp_path / "data.csv"
    data_path.write_text("target,prediction\n1,1\n", encoding="utf-8")
    config_path = write_yaml(
        tmp_path / "config.yaml",
        {
            "task_type": "classification",
            "data_path": "data.csv",
            "columns": {"target": "target", "prediction": "prediction"},
            "reports": {"json": "reports/results.json", "markdown": "reports/report.md"},
        },
    )

    config = load_config(config_path)

    assert isinstance(config, ClassificationConfig)
    assert config.data_path == data_path.resolve()
    assert config.reports.json_output == (tmp_path / "reports" / "results.json").resolve()
    assert config.reports.markdown_output == (tmp_path / "reports" / "report.md").resolve()


def test_load_config_rejects_empty_file(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="is empty"):
        load_config(config_path)


def test_load_config_rejects_top_level_sequence(tmp_path: Path, write_yaml) -> None:
    config_path = write_yaml(tmp_path / "config.yaml", ["not", "a", "mapping"])

    with pytest.raises(ConfigError, match="top-level YAML mapping"):
        load_config(config_path)


def test_load_config_rejects_missing_task_type(tmp_path: Path, write_yaml) -> None:
    config_path = write_yaml(
        tmp_path / "config.yaml",
        {
            "data_path": "data.csv",
            "columns": {"target": "target", "prediction": "prediction"},
        },
    )

    with pytest.raises(ConfigError, match="missing required field `task_type`"):
        load_config(config_path)


def test_load_config_rejects_unsupported_task_type(tmp_path: Path, write_yaml) -> None:
    config_path = write_yaml(
        tmp_path / "config.yaml",
        {
            "task_type": "ranking",
            "data_path": "data.csv",
            "columns": {"target": "target", "prediction": "prediction"},
        },
    )

    with pytest.raises(ConfigError, match="Unsupported task type 'ranking'"):
        load_config(config_path)


def test_load_config_rejects_invalid_metric_for_task(tmp_path: Path, write_yaml) -> None:
    config_path = write_yaml(
        tmp_path / "config.yaml",
        {
            "task_type": "classification",
            "data_path": "data.csv",
            "columns": {"target": "target", "prediction": "prediction"},
            "metrics": ["accuracy", "mae"],
        },
    )

    with pytest.raises(ConfigError, match="Unsupported metrics for task 'classification'"):
        load_config(config_path)


def test_load_config_rejects_invalid_threshold_metric_for_task(
    tmp_path: Path,
    write_yaml,
) -> None:
    config_path = write_yaml(
        tmp_path / "config.yaml",
        {
            "task_type": "text",
            "data_path": "data.jsonl",
            "columns": {"reference": "reference", "prediction": "prediction"},
            "thresholds": {"accuracy": 0.9},
        },
    )

    with pytest.raises(ConfigError, match="Unsupported threshold metrics for task 'text'"):
        load_config(config_path)


def test_load_config_reports_yaml_location_for_malformed_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("task_type: classification\ncolumns: [\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Malformed YAML"):
        load_config(config_path)
