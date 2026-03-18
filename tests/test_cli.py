from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from orya_eval.cli import app

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"

runner = CliRunner()


def test_root_help_is_clean_and_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Run repeatable evaluations and regression checks" in result.stdout
    assert "init" in result.stdout
    assert "run" in result.stdout
    assert "compare" in result.stdout


def test_compare_help_uses_clear_option_names() -> None:
    result = runner.invoke(app, ["compare", "--help"])

    assert result.exit_code == 0
    assert "--delta-threshold" in result.stdout
    assert "--markdown-report" in result.stdout
    assert "Allowed metric delta as" in result.stdout
    assert "METRIC=VALUE" in result.stdout


def test_init_creates_starter_files(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["init", "--template", "classification", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Status: READY" in result.stdout
    assert (tmp_path / "orya-eval.classification.yaml").exists()
    assert (tmp_path / "classification_sample.csv").exists()


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    (tmp_path / "orya-eval.classification.yaml").write_text("existing", encoding="utf-8")

    result = runner.invoke(
        app,
        ["init", "--template", "classification", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 2
    assert "Refusing to overwrite" in result.stderr


def test_run_classification_example_writes_reports(tmp_path: Path) -> None:
    example_dir = _copy_example("classification", tmp_path)

    result = runner.invoke(app, ["run", str(example_dir / "config.yaml")])

    assert result.exit_code == 0
    assert "Status: PASS" in result.stdout
    assert "Threshold checks:" in result.stdout
    assert "Artifacts:" in result.stdout
    payload = _read_json(example_dir / "reports" / "results.json")
    assert payload["task_type"] == "classification"
    assert payload["passed"] is True
    assert payload["metrics"]["accuracy"] == 5 / 6
    assert "roc_auc" in payload["metrics"]
    assert (example_dir / "reports" / "report.md").exists()


def test_run_regression_example_succeeds(tmp_path: Path) -> None:
    example_dir = _copy_example("regression", tmp_path)

    result = runner.invoke(app, ["run", str(example_dir / "config.yaml")])

    assert result.exit_code == 0
    payload = _read_json(example_dir / "reports" / "results.json")
    assert payload["task_type"] == "regression"
    assert payload["metrics"]["mae"] == pytest.approx(0.22)


def test_run_text_example_succeeds(tmp_path: Path) -> None:
    example_dir = _copy_example("text", tmp_path)

    result = runner.invoke(app, ["run", str(example_dir / "config.yaml")])

    assert result.exit_code == 0
    payload = _read_json(example_dir / "reports" / "results.json")
    assert payload["task_type"] == "text"
    assert payload["metrics"]["exact_match"] == 0.25
    assert payload["metrics"]["contains_match"] == 0.75


def test_run_fails_when_threshold_is_not_met(tmp_path: Path) -> None:
    example_dir = _copy_example("classification", tmp_path)
    config_path = example_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["thresholds"]["accuracy"] = 0.95
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["run", str(config_path)])

    assert result.exit_code == 1
    assert "Status: FAIL" in result.stdout
    assert "threshold checks" in result.stdout.lower()
    assert "one or more threshold checks did not pass" in result.stderr.lower()


def test_run_committed_failing_example_exits_non_zero(tmp_path: Path) -> None:
    example_dir = _copy_example("classification", tmp_path)

    result = runner.invoke(app, ["run", str(example_dir / "failing_thresholds.yaml")])

    assert result.exit_code == 1
    assert "Status: FAIL" in result.stdout
    assert "failing-results.json" in result.stdout


def test_run_missing_config_path_is_actionable(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    result = runner.invoke(app, ["run", str(missing_path)])

    assert result.exit_code == 2
    assert "Config file not found" in result.stderr
    assert "orya-eval run examples/classification/config.yaml" in result.stderr


def test_run_malformed_yaml_is_actionable(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("task_type: classification\ncolumns: [\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(config_path)])

    assert result.exit_code == 2
    assert "Malformed YAML" in result.stderr
    assert "line" in result.stderr


def test_run_missing_columns_is_actionable(tmp_path: Path) -> None:
    example_dir = _copy_example("classification", tmp_path)
    data_path = example_dir / "data.csv"
    data_path.write_text("target,guess,probability\n1,1,0.9\n", encoding="utf-8")

    result = runner.invoke(app, ["run", str(example_dir / "config.yaml")])

    assert result.exit_code == 2
    assert "Input data is missing required columns" in result.stderr
    assert "Update the `columns` mapping" in result.stderr


def test_compare_can_pass_with_explicit_delta_thresholds(tmp_path: Path) -> None:
    comparison_dir = _copy_comparison(tmp_path)
    markdown_path = tmp_path / "comparison.md"

    result = runner.invoke(
        app,
        [
            "compare",
            str(comparison_dir / "classification_baseline.json"),
            str(comparison_dir / "classification_candidate.json"),
            "--delta-threshold",
            "accuracy=-0.05",
            "--delta-threshold",
            "roc_auc=-0.03",
            "--markdown-report",
            str(markdown_path),
        ],
    )

    assert result.exit_code == 0
    assert markdown_path.exists()
    assert "Status: PASS" in result.stdout
    assert "Metric deltas:" in result.stdout
    assert "Delta thresholds:" in result.stdout


def test_compare_fails_on_regression_flag(tmp_path: Path) -> None:
    comparison_dir = _copy_comparison(tmp_path)

    result = runner.invoke(
        app,
        [
            "compare",
            str(comparison_dir / "classification_baseline.json"),
            str(comparison_dir / "classification_candidate.json"),
            "--fail-on-regression",
        ],
    )

    assert result.exit_code == 1
    assert "Status: FAIL" in result.stdout
    assert "Comparison failed because a regression" in result.stderr


def test_compare_malformed_json_is_actionable(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text("{bad json", encoding="utf-8")
    candidate_path.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["compare", str(baseline_path), str(candidate_path)])

    assert result.exit_code == 2
    assert "Malformed JSON" in result.stderr
    assert "Fix the JSON" in result.stderr


def _copy_example(name: str, tmp_path: Path) -> Path:
    source = EXAMPLES_DIR / name
    destination = tmp_path / name
    shutil.copytree(source, destination)
    return destination


def _copy_comparison(tmp_path: Path) -> Path:
    source = EXAMPLES_DIR / "comparison"
    destination = tmp_path / "comparison"
    shutil.copytree(source, destination)
    return destination


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
