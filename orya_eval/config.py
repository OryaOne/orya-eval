"""Configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from orya_eval.exceptions import ConfigError
from orya_eval.metrics.registry import get_task_metrics


class ReportOutputs(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    json_output: Path = Field(default=Path("results.json"), alias="json")
    markdown_output: Path | None = Field(default=None, alias="markdown")


class ClassificationColumns(BaseModel):
    target: str
    prediction: str
    probability: str | None = None


class RegressionColumns(BaseModel):
    target: str
    prediction: str


class TextColumns(BaseModel):
    reference: str
    prediction: str


class EvalConfigBase(BaseModel):
    task_type: Literal["classification", "regression", "text"]
    run_name: str | None = None
    description: str | None = None
    data_path: Path
    metrics: list[str] | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    reports: ReportOutputs = Field(default_factory=ReportOutputs)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(
        cls,
        metrics: list[str] | None,
        info: ValidationInfo,
    ) -> list[str] | None:
        if metrics is None:
            return None
        task_type = info.data.get("task_type")
        if task_type is None:
            return metrics
        allowed = get_task_metrics(task_type)
        invalid = [metric for metric in metrics if metric not in allowed]
        if invalid:
            allowed_text = ", ".join(sorted(allowed))
            invalid_text = ", ".join(invalid)
            raise ValueError(
                f"Unsupported metrics for task '{task_type}': "
                f"{invalid_text}. Allowed: {allowed_text}."
            )
        return metrics

    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(
        cls,
        thresholds: dict[str, float],
        info: ValidationInfo,
    ) -> dict[str, float]:
        task_type = info.data.get("task_type")
        if task_type is None:
            return thresholds
        allowed = get_task_metrics(task_type)
        invalid = [metric for metric in thresholds if metric not in allowed]
        if invalid:
            allowed_text = ", ".join(sorted(allowed))
            invalid_text = ", ".join(invalid)
            raise ValueError(
                f"Unsupported threshold metrics for task '{task_type}': "
                f"{invalid_text}. Allowed: {allowed_text}."
            )
        return thresholds


class ClassificationConfig(EvalConfigBase):
    task_type: Literal["classification"]
    columns: ClassificationColumns


class RegressionConfig(EvalConfigBase):
    task_type: Literal["regression"]
    columns: RegressionColumns


class TextConfig(EvalConfigBase):
    task_type: Literal["text"]
    columns: TextColumns


EvalConfig = ClassificationConfig | RegressionConfig | TextConfig

_CONFIG_MODELS = {
    "classification": ClassificationConfig,
    "regression": RegressionConfig,
    "text": TextConfig,
}


def _resolve_path(base_dir: Path, value: Path | None) -> Path | None:
    if value is None or value.is_absolute():
        return value
    return (base_dir / value).resolve()


def load_config(config_path: str | Path) -> EvalConfig:
    """Load, validate, and resolve a YAML config file."""
    path = Path(config_path).resolve()
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}. Provide a path to a YAML file, for example "
            "`orya-eval run examples/classification/config.yaml`."
        )

    try:
        raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(_format_yaml_error(path, exc)) from exc

    if raw_data is None:
        raise ConfigError(
            f"Config file '{path}' is empty. Add a YAML mapping with at least `task_type`, "
            "`data_path`, and `columns`."
        )
    if not isinstance(raw_data, dict):
        raise ConfigError(
            f"Config file '{path}' must contain a top-level YAML mapping. "
            "Example: `task_type: classification`."
        )

    task_type = raw_data.get("task_type")
    if task_type is None:
        allowed = ", ".join(sorted(_CONFIG_MODELS))
        raise ConfigError(
            f"Config file '{path}' is missing required field `task_type`. Choose one of: {allowed}."
        )
    if task_type not in _CONFIG_MODELS:
        allowed = ", ".join(sorted(_CONFIG_MODELS))
        raise ConfigError(
            f"Unsupported task type '{task_type}' in '{path}'. Choose one of: {allowed}."
        )

    try:
        config = _CONFIG_MODELS[task_type].model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc

    base_dir = path.parent
    updated = config.model_copy(
        update={
            "data_path": _resolve_path(base_dir, config.data_path),
            "reports": config.reports.model_copy(
                update={
                    "json_output": _resolve_path(base_dir, config.reports.json_output),
                    "markdown_output": _resolve_path(
                        base_dir,
                        config.reports.markdown_output,
                    ),
                }
            ),
        }
    )
    return updated


def _format_yaml_error(path: Path, exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None)
    if mark is not None:
        return (
            f"Malformed YAML in '{path}' at line {mark.line + 1}, column {mark.column + 1}. "
            "Check indentation, colons, and list formatting."
        )
    return f"Malformed YAML in '{path}'. Check indentation and YAML syntax."


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    details: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(_humanize_location(error["loc"]))
        details.append(f"- {location}: {error['msg']}")

    return (
        f"Invalid evaluation config '{path}'. Fix the fields below and run the command again.\n"
        + "\n".join(details)
    )


def _humanize_location(location: tuple[object, ...]) -> list[str]:
    alias_map = {
        "json_output": "json",
        "markdown_output": "markdown",
    }
    parts: list[str] = []
    for part in location:
        text = str(part)
        parts.append(alias_map.get(text, text))
    return parts
