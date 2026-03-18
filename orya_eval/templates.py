"""Starter template generation for `orya-eval init`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orya_eval.exceptions import ConfigError
from orya_eval.io.files import write_text


@dataclass(frozen=True)
class TemplateBundle:
    config_name: str
    data_name: str
    config_content: str
    data_content: str


@dataclass(frozen=True)
class TemplateFiles:
    config_path: Path
    data_path: Path


def create_starter_template(
    template: str, output_dir: str | Path, force: bool = False
) -> TemplateFiles:
    """Write a starter YAML config and small example dataset."""
    key = template.strip().lower()
    if key not in _TEMPLATES:
        raise ConfigError(
            f"Unknown template '{template}'. Choose one of: classification, regression, text."
        )

    bundle = _TEMPLATES[key]
    base_dir = Path(output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    config_path = base_dir / bundle.config_name
    data_path = base_dir / bundle.data_name
    existing = [path for path in [config_path, data_path] if path.exists()]
    if existing and not force:
        existing_text = ", ".join(str(path) for path in existing)
        raise ConfigError(
            "Refusing to overwrite existing files: "
            f"{existing_text}. Re-run with --force to replace them."
        )

    write_text(config_path, bundle.config_content)
    write_text(data_path, bundle.data_content)
    return TemplateFiles(config_path=config_path, data_path=data_path)


_TEMPLATES: dict[str, TemplateBundle] = {
    "classification": TemplateBundle(
        config_name="orya-eval.classification.yaml",
        data_name="classification_sample.csv",
        config_content="""run_name: Starter classification evaluation
description: Small example for local checks or CI smoke tests.
task_type: classification
data_path: classification_sample.csv
columns:
  target: target
  prediction: prediction
  probability: probability
metrics:
  - accuracy
  - precision
  - recall
  - f1
  - roc_auc
thresholds:
  accuracy: 0.80
  roc_auc: 0.90
reports:
  json: reports/classification-results.json
  markdown: reports/classification-report.md
metadata:
  owner: starter
""",
        data_content="""target,prediction,probability
1,1,0.95
0,0,0.10
1,1,0.82
0,0,0.20
1,0,0.35
0,0,0.08
""",
    ),
    "regression": TemplateBundle(
        config_name="orya-eval.regression.yaml",
        data_name="regression_sample.csv",
        config_content="""run_name: Starter regression evaluation
description: Example regression metrics and threshold checks.
task_type: regression
data_path: regression_sample.csv
columns:
  target: target
  prediction: prediction
thresholds:
  mae: 0.35
  rmse: 0.40
reports:
  json: reports/regression-results.json
  markdown: reports/regression-report.md
metadata:
  owner: starter
""",
        data_content="""target,prediction
3.0,2.8
4.5,4.8
5.1,5.0
6.2,6.0
7.0,7.3
""",
    ),
    "text": TemplateBundle(
        config_name="orya-eval.text.yaml",
        data_name="text_sample.jsonl",
        config_content="""run_name: Starter text evaluation
description: Example string-output evaluation with simple exact and fuzzy metrics.
task_type: text
data_path: text_sample.jsonl
columns:
  reference: reference
  prediction: prediction
thresholds:
  exact_match: 0.25
  contains_match: 0.75
reports:
  json: reports/text-results.json
  markdown: reports/text-report.md
metadata:
  owner: starter
""",
        data_content="""{"reference":"Paris","prediction":"The answer is Paris."}
{"reference":"Athens","prediction":"athens"}
{"reference":"red apple","prediction":"fresh red apples"}
{"reference":"open source","prediction":"closed model"}
""",
    ),
}
