from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml


@pytest.fixture
def classification_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [1, 0, 1, 0, 1, 0],
            "prediction": [1, 0, 1, 0, 0, 0],
            "probability": [0.95, 0.10, 0.82, 0.20, 0.35, 0.08],
        }
    )


@pytest.fixture
def regression_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [3.0, 4.5, 5.1, 6.2, 7.0],
            "prediction": [2.8, 4.8, 5.0, 6.0, 7.3],
        }
    )


@pytest.fixture
def text_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reference": ["Paris", "Athens", "red apple", "open source"],
            "prediction": [
                "The answer is Paris.",
                "athens",
                "fresh red apples",
                "closed model",
            ],
        }
    )


@pytest.fixture
def write_yaml() -> Any:
    def _write_yaml(path: Path, payload: Any) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    return _write_yaml


@pytest.fixture
def read_json() -> Any:
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    return _read_json
