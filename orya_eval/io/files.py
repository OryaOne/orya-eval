"""File writing utilities that do not require heavy optional imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    Path(path).resolve().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    ensure_parent_dir(path)
    Path(path).resolve().write_text(content, encoding="utf-8")
