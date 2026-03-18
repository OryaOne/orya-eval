"""Shared helpers for task evaluation modules."""

from __future__ import annotations

import pandas as pd

from orya_eval.exceptions import DataError


def validate_required_columns(frame: pd.DataFrame, required: list[str]) -> None:
    """Ensure the input frame contains the configured columns."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        missing_text = ", ".join(missing)
        available_text = ", ".join(str(column) for column in frame.columns)
        raise DataError(
            f"Input data is missing required columns: {missing_text}. "
            "Update the `columns` mapping in the config or fix the input file headers. "
            f"Available columns: {available_text}."
        )
