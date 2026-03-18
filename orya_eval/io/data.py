"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from orya_eval.exceptions import DataError


def load_dataframe(data_path: str | Path) -> pd.DataFrame:
    """Load evaluation data from CSV or JSONL."""
    path = Path(data_path).resolve()
    if not path.exists():
        raise DataError(
            f"Data file not found: {path}. Update `data_path` in the evaluation config "
            "or create the input file."
        )

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix == ".jsonl":
            frame = pd.read_json(path, lines=True)
        else:
            raise DataError(
                f"Unsupported data file format '{suffix}' for {path}. "
                "Use `.csv` for tabular data or `.jsonl` for record-per-line data."
            )
    except ValueError as exc:
        raise DataError(
            f"Could not read data file '{path}': {exc}. "
            "Ensure the file is valid CSV or JSONL."
        ) from exc

    if frame.empty:
        raise DataError(
            f"Data file '{path}' contains no rows. Add at least one record before running "
            "the evaluation."
        )
    return frame
