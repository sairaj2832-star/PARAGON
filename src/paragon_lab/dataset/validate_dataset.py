"""Validate PARAGON source or master datasets without mutating them."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ("sample_id", "text", "label")


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"sample_id": str})
    raise ValueError(f"unsupported dataset format: {path.suffix}")


def validate_dataset(frame: pd.DataFrame, *, expected_count: int | None = None) -> dict[str, Any]:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    report: dict[str, Any] = {
        "rows": len(frame),
        "missing_columns": missing_columns,
        "missing_text": None,
        "empty_text": None,
        "duplicate_ids": None,
        "duplicate_texts": None,
        "invalid_labels": None,
        "label_counts": {},
        "valid": False,
    }
    if missing_columns:
        return report
    text = frame["text"]
    labels = pd.to_numeric(frame["label"], errors="coerce")
    report.update({
        "missing_text": int(text.isna().sum()),
        "empty_text": int(text.fillna("").astype(str).str.strip().eq("").sum()),
        "duplicate_ids": int(frame["sample_id"].duplicated().sum()),
        "duplicate_texts": int(text.duplicated().sum()),
        "invalid_labels": int((~labels.isin([0, 1])).sum()),
        "label_counts": {str(key): int(value) for key, value in Counter(labels.dropna()).items()},
    })
    count_ok = expected_count is None or len(frame) == expected_count
    report["expected_count"] = expected_count
    report["valid"] = bool(
        count_ok and report["missing_text"] == 0 and report["empty_text"] == 0
        and report["duplicate_ids"] == 0 and report["invalid_labels"] == 0
    )
    return report
