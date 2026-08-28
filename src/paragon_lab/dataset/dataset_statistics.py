"""Descriptive statistics for master datasets without changing their contents."""
from __future__ import annotations

from typing import Any

import pandas as pd


def dataset_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    lengths = frame["text"].fillna("").astype(str).str.len()
    words = frame["text"].fillna("").astype(str).str.split().str.len()
    return {
        "records": len(frame),
        "labels": {str(key): int(value) for key, value in frame["label"].value_counts().items()},
        "character_length": {"min": int(lengths.min()), "mean": round(float(lengths.mean()), 2), "max": int(lengths.max())},
        "word_length": {"min": int(words.min()), "mean": round(float(words.mean()), 2), "max": int(words.max())},
    }
