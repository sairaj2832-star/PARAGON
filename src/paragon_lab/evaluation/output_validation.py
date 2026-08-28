"""Validate persisted inference output before it is used for experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def validate_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"valid": False, "error": f"output does not exist: {path}"}
    frame = pd.read_csv(path, dtype={"sample_id": str})
    required = {"sample_id", "generated_text", "status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"valid": False, "error": f"missing columns: {missing}"}
    successful = frame[frame["status"] == "success"]
    report = {
        "rows": len(frame), "successful": len(successful),
        "duplicate_sample_ids": int(successful["sample_id"].duplicated().sum()),
        "empty_outputs": int(successful["generated_text"].fillna("").astype(str).str.strip().eq("").sum()),
    }
    report["valid"] = report["duplicate_sample_ids"] == 0 and report["empty_outputs"] == 0
    return report
