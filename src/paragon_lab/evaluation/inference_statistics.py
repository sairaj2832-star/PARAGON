"""Compute lightweight summary statistics for generated output."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def inference_statistics(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    success = frame[frame["status"] == "success"]
    return {
        "total_attempted": len(frame), "successful": len(success), "failed": len(frame) - len(success),
        "failure_rate": round((len(frame) - len(success)) / len(frame), 6) if len(frame) else 0.0,
        "average_inference_seconds": float(success["generation_seconds"].mean()) if len(success) else None,
        "average_input_tokens": float(success["input_tokens"].mean()) if len(success) else None,
        "average_output_tokens": float(success["output_tokens"].mean()) if len(success) else None,
    }
