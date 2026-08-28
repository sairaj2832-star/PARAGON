"""Read the failure log and identify samples eligible for a deliberate retry."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def failed_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, dtype={"sample_id": str})["sample_id"])
