"""Create deterministic, family-level splits to prevent paraphrase leakage."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd


def create_grouped_splits(frame: pd.DataFrame, settings: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    ratios = settings["splits"]
    if round(sum(ratios.values()), 8) != 1.0:
        raise ValueError("split ratios must sum to 1")
    key = frame["parent_sample_id"].where(frame["parent_sample_id"].notna(), frame["sample_id"])
    groups = list(dict.fromkeys(key.astype(str)))
    random.Random(settings["seed"]).shuffle(groups)
    boundaries = {
        "train": round(len(groups) * ratios["train"]),
        "validation": round(len(groups) * (ratios["train"] + ratios["validation"])),
    }
    allocation = {
        "train": set(groups[:boundaries["train"]]),
        "validation": set(groups[boundaries["train"]:boundaries["validation"]]),
        "test": set(groups[boundaries["validation"]:]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, group_ids in allocation.items():
        split = frame[key.astype(str).isin(group_ids)]
        path = output_dir / f"{name}.csv"
        split.to_csv(path, index=False)
        result[name] = path
    return result
