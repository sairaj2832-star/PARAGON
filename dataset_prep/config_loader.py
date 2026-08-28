"""Load and strictly validate the frozen Phase 0 configuration."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = REPO_ROOT / "project_dataset"
RAW_DIR = PROJECT_DIR / "raw" / "source_dataset"
PROCESSED_DIR = PROJECT_DIR / "processed"
SAMPLED_DIR = PROJECT_DIR / "sampled"
EXPORTS_DIR = PROJECT_DIR / "exports"
CONFIGS_DIR = PROJECT_DIR / "configs"
REPORTS_DIR = PROJECT_DIR / "reports"

DEFAULT_CONFIG_PATH = CONFIGS_DIR / "sampling_config.yaml"

FROZEN_SPEC = {
    "source": "andythetechnerd03/AI-human-text",
    "source_splits": ["train", "test"],
    "human_samples": 5000,
    "ai_samples": 5000,
    "random_seed": 2026,
    "remove_missing": True,
    "remove_empty": True,
    "remove_duplicates": True,
    "format_primary": "parquet",
    "format_secondary": "csv",
}


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(cfg: dict) -> None:
    d, q, o = cfg["dataset"], cfg["quality"], cfg["output"]
    problems = []
    if d["source"] != FROZEN_SPEC["source"]:
        problems.append(f"dataset.source must be {FROZEN_SPEC['source']!r}")
    source_splits = d.get("source_splits", d.get("splits_to_use"))
    if source_splits != FROZEN_SPEC["source_splits"]:
        problems.append("dataset.source_splits must be ['train', 'test']")
    if d["human_samples"] != FROZEN_SPEC["human_samples"]:
        problems.append("dataset.human_samples must be 5000")
    if d["ai_samples"] != FROZEN_SPEC["ai_samples"]:
        problems.append("dataset.ai_samples must be 5000")
    if d["random_seed"] != FROZEN_SPEC["random_seed"]:
        problems.append("dataset.random_seed must be 2026")
    for k in ("remove_missing", "remove_empty", "remove_duplicates"):
        if q[k] is not FROZEN_SPEC[k]:
            problems.append(f"quality.{k} must be true")
    if o["format_primary"] != FROZEN_SPEC["format_primary"]:
        problems.append("output.format_primary must be 'parquet'")
    if o["format_secondary"] != FROZEN_SPEC["format_secondary"]:
        problems.append("output.format_secondary must be 'csv'")
    if problems:
        raise ValueError(
            "Config violates the frozen Phase 0 spec:\n- " + "\n- ".join(problems)
        )


if __name__ == "__main__":
    cfg = load_config()
    validate_config(cfg)
    print("CONFIG OK (frozen Phase 0 spec satisfied)")
