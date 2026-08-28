"""Validate the two frozen source artifacts and an existing master dataset."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from paragon_lab.dataset.validate_dataset import load_table, validate_dataset
from paragon_lab.utils.config_utils import load_yaml, repo_path


def main() -> int:
    settings = load_yaml(ROOT / "configs/dataset.yaml")["dataset"]
    checks = [
        ("human", settings["human_input"], settings["expected_human_count"]),
        ("ai", settings["ai_input"], settings["expected_ai_count"]),
    ]
    all_valid = True
    for name, source, expected in checks:
        report = validate_dataset(load_table(repo_path(source)), expected_count=expected)
        all_valid &= report["valid"]
        print(f"{name}: {'PASS' if report['valid'] else 'FAIL'} - {report}")
    master = repo_path(settings["master_output"])
    if master.exists():
        report = validate_dataset(load_table(master), expected_count=10000)
        all_valid &= report["valid"]
        print(f"master: {'PASS' if report['valid'] else 'FAIL'} - {report}")
    else:
        print("master: NOT CREATED - run scripts/03_merge_dataset.py")
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
