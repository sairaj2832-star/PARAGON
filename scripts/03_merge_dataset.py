"""Create data/processed/master_10k.csv and its validation reports."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from paragon_lab.dataset.merge_dataset import merge_master_dataset, render_dataset_report
from paragon_lab.utils.config_utils import load_yaml


def main() -> int:
    _, report = merge_master_dataset(load_yaml(ROOT / "configs/dataset.yaml"))
    print(render_dataset_report(report))
    return 0 if report["master"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
