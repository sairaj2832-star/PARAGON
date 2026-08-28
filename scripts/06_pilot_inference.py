"""Run a small resumable pilot; start at 5 or 10 samples before 100."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from paragon_lab.inference.runner import run_inference
from paragon_lab.model.load_model import load_model_and_tokenizer
from paragon_lab.utils.config_utils import ensure_parent, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.samples not in {5, 10, 50, 100}:
        parser.error("pilot samples must be one of: 5, 10, 50, 100")
    config = load_yaml(ROOT / "configs/inference.yaml")
    try:
        model, tokenizer = load_model_and_tokenizer(config["model"])
        report = run_inference(model, tokenizer, config["inference"], sample_limit=args.samples)
        if report["successful_total"]:
            report["estimated_full_run_seconds"] = round(
                report["successful_total"] / max(report["attempted_now"], 1) * 5000, 2
            )
        path = ensure_parent("outputs/reports/pilot_report.json")
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["failed_now"] == 0 else 1
    except Exception as exc:
        print(f"PILOT FAILED: {type(exc).__name__}: {exc}\nNext: scripts/04_test_model.py --load")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
