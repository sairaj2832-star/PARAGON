"""Check checkpoint/tokenizer accessibility; use --load for a real model load."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from paragon_lab.model.check_checkpoint import check_checkpoint
from paragon_lab.model.load_model import load_model_and_tokenizer, model_details
from paragon_lab.utils.config_utils import ensure_parent, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="load all model weights; may download a large checkpoint")
    args = parser.parse_args()
    model_config = load_yaml(ROOT / "configs/inference.yaml")["model"]
    report = check_checkpoint(model_config)
    if args.load and report["accessible"]:
        try:
            model, _ = load_model_and_tokenizer(model_config)
            report["model_load"] = {"status": "PASS", **model_details(model, model_config)}
        except Exception as exc:
            report["model_load"] = {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}
    path = ensure_parent("outputs/reports/model_check.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["accessible"] and report.get("model_load", {"status": "PASS"})["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
