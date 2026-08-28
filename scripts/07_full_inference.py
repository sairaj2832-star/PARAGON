"""Launch the full resumable AI-original inference only after explicit confirmation."""
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
    parser.add_argument("--confirm-full-run", action="store_true", help="required acknowledgement before running all eligible samples")
    args = parser.parse_args()
    if not args.confirm_full_run:
        parser.error("full inference is protected; rerun with --confirm-full-run after a successful pilot")
    config = load_yaml(ROOT / "configs/inference.yaml")
    try:
        model, tokenizer = load_model_and_tokenizer(config["model"])
        report = run_inference(model, tokenizer, config["inference"])
        path = ensure_parent(config["inference"]["report_path"])
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0 if report["failed_now"] == 0 else 1
    except KeyboardInterrupt:
        print("Interrupted safely. Completed samples remain in the incremental output; rerun this command to resume.")
        return 130
    except Exception as exc:
        print(f"FULL INFERENCE FAILED: {type(exc).__name__}: {exc}\nCheck outputs/failures/failed_samples.csv and retry after fixing the cause.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
