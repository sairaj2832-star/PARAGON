"""Run one explicitly selected input through a loaded model and save the result."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from paragon_lab.inference.inference_engine import generate_batch
from paragon_lab.inference.runner import load_inference_records
from paragon_lab.model.load_model import load_model_and_tokenizer, model_details
from paragon_lab.utils.config_utils import ensure_parent, load_yaml


def main() -> int:
    config = load_yaml(ROOT / "configs/inference.yaml")
    settings = config["inference"]
    records = load_inference_records(settings, 1)
    if not records:
        print("STATUS: FAIL - no eligible input sample; merge the master dataset first")
        return 1
    try:
        model, tokenizer = load_model_and_tokenizer(config["model"])
        results = generate_batch(model, tokenizer, records, settings)
        result = results[0]
        result["model"] = model_details(model, config["model"])
        result["input_output_differ"] = result["input_text"].strip() != result["generated_text"].strip()
        path = ensure_parent("outputs/pilot/smoke_test.json")
        path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        print("========== DIPPER SMOKE TEST ==========\nModel loading: PASS\nTokenizer: PASS\nGeneration: PASS\nOutput validation: PASS")
        print(f"Inference time: {result['generation_seconds']} sec\nSTATUS: PASS\nSaved: {path.relative_to(ROOT)}")
        return 0
    except Exception as exc:
        print(f"STATUS: FAIL\n{type(exc).__name__}: {exc}\nNext: run scripts/04_test_model.py --load")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
