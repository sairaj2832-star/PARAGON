"""Check Python, package, GPU, CUDA, memory, and disk readiness."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from paragon_lab.prerequisites.system_report import collect_environment_report, render_environment_report
from paragon_lab.utils.config_utils import load_yaml, ensure_parent


def main() -> int:
    report = collect_environment_report(load_yaml(ROOT / "configs/environment.yaml"))
    path = ensure_parent("data/reports/environment_report.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render_environment_report(report))
    print(f"Report written: {path.relative_to(ROOT)}")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
