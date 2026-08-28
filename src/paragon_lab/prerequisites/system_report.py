"""Collect a portable lab-readiness report without requiring a GPU."""
from __future__ import annotations

import importlib.metadata
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paragon_lab.utils.config_utils import repo_path


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _ram_gb() -> float | None:
    try:
        import psutil  # optional
        return round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        return None


def collect_environment_report(config: dict[str, Any]) -> dict[str, Any]:
    rules = config["environment"]
    disk = shutil.disk_usage(repo_path("."))
    packages = {name: _version(name) for name in rules["required_packages"]}
    torch_info: dict[str, Any] = {"available": False}
    try:
        import torch
        torch_info = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            torch_info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_vram_gb": round(properties.total_memory / 1024**3, 2),
            })
    except ImportError:
        pass
    python_ok = sys.version_info >= tuple(map(int, rules["min_python"].split(".")))
    checks = {
        "python": python_ok,
        "disk": disk.free / 1024**3 >= rules["min_free_disk_gb"],
        "ram": _ram_gb() is None or _ram_gb() >= rules["min_ram_gb"],
        "packages": all(version is not None for version in packages.values()),
        "torch": torch_info["available"],
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "ram_gb": _ram_gb(),
        "packages": packages,
        "torch": torch_info,
        "checks": checks,
        "ready": all(checks.values()),
    }


def render_environment_report(report: dict[str, Any]) -> str:
    torch_info = report["torch"]
    package_status = "PASS" if report["checks"]["packages"] else "FAIL"
    cuda_status = "PASS" if torch_info.get("cuda_available") else "FAIL (CPU fallback only)"
    return "\n".join([
        "========== PARAGON ENVIRONMENT CHECK ==========",
        f"Python              : {'PASS' if report['checks']['python'] else 'FAIL'} ({report['python']})",
        f"PyTorch             : {'PASS' if report['checks']['torch'] else 'FAIL'} ({torch_info.get('version', 'not installed')})",
        f"CUDA available      : {cuda_status}",
        f"GPU                 : {torch_info.get('gpu_name', 'not detected')}",
        f"GPU VRAM            : {torch_info.get('gpu_vram_gb', 'n/a')} GB",
        f"Packages            : {package_status}",
        f"Disk space          : {'PASS' if report['checks']['disk'] else 'FAIL'} ({report['disk_free_gb']} GB free)",
        f"RAM                 : {'PASS' if report['checks']['ram'] else 'FAIL'} ({report['ram_gb'] or 'unknown'} GB)",
        "OVERALL STATUS     : " + ("READY" if report["ready"] else "NOT READY - inspect failed checks"),
    ])
