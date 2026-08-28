"""Resumable staged inference with explicit failure and OOM records."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from paragon_lab.inference.checkpoint_manager import CheckpointManager
from paragon_lab.inference.inference_engine import generate_batch


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _clear_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_inference_records(settings: dict[str, Any], sample_limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(settings["input_path"])
    if not path.exists():
        raise FileNotFoundError(f"master dataset is missing: {path}; run scripts/03_merge_dataset.py first")
    frame = pd.read_csv(path, dtype={"sample_id": str})
    source_type = settings.get("source_type")
    if source_type:
        if "source_type" not in frame:
            raise ValueError("master dataset has no source_type column")
        frame = frame[frame["source_type"] == source_type]
    if sample_limit is not None:
        frame = frame.head(sample_limit)
    return frame.to_dict("records")


def run_inference(
    model: Any, tokenizer: Any, settings: dict[str, Any], *, sample_limit: int | None = None, restart: bool = False
) -> dict[str, Any]:
    records = load_inference_records(settings, sample_limit)
    manager = CheckpointManager(
        Path(settings["output_path"]), Path(settings["checkpoint_path"]), Path(settings["failures_path"])
    )
    if restart and manager.output_path.exists():
        raise ValueError("restart requires manually moving prior output; successful samples are never overwritten")
    completed = manager.completed_ids() if settings.get("resume", True) else set()
    pending = [record for record in records if str(record["sample_id"]) not in completed]
    requested_size = int(settings["batch_size"])
    effective_size = requested_size
    configuration = json.dumps(settings, sort_keys=True)
    failures = 0
    fallback_events: list[dict[str, Any]] = []
    index = 0
    while index < len(pending):
        batch = pending[index:index + effective_size]
        try:
            results = generate_batch(model, tokenizer, batch, settings)
            for result in results:
                result["configuration"] = configuration
            manager.append_results(results)
            index += len(batch)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if _is_oom(exc):
                _clear_cuda_cache()
                minimum = int(settings["oom"]["minimum_batch_size"])
                if settings["oom"].get("auto_reduce_batch_size") and effective_size > minimum:
                    new_size = max(minimum, effective_size // 2)
                    fallback_events.append({"event": "cuda_oom_batch_reduction", "from": effective_size, "to": new_size})
                    effective_size = new_size
                    continue
            if len(batch) > 1:
                fallback_events.append({
                    "event": "batch_failure_isolated_to_single_samples",
                    "error_type": type(exc).__name__, "from": len(batch), "to": 1,
                })
                effective_size = 1
                continue
            for record in batch:
                manager.append_failure(str(record["sample_id"]), exc, configuration)
            failures += len(batch)
            index += len(batch)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_requested": len(records), "already_completed": len(completed), "attempted_now": len(pending),
        "successful_total": len(manager.completed_ids()), "failed_now": failures,
        "requested_batch_size": requested_size, "effective_batch_size": effective_size,
        "fallback_events": fallback_events,
    }
