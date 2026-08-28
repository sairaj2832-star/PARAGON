"""Incremental output and checkpoint state for crash-safe experiment resumes."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from paragon_lab.utils.file_utils import atomic_write_json

OUTPUT_FIELDS = [
    "sample_id", "parent_sample_id", "source_id", "input_text", "generated_text",
    "status", "generation_seconds", "input_tokens", "output_tokens", "configuration",
]
FAILURE_FIELDS = ["timestamp", "sample_id", "error_type", "error_message", "configuration"]


class CheckpointManager:
    def __init__(self, output_path: Path, checkpoint_path: Path, failures_path: Path):
        self.output_path = output_path
        self.checkpoint_path = checkpoint_path
        self.failures_path = failures_path

    def completed_ids(self) -> set[str]:
        if not self.output_path.exists():
            return set()
        with self.output_path.open("r", newline="", encoding="utf-8") as handle:
            return {row["sample_id"] for row in csv.DictReader(handle) if row.get("status") == "success"}

    def append_results(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.output_path.exists()
        with self.output_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        self._write_checkpoint()

    def append_failure(self, sample_id: str, exc: Exception, configuration: str) -> None:
        self.failures_path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.failures_path.exists()
        with self.failures_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FAILURE_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(), "sample_id": sample_id,
                "error_type": type(exc).__name__, "error_message": str(exc),
                "configuration": configuration,
            })

    def _write_checkpoint(self) -> None:
        atomic_write_json(self.checkpoint_path, {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "completed_samples": len(self.completed_ids()),
            "output_path": str(self.output_path),
        })
