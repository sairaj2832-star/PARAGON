"""Build a lineage-preserving master CSV from the frozen source artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from paragon_lab.dataset.validate_dataset import load_table, validate_dataset
from paragon_lab.utils.config_utils import ensure_parent, repo_path


def _lineage_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    aliases = {
        "source_id": "source_row_id",
        "source_type": "text_type",
        "parent_sample_id": "parent_id",
        "paraphraser": "generation_model",
    }
    for target, source in aliases.items():
        if target not in result:
            result[target] = result[source] if source in result else None
    for column in ("paraphrase_level", "parent_sample_id", "paraphraser"):
        if column not in result:
            result[column] = None
    return result


def merge_master_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = config["dataset"]
    human = load_table(repo_path(settings["human_input"]))
    ai = load_table(repo_path(settings["ai_input"]))
    human_report = validate_dataset(human, expected_count=settings["expected_human_count"])
    ai_report = validate_dataset(ai, expected_count=settings["expected_ai_count"])
    if not human_report["valid"] or not ai_report["valid"]:
        raise ValueError(f"source validation failed: human={human_report}, ai={ai_report}")
    master = _lineage_columns(pd.concat([human, ai], ignore_index=True, sort=False))
    master_report = validate_dataset(
        master, expected_count=settings["expected_human_count"] + settings["expected_ai_count"]
    )
    class_counts = master["label"].value_counts().to_dict()
    if class_counts.get(0, 0) != settings["expected_human_count"] or class_counts.get(1, 0) != settings["expected_ai_count"]:
        raise ValueError(f"master label counts do not match configuration: {class_counts}")
    output_path = ensure_parent(settings["master_output"])
    master.to_csv(output_path, index=False)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": settings["version"],
        "inputs": {"human": settings["human_input"], "ai": settings["ai_input"]},
        "output": str(output_path),
        "human": human_report,
        "ai": ai_report,
        "master": master_report,
        "master_label_counts": {str(key): int(value) for key, value in class_counts.items()},
    }
    json_path = ensure_parent(settings["report_json"])
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text_path = ensure_parent(settings["report_text"])
    text_path.write_text(render_dataset_report(report), encoding="utf-8")
    return master, report


def render_dataset_report(report: dict[str, Any]) -> str:
    master = report["master"]
    return "\n".join([
        "========== PARAGON DATASET REPORT ==========",
        f"Dataset version      : {report['dataset_version']}",
        f"Human count          : {report['human']['rows']}",
        f"AI count             : {report['ai']['rows']}",
        f"Total                : {master['rows']}",
        f"Missing text         : {master['missing_text']}",
        f"Empty samples        : {master['empty_text']}",
        f"Duplicate IDs        : {master['duplicate_ids']}",
        f"Invalid labels       : {master['invalid_labels']}",
        f"Duplicate texts      : {master['duplicate_texts']} (reported; not a merge failure)",
        "STATUS               : " + ("PASS" if master["valid"] else "FAIL"),
        "",
    ])
