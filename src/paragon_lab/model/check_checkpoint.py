"""Validate tokenizer/checkpoint accessibility without running a full experiment."""
from __future__ import annotations

from typing import Any


def check_checkpoint(model_config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"name_or_path": model_config["name_or_path"], "accessible": False}
    try:
        from transformers import AutoConfig, AutoTokenizer
        kwargs = {
            "local_files_only": model_config.get("local_files_only", False),
            "trust_remote_code": model_config.get("trust_remote_code", False),
        }
        config = AutoConfig.from_pretrained(model_config["name_or_path"], **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_config["name_or_path"], **kwargs)
        result.update({
            "accessible": True,
            "model_type": config.model_type,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
        })
    except Exception as exc:  # expose type/message to the lab operator
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
    return result
