"""Pure generation operations; persistence and resume live in the runner."""
from __future__ import annotations

import time
from typing import Any


def build_prompt(text: str, settings: dict[str, Any]) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input text is empty")
    return settings["prompt_template"].format(
        text=text.strip(), lexical_diversity=settings["lexical_diversity"], order_diversity=settings["order_diversity"]
    )


def _model_device(model: Any) -> Any:
    return getattr(model, "device", None)


def _token_count(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer(text, truncation=True).input_ids
    return len(token_ids[0]) if token_ids and isinstance(token_ids[0], list) else len(token_ids)


def generate_batch(model: Any, tokenizer: Any, records: list[dict[str, Any]], settings: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = [build_prompt(str(record["text"]), settings) for record in records]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=settings["max_input_tokens"])
    device = _model_device(model)
    if device is not None and hasattr(encoded, "to"):
        encoded = encoded.to(device)
    generation = {key: value for key, value in settings["generation"].items() if value is not None}
    started = time.perf_counter()
    outputs = model.generate(**encoded, max_new_tokens=settings["max_new_tokens"], **generation)
    elapsed = time.perf_counter() - started
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results = []
    for record, prompt, output in zip(records, prompts, decoded, strict=True):
        if not output or not output.strip():
            raise RuntimeError(f"model generated empty output for {record['sample_id']}")
        results.append({
            "sample_id": str(record["sample_id"]),
            "parent_sample_id": record.get("parent_sample_id") or "",
            "source_id": record.get("source_id") or "",
            "input_text": record["text"], "generated_text": output.strip(), "status": "success",
            "generation_seconds": round(elapsed / len(records), 6),
            "input_tokens": _token_count(tokenizer, prompt), "output_tokens": _token_count(tokenizer, output),
        })
    return results
