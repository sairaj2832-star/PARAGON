"""Transparent lexical proxy checks; not a substitute for a semantic-similarity model."""
from __future__ import annotations

import re
from typing import Any

TOKEN = re.compile(r"\b\w+\b")


def lexical_quality_proxy(input_text: str, generated_text: str) -> dict[str, Any]:
    source = set(TOKEN.findall(input_text.lower()))
    generated = set(TOKEN.findall(generated_text.lower()))
    union = source | generated
    return {
        "jaccard_token_overlap": round(len(source & generated) / len(union), 6) if union else 0.0,
        "length_ratio": round(len(generated_text) / len(input_text), 6) if input_text else None,
        "note": "Lexical overlap is a diagnostic proxy, not a semantic-equivalence score.",
    }
