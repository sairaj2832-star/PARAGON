"""Load configured seq2seq checkpoints with an explicit precision policy."""
from __future__ import annotations

from typing import Any


def _imports():
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Model dependencies are missing. Install requirements-gpu.txt and the lab-specific PyTorch CUDA build."
        ) from exc
    return torch, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(model_config: dict[str, Any]):
    torch, model_class, tokenizer_class, quantization_class = _imports()
    precision = model_config["dtype"]
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if precision not in dtype_map:
        raise ValueError(f"unsupported dtype: {precision}")
    quantization = model_config.get("quantization", "none")
    kwargs: dict[str, Any] = {
        "local_files_only": model_config.get("local_files_only", False),
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "torch_dtype": dtype_map[precision],
    }
    if model_config.get("device_map"):
        kwargs["device_map"] = model_config["device_map"]
    if quantization == "int8":
        kwargs["quantization_config"] = quantization_class(load_in_8bit=True)
    elif quantization == "int4":
        kwargs["quantization_config"] = quantization_class(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
    elif quantization != "none":
        raise ValueError("quantization must be one of: none, int8, int4")
    name = model_config["name_or_path"]
    tokenizer = tokenizer_class.from_pretrained(
        name,
        local_files_only=model_config.get("local_files_only", False),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    model = model_class.from_pretrained(name, **kwargs)
    model.eval()
    return model, tokenizer


def model_details(model: Any, model_config: dict[str, Any]) -> dict[str, Any]:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    device = str(getattr(model, "device", "managed by device_map"))
    details = {
        "parameter_count": parameter_count,
        "dtype": str(next(model.parameters()).dtype),
        "device": device,
        "configured_quantization": model_config.get("quantization", "none"),
    }
    try:
        import torch
        if torch.cuda.is_available():
            details["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
            details["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
    except ImportError:
        pass
    return details
