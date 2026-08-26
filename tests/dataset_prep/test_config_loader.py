import pytest
import yaml

from dataset_prep.config_loader import load_config, validate_config, DEFAULT_CONFIG_PATH


def _write_cfg(tmp_path, overrides=None):
    cfg = {
        "dataset": {
            "source": "andythetechnerd03/AI-human-text",
            "source_splits": ["train", "test"],
            "human_samples": 5000,
            "ai_samples": 5000,
            "random_seed": 2026,
        },
        "quality": {
            "remove_missing": True,
            "remove_empty": True,
            "remove_duplicates": True,
        },
        "label_mapping": {"status": "pending_verification"},
        "output": {
            "format_primary": "parquet",
            "format_secondary": "csv",
            "root": "project_dataset",
        },
    }
    if overrides:
        for section, values in overrides.items():
            cfg[section].update(values)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_load_config_reads_frozen_file():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg["dataset"]["random_seed"] == 2026
    assert cfg["dataset"]["human_samples"] == 5000
    assert cfg["dataset"]["ai_samples"] == 5000


def test_validate_accepts_frozen_spec():
    validate_config(load_config(DEFAULT_CONFIG_PATH))  # must not raise


@pytest.mark.parametrize(
    "overrides",
    [
        ({"dataset": {"random_seed": 42}}),
        ({"dataset": {"human_samples": 4000}}),
        ({"dataset": {"ai_samples": 6000}}),
        ({"dataset": {"source": "some/other-dataset"}}),
        ({"quality": {"remove_duplicates": False}}),
        ({"output": {"format_primary": "sqlite"}}),
    ],
)
def test_validate_rejects_spec_violations(tmp_path, overrides):
    p = _write_cfg(tmp_path, overrides)
    with pytest.raises(ValueError):
        validate_config(load_config(p))
