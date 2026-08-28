import pandas as pd

from paragon_lab.dataset.merge_dataset import merge_master_dataset
from paragon_lab.dataset.validate_dataset import validate_dataset


def _frame(prefix, label):
    return pd.DataFrame({
        "sample_id": [f"{prefix}00001", f"{prefix}00002"],
        "source_row_id": [1, 2],
        "text": ["one", "two"],
        "label": [label, label],
        "text_type": ["human" if label == 0 else "ai_original"] * 2,
        "parent_id": [None, None],
        "paraphrase_level": [None, None],
        "generation_model": [None, None],
    })


def test_validation_rejects_empty_text():
    report = validate_dataset(pd.DataFrame({"sample_id": ["A"], "text": [""], "label": [1]}))
    assert not report["valid"]
    assert report["empty_text"] == 1


def test_merge_preserves_lineage_aliases(tmp_path, monkeypatch):
    human, ai = _frame("H", 0), _frame("A", 1)
    human_path, ai_path = tmp_path / "human.parquet", tmp_path / "ai.parquet"
    human.to_parquet(human_path); ai.to_parquet(ai_path)
    config = {"dataset": {
        "version": "test", "human_input": str(human_path), "ai_input": str(ai_path),
        "master_output": str(tmp_path / "master.csv"), "report_json": str(tmp_path / "report.json"),
        "report_text": str(tmp_path / "report.txt"),
        "expected_human_count": 2, "expected_ai_count": 2,
    }}
    master, report = merge_master_dataset(config)
    assert report["master"]["valid"]
    assert master["source_id"].tolist() == [1, 2, 1, 2]
    assert master["source_type"].tolist() == ["human", "human", "ai_original", "ai_original"]
