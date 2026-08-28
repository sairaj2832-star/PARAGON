from paragon_lab.evaluation.output_validation import validate_output


def test_output_validation_rejects_empty_generated_text(tmp_path):
    path = tmp_path / "outputs.csv"
    path.write_text("sample_id,generated_text,status\nA1,,success\n", encoding="utf-8")
    report = validate_output(path)
    assert not report["valid"]
    assert report["empty_outputs"] == 1
