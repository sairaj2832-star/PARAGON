from paragon_lab.inference.checkpoint_manager import CheckpointManager


def test_checkpoint_tracks_only_successes(tmp_path):
    manager = CheckpointManager(tmp_path / "out.csv", tmp_path / "checkpoint.json", tmp_path / "failed.csv")
    manager.append_results([{"sample_id": "A1", "status": "success"}])
    assert manager.completed_ids() == {"A1"}
    assert "completed_samples" in (tmp_path / "checkpoint.json").read_text(encoding="utf-8")


def test_checkpoint_records_failures(tmp_path):
    manager = CheckpointManager(tmp_path / "out.csv", tmp_path / "checkpoint.json", tmp_path / "failed.csv")
    manager.append_failure("A1", ValueError("bad text"), "{}")
    failure_log = (tmp_path / "failed.csv").read_text(encoding="utf-8")
    assert "ValueError" in failure_log
    assert "A1" in failure_log
