from paragon_lab.prerequisites.system_report import collect_environment_report


def test_environment_report_is_structured():
    report = collect_environment_report({"environment": {
        "min_python": "3.0", "min_free_disk_gb": 0, "min_ram_gb": 0, "required_packages": {}
    }})
    assert "checks" in report
    assert "torch" in report
