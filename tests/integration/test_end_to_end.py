from site_safety_monitor.pipelines.run_monitor import run_case


def test_end_to_end_work_at_height_case_returns_no_and_head_injury_hazard():
    result = run_case(
        regulation_path="tests/fixtures/regulations/work_at_height.json",
        scene_path="tests/fixtures/scenes/work_at_height.json",
    )

    assert result["compliance"] == "No"
    assert result["hazards"] == ["head injury from falls"]
