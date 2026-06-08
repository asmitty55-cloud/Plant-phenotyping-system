import importlib


def test_lighting_transitions_record_only_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("PT_DATA_ROOT", str(tmp_path))
    import pt.core.analysis.metric_store as metric_store

    metric_store = importlib.reload(metric_store)

    first = metric_store.record_lighting_transition("cam1", "2026-06-09 21:00:00", "dark", filename="a.jpg", luma=20)
    duplicate = metric_store.record_lighting_transition("cam1", "2026-06-09 21:04:00", "dark", filename="b.jpg", luma=19)
    second = metric_store.record_lighting_transition("cam1", "2026-06-10 05:40:00", "light", filename="c.jpg", luma=80)

    transitions = metric_store.lighting_transitions_for_device("cam1")

    assert first["from_mode"] is None
    assert duplicate is None
    assert second["from_mode"] == "dark"
    assert [row["to_mode"] for row in transitions] == ["light", "dark"]


def test_lighting_summary_uses_adjacent_transitions(tmp_path, monkeypatch):
    monkeypatch.setenv("PT_DATA_ROOT", str(tmp_path))
    import pt.core.analysis.metric_store as metric_store
    import pt.api.app as api_app

    metric_store = importlib.reload(metric_store)
    api_app.metric_store = metric_store

    metric_store.record_lighting_transition("cam1", "2026-06-09 06:00:00", "light")
    metric_store.record_lighting_transition("cam1", "2026-06-09 21:30:00", "dark")
    metric_store.record_lighting_transition("cam1", "2026-06-10 05:45:00", "light")

    summary = api_app.lighting_summary("cam1")

    assert summary["first_dark"] == "2026-06-09 21:30:00"
    assert summary["first_light"] == "2026-06-10 05:45:00"
    assert summary["light_seconds"] == 15.5 * 3600
    assert summary["dark_seconds"] == 8.25 * 3600
