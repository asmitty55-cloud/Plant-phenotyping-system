import numpy as np

from pt.core.analysis.modules import AnalysisObservation
from pt.core.analysis.registry import ANALYSIS_REGISTRY


def test_disabling_circadian_keeps_other_analysis_modules_runnable():
    observation = AnalysisObservation(
        device_id="camera-a",
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        history=[],
        previous={"area": 10.0, "timestamp": "2026-09-30 09:00:00", "movement": {"centroid_px": [2, 3]}},
        metrics={
            "plant_area_mm2": 20.0,
            "scale_px_per_mm": 1.0,
            "canopy_bounding_box": {"center": [3, 5]},
            "color_metrics": {"green_index": 0.5},
        },
        context={
            "now_timestamp": 1790762700.0,
            "canopy_calculator": lambda *args, **kwargs: {"canopy_area_mm2": 20.0, "mask": np.ones((2, 2))},
            "reconstruct": lambda masks: {"status": "ok", "volume_mm3": 1.0},
            "device_masks": {"camera-a": np.ones((2, 2))},
        },
    )
    ANALYSIS_REGISTRY.disable("circadian")
    try:
        results = ANALYSIS_REGISTRY.run(observation)
    finally:
        ANALYSIS_REGISTRY.enable("circadian")

    assert set(results) == {"canopy", "growth", "color_index", "volumetric"}
    assert "growth_rate_mm2_hr" in observation.metrics
    assert "baseline" in observation.metrics
    assert "canopy" in observation.metrics
    assert "volumetric" in observation.metrics
    assert "circadian" not in results


def test_module_registry_rejects_duplicate_names_and_unknown_toggles():
    class EmptyModule:
        name = "empty"

        def analyze(self, observation):
            from pt.core.analysis.modules import AnalysisResult
            return AnalysisResult()

    from pt.core.analysis.modules import AnalysisRegistry

    registry = AnalysisRegistry([EmptyModule()])
    try:
        registry.register(EmptyModule())
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate module registration should fail")
    try:
        registry.disable("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("disabling an unknown module should fail")


def test_biology_state_keeps_color_and_growth_alerts_when_circadian_disabled(monkeypatch):
    import pt.api.app as api_app

    history = []
    for index in range(60):
        day = 28 + index // 24
        history.append({
            "timestamp": f"2026-09-{day:02d} {index % 24:02d}:00:00",
            "color_metrics": {"green_index": 1.0 if index < 48 else 0.7},
            "movement": {
                "centroid_px": [0.0, 20.0 if index == 59 else 0.0],
                "speed_mm_hr": 1.0,
            },
        })
    history[-1].update({
        "filename": "capture_20260930_230000.jpg",
        "analysis_run_id": "AR_ALERT_1",
        "capture_set_id": "CS_ALERT_1",
    })
    monkeypatch.setattr(api_app, "load_dashboard_stats", lambda: {"camera-a": {"history": history}})
    monkeypatch.setattr(api_app.metric_store, "provenance_for_image", lambda device_id, filename: {
        "experiment_id": "trial_wheat",
        "plants": [{"plant_id": "P17"}],
    })
    ANALYSIS_REGISTRY.disable("circadian")
    try:
        biology = api_app._biology_state()
    finally:
        ANALYSIS_REGISTRY.enable("circadian")

    alert_types = {alert["type"] for alert in biology["alerts"]}
    assert biology["rhythms"] == {}
    assert "green_index_drop" in alert_types
    assert "droop" in alert_types
    assert [alert["type"] for alert in biology["alerts"]] == ["green_index_drop", "droop"]
    green_alert, droop_alert = biology["alerts"]
    assert green_alert["trigger_module"] == "color_index"
    assert green_alert["metric_basis"]["name"] == "green_index"
    assert green_alert["threshold_basis"]["baseline_multiplier"] == 0.85
    assert droop_alert["trigger_module"] == "growth"
    assert droop_alert["analysis_run_id"] == "AR_ALERT_1"
    assert droop_alert["capture_set_id"] == "CS_ALERT_1"
    assert droop_alert["experiment_id"] == "trial_wheat"
    assert droop_alert["plant_ids"] == ["P17"]
