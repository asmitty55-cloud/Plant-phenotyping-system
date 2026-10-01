import importlib
from datetime import datetime, timezone


def test_capture_filename_timestamp_is_converted_to_utc():
    from pt.core.analysis.image_analysis import _capture_time_utc

    parsed = _capture_time_utc("capture_20260929_120000.jpg")

    assert parsed is not None
    assert datetime.fromisoformat(parsed).tzinfo == timezone.utc


def test_metric_provenance_traces_to_capture_calibration_and_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("PT_DATA_ROOT", str(tmp_path))
    import pt.core.analysis.metric_store as metric_store

    metric_store = importlib.reload(metric_store)
    image_path = tmp_path / "capture_20260929_120000.jpg"
    image_path.write_bytes(b"test-image-content")
    capture_set_id = metric_store.create_capture_set(
        "CS_RESEARCH_001",
        experiment_id="Wheat_Drought_Trial_2026",
        captured_at_utc="2026-09-29T12:00:00+00:00",
        source="synchronized_android",
    )
    metric_store.register_capture_image(
        capture_set_id,
        "Pixel6_A",
        image_path.name,
        str(image_path),
        captured_at_utc="2026-09-29T12:00:00+00:00",
        camera_metadata={"iso": "200", "white_balance": "daylight"},
    )
    calibration = {"camera_params": {"Pixel6_A": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}}}
    calibration_id = metric_store.create_calibration_snapshot(calibration)
    analysis_parameters = {"segmentation": {"revision": "hsv-exg-v1", "kernel_px": 5}}
    run_id = metric_store.create_analysis_run(
        capture_set_id,
        calibration_id,
        "0.1.0",
        analysis_parameters,
    )
    metric_store.upsert_history_point("Pixel6_A", {
        "timestamp": "2026-09-29 12:00:00",
        "filename": image_path.name,
        "area": 24580.0,
        "scale": 2.0,
        "canopy_coverage": 0.821,
        "capture_set_id": capture_set_id,
        "analysis_run_id": run_id,
    })
    metric_store.upsert_plant_metric_points(
        "Pixel6_A",
        "2026-09-29 12:00:00",
        image_path.name,
        [{
            "plant_id": "P17",
            "tray_id": "TRAY_1",
            "cell_id": "C17",
            "area_mm2": 24580.0,
            "coverage": 0.821,
            "status": "germinated",
        }],
        capture_set_id=capture_set_id,
        analysis_run_id=run_id,
    )

    provenance = metric_store.provenance_for_image("Pixel6_A", image_path.name)

    assert provenance["experiment_id"] == "Wheat_Drought_Trial_2026"
    assert provenance["capture_set_id"] == "CS_RESEARCH_001"
    assert provenance["image_sha256"]
    assert provenance["area"] == 24580.0
    assert provenance["calibration_snapshot_id"] == calibration_id
    assert provenance["calibration_snapshot"] == calibration
    assert provenance["pipeline_version"] == "0.1.0"
    assert provenance["parameters"] == analysis_parameters
    assert len(provenance["parameter_hash"]) == 64
    assert provenance["camera_metadata"]["iso"] == "200"
    assert provenance["plants"] == [{
        "plant_id": "P17",
        "tray_id": "TRAY_1",
        "cell_id": "C17",
        "area": 24580.0,
        "coverage": 0.821,
        "status": "germinated",
    }]


def test_calibration_snapshots_are_content_addressed(tmp_path, monkeypatch):
    monkeypatch.setenv("PT_DATA_ROOT", str(tmp_path))
    import pt.core.analysis.metric_store as metric_store

    metric_store = importlib.reload(metric_store)

    original = {"marker_size_mm": 41.18}
    original_id = metric_store.create_calibration_snapshot(original)
    assert metric_store.create_calibration_snapshot(original) == original_id
    assert metric_store.create_calibration_snapshot({"marker_size_mm": 42.0}) != original_id


def test_latest_capture_processing_persists_metric_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("PT_DATA_ROOT", str(tmp_path))
    import pt.core.analysis.image_analysis as image_analysis
    import pt.core.analysis.metric_store as metric_store

    metric_store = importlib.reload(metric_store)
    capture_dir = tmp_path / "captures" / "Pixel6_A"
    capture_dir.mkdir(parents=True)
    filename = "capture_20260929_120000.jpg"
    (capture_dir / filename).write_bytes(b"captured-image")
    monkeypatch.setattr(image_analysis, "analyze_image", lambda *args, **kwargs: {
        "plant_area_mm2": 24580.0,
        "scale_px_per_mm": 2.0,
        "segments": [],
        "tray_cells": [{
            "plant_id": "P17",
            "tray_id": "TRAY_1",
            "cell_id": "C17",
            "area_mm2": 24580.0,
            "coverage": 0.821,
            "canopy_pixels": 98320,
            "delta_pixels": 100,
            "status": "germinated",
        }],
        "canopy_coverage": 0.821,
        "color_metrics": {"green_index": 0.4},
        "color_correction": {},
        "canopy_bounding_box": {"center": [10.0, 20.0]},
        "markers_found": 4,
        "method": "Green",
        "dictionary": "4X4_50",
        "mask": None,
    })

    image_analysis.process_latest_captures(
        str(tmp_path / "captures"),
        capture_metadata_by_device={"Pixel6_A": {
            "source": "test_capture",
            "experiment_id": "Wheat_Drought_Trial_2026",
            "camera_settings": {"iso": "200"},
        }},
    )

    provenance = metric_store.provenance_for_image("Pixel6_A", filename)
    assert provenance["experiment_id"] == "Wheat_Drought_Trial_2026"
    assert provenance["source"] == "test_capture"
    assert provenance["analysis_run_id"]
    assert provenance["calibration_snapshot_id"]
    assert provenance["pipeline_version"] == "0.1.0"
    assert set(provenance["enabled_modules"]) == {"canopy", "growth", "circadian", "color_index", "volumetric"}
    assert provenance["module_versions"]["circadian"] == "1.0.0"
    assert provenance["area"] == 24580.0
    assert provenance["plants"][0]["plant_id"] == "P17"

    image_analysis.process_latest_captures(str(tmp_path / "captures"))
    repeated_provenance = metric_store.provenance_for_image("Pixel6_A", filename)
    assert repeated_provenance["capture_set_id"] == provenance["capture_set_id"]
    assert repeated_provenance["camera_metadata"] == {"iso": "200"}
    assert repeated_provenance["analysis_run_id"] == provenance["analysis_run_id"]

    import pt.api.app as api_app

    response = api_app.app.test_client().get(f"/api/provenance/Pixel6_A/{filename}")
    assert response.status_code == 200
    assert response.get_json()["experiment_id"] == "Wheat_Drought_Trial_2026"
    assert response.get_json()["analysis_run_id"] == provenance["analysis_run_id"]