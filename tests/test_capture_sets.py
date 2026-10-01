import importlib

import numpy as np


def _reload_metric_store(tmp_path, monkeypatch):
    monkeypatch.setenv("PT_DATA_ROOT", str(tmp_path))
    import pt.core.analysis.metric_store as metric_store

    return importlib.reload(metric_store)


def test_metric_store_reinitializes_when_data_root_changes(tmp_path, monkeypatch):
    metric_store = _reload_metric_store(tmp_path, monkeypatch)
    first_path = metric_store.db_path()
    metric_store.ensure_db()
    alternate_root = tmp_path / "alternate-root"
    monkeypatch.setattr(metric_store, "get_data_root", lambda: str(alternate_root))

    with metric_store.connect() as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    assert metric_store.db_path() != first_path
    assert "metric_history" in tables
    assert "capture_sets" in tables


def test_system_manifest_reports_operational_configuration(tmp_path, monkeypatch):
    metric_store = _reload_metric_store(tmp_path, monkeypatch)
    import pt.api.app as api_app

    monkeypatch.setattr(api_app, "metric_store", metric_store)
    monkeypatch.setattr(api_app, "load_experiments", lambda: [
        {"id": "trial_active", "name": "Active trial", "status": "active", "devices": ["camera_a"]},
        {"id": "trial_done", "name": "Finished trial", "status": "complete", "devices": []},
    ])
    monkeypatch.setattr(api_app.calib_store, "data", {
        "camera_params": {"camera_a": {"K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
        "landmarks": {"6X6_250:1": {"id": 1}},
    })

    response = api_app.app.test_client().get("/api/system/manifest")

    assert response.status_code == 200
    manifest = response.get_json()
    assert manifest["pipeline_version"] == "0.1.0"
    assert manifest["schema_version"] == "plant_metrics.v1"
    assert {module["name"] for module in manifest["analysis_modules"]} == {
        "canopy", "growth", "circadian", "color_index", "volumetric",
    }
    assert all(module["enabled"] for module in manifest["analysis_modules"])
    assert manifest["active_experiments"] == [{
        "id": "trial_active", "name": "Active trial", "devices": ["camera_a"],
    }]
    assert manifest["calibrations"]["camera_devices"] == 1
    assert manifest["calibrations"]["landmarks"] == 1


def test_capture_set_summary_contains_qc_images_runs_and_reconstruction(tmp_path, monkeypatch):
    metric_store = _reload_metric_store(tmp_path, monkeypatch)
    image_path = tmp_path / "camera_a.jpg"
    image_path.write_bytes(b"camera-a")
    capture_set_id = metric_store.create_capture_set(
        "CS_OBS_001",
        experiment_id="trial_1",
        captured_at_utc="2026-09-30T10:00:00+00:00",
        source="synchronized_android",
    )
    metric_store.register_capture_image(capture_set_id, "camera_a", "camera_a.jpg", str(image_path))
    calibration_id = metric_store.create_calibration_snapshot({"camera_a": "calibration"})
    run_id = metric_store.create_analysis_run(capture_set_id, calibration_id, "1.0", {"algorithm": "v1"})
    alternate_run_id = metric_store.create_analysis_run(
        capture_set_id,
        calibration_id,
        "1.0",
        {"algorithm": "v1"},
        enabled_modules=["canopy"],
        module_versions={"canopy": "1.0.0"},
    )
    assert alternate_run_id != run_id
    assert metric_store.analysis_run_detail(alternate_run_id)["parameter_hash"] != metric_store.analysis_run_detail(run_id)["parameter_hash"]
    metric_store.upsert_history_point("camera_a", {
        "timestamp": "2026-09-30 10:00:00",
        "filename": "camera_a.jpg",
        "area": 1000.0,
        "capture_set_id": capture_set_id,
        "analysis_run_id": run_id,
    })
    assert metric_store.update_capture_set_qc(
        capture_set_id,
        device_count=2,
        successful_devices=["camera_a"],
        failed_devices=["camera_b"],
        sync_quality_ms=12.0,
        capture_duration_ms=321.5,
    ) == "marginal"
    metric_store.record_capture_set_reconstruction(capture_set_id, "complete", {"volume_mm3": 12.5})

    summary = metric_store.list_capture_sets(limit=10)[0]
    detail = metric_store.capture_set_detail(capture_set_id)

    assert summary["device_count"] == 2
    assert summary["successful_devices"] == ["camera_a"]
    assert summary["failed_devices"] == ["camera_b"]
    assert summary["image_count"] == 1
    assert summary["capture_duration_ms"] == 321.5
    assert summary["analysis_run_count"] == 2
    assert summary["qc_status"] == "marginal"
    assert detail["images"][0]["analysis_run_id"] == run_id
    assert detail["analysis_runs"][0]["calibration_snapshot_id"] == calibration_id
    assert detail["analysis_runs"][0]["calibration_snapshot"] == {"camera_a": "calibration"}
    assert detail["reconstruction"]["result"]["volume_mm3"] == 12.5

    import pt.api.app as api_app

    monkeypatch.setattr(api_app, "metric_store", metric_store)
    run_response = api_app.app.test_client().get(f"/api/analysis-runs/{run_id}")
    assert run_response.status_code == 200
    run_record = run_response.get_json()
    assert run_record["capture_set_id"] == capture_set_id
    assert run_record["experiment_id"] == "trial_1"
    assert run_record["enabled_modules"] == []
    assert run_record["calibration_hash"]


def test_previous_metric_selection_stays_within_experiment(tmp_path, monkeypatch):
    metric_store = _reload_metric_store(tmp_path, monkeypatch)
    records = [
        ("CS_OLD", "trial_a", "2026-09-30T08:00:00+00:00", "old.jpg", 100.0),
        ("CS_OTHER", "trial_b", "2026-09-30T09:30:00+00:00", "other.jpg", 900.0),
        ("CS_CURRENT", "trial_a", "2026-09-30T10:00:00+00:00", "current.jpg", 0.0),
    ]
    for capture_set_id, experiment_id, captured_at, filename, area in records:
        metric_store.create_capture_set(capture_set_id, experiment_id, captured_at, source="test")
        image_path = tmp_path / filename
        image_path.write_bytes(filename.encode())
        metric_store.register_capture_image(capture_set_id, "camera_a", filename, str(image_path))
        if area:
            metric_store.upsert_history_point("camera_a", {
                "timestamp": captured_at,
                "filename": filename,
                "area": area,
                "capture_set_id": capture_set_id,
            })

    previous = metric_store.previous_metric_in_experiment("CS_CURRENT", "camera_a")

    assert previous["capture_set_id"] == "CS_OLD"
    assert previous["area"] == 100.0


def test_growth_uses_prior_capture_set_in_same_experiment(tmp_path, monkeypatch):
    metric_store = _reload_metric_store(tmp_path, monkeypatch)
    import pt.core.analysis.image_analysis as image_analysis

    captures_dir = tmp_path / "captures"
    device_dir = captures_dir / "camera_a"
    device_dir.mkdir(parents=True)
    first_filename = "capture_20260930_080000.jpg"
    (device_dir / first_filename).write_bytes(b"first observation")
    monkeypatch.setattr(image_analysis, "analyze_image", lambda path, **kwargs: {
        "plant_area_mm2": 100.0 if "080000" in path else 200.0,
        "scale_px_per_mm": 2.0,
        "segments": [],
        "tray_cells": [],
        "canopy_coverage": 0.5,
        "color_metrics": {},
        "color_correction": {},
        "canopy_bounding_box": {"center": [5.0, 5.0]},
        "markers_found": 4,
        "method": "Green",
        "dictionary": "4X4_50",
        "mask": None,
    })

    image_analysis.process_latest_captures(
        str(captures_dir),
        capture_set_ids={"camera_a": "CS_TRIAL_FIRST"},
        capture_metadata_by_device={"camera_a": {
            "source": "test",
            "experiment_id": "trial_a",
            "captured_at_utc": "2026-09-30T08:00:00+00:00",
        }},
    )
    second_filename = "capture_20260930_100000.jpg"
    (device_dir / second_filename).write_bytes(b"second observation")
    image_analysis.process_latest_captures(
        str(captures_dir),
        capture_set_ids={"camera_a": "CS_TRIAL_SECOND"},
        capture_metadata_by_device={"camera_a": {
            "source": "test",
            "experiment_id": "trial_a",
            "captured_at_utc": "2026-09-30T10:00:00+00:00",
        }},
    )

    history = metric_store.history_for_device("camera_a")
    latest = next(row for row in history if row["filename"] == second_filename)

    assert latest["growth_rate_mm2_hr"] == 50.0
    assert latest["capture_set_id"] == "CS_TRIAL_SECOND"
    assert latest["previous_capture_set_id"] == "CS_TRIAL_FIRST"


def test_reconstruction_uses_only_images_from_requested_capture_set(tmp_path, monkeypatch):
    metric_store = _reload_metric_store(tmp_path, monkeypatch)
    selected_paths = []
    selected_devices = []
    for capture_set_id, devices in (("CS_SELECTED", ["camera_a", "camera_b"]), ("CS_NEWER", ["camera_c"])):
        metric_store.create_capture_set(capture_set_id, source="test")
        for device_id in devices:
            filename = f"{capture_set_id}_{device_id}.jpg"
            image_path = tmp_path / filename
            image_path.write_bytes(filename.encode())
            metric_store.register_capture_image(capture_set_id, device_id, filename, str(image_path))
            if capture_set_id == "CS_SELECTED":
                selected_paths.append(str(image_path))
                selected_devices.append(device_id)

    import pt.api.app as api_app
    import pt.core.analysis.image_analysis as image_analysis

    monkeypatch.setattr(api_app, "metric_store", metric_store)
    monkeypatch.setattr(api_app, "DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(api_app, "network_camera_statuses", lambda probe=False: {})
    monkeypatch.setattr(api_app, "visible_device_ids", lambda statuses=None, probe_network=False: [])
    monkeypatch.setattr(image_analysis, "analyze_image", lambda path, device_id=None: {"mask": np.ones((3, 3), dtype=np.uint8)})
    reconstructed_devices = []

    def fake_reconstruct(device_masks):
        reconstructed_devices.extend(sorted(device_masks))
        return {"status": "ok", "volume_mm3": 10.0, "cameras_used": sorted(device_masks)}

    monkeypatch.setattr(api_app, "reconstruct_visual_hull", fake_reconstruct)
    response = api_app.app.test_client().post("/api/capture-sets/CS_SELECTED/reconstruct")

    assert response.status_code == 200
    assert reconstructed_devices == sorted(selected_devices)
    assert set(selected_paths) == {image["image_path"] for image in metric_store.capture_set_detail("CS_SELECTED")["images"]}
    assert response.get_json()["capture_set_id"] == "CS_SELECTED"
    reconstruction_run_id = response.get_json()["analysis_run_id"]
    reconstruction_run = api_app.app.test_client().get(
        f"/api/analysis-runs/{reconstruction_run_id}"
    ).get_json()
    assert reconstruction_run["capture_set_id"] == "CS_SELECTED"
    assert reconstruction_run["parameters"]["analysis_revision"] == "visual-hull.v1"
    assert "volumetric" in reconstruction_run["enabled_modules"]
    dashboard = api_app.app.test_client().get("/")
    assert dashboard.status_code == 200
    assert b"Capture Sets" in dashboard.data
    assert b"capture-set-list" in dashboard.data
    reconstructed_devices.clear()
    latest_set_response = api_app.app.test_client().get("/reconstruct")
    assert latest_set_response.status_code == 200
    assert latest_set_response.get_json()["capture_set_id"] == "CS_SELECTED"
    assert reconstructed_devices == sorted(selected_devices)


def test_offline_synchronized_capture_is_recorded_as_invalid(tmp_path, monkeypatch):
    metric_store = _reload_metric_store(tmp_path, monkeypatch)
    import pt.api.app as api_app

    monkeypatch.setattr(api_app, "metric_store", metric_store)
    monkeypatch.setattr(api_app, "detect_connected_devices", lambda: ["camera_a"])

    response = api_app.app.test_client().post(
        "/volumetric/capture",
        json={"devices": ["camera_a", "camera_b"], "experiment_id": "trial_1"},
    )

    assert response.status_code == 502
    capture_set_id = response.get_json()["shot_id"]
    record = metric_store.capture_set_detail(capture_set_id)
    assert record["experiment_id"] == "trial_1"
    assert record["device_count"] == 2
    assert record["failed_devices"] == ["camera_a", "camera_b"]
    assert record["qc_status"] == "invalid"