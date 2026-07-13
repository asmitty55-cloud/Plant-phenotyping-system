import json

import numpy as np

import pt.core.analysis.calibration_store as calibration_store
from pt.core.analysis.volumetric import _landmark_world_corners


def test_landmarks_are_keyed_by_dictionary_and_id(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    monkeypatch.setattr(calibration_store, "CALIBRATION_FILE", str(path))
    store = calibration_store.CalibrationStore()

    saved = store.upsert_landmark({
        "dictionary": "6X6_250",
        "id": 7,
        "size_mm": 52,
        "position_mm": [100, 200, 300],
        "rotation_deg": [0, 90, 0],
        "label": "east wall",
    })

    assert saved["label"] == "east wall"
    assert store.get_marker_size(7, dictionary="6X6_250") == 52
    assert store.get_marker_size(7, dictionary="4X4_50") != 52
    assert store.get_landmark("6X6_250", 7)["position_mm"] == [100.0, 200.0, 300.0]
    assert "6X6_250:7" in json.loads(path.read_text())["landmarks"]


def test_landmark_world_corners_preserve_physical_size():
    corners = _landmark_world_corners({
        "size_mm": 52,
        "position_mm": [10, 20, 30],
        "rotation_deg": [0, 0, 0],
    })

    assert np.allclose(corners.mean(axis=0), [10, 20, 30])
    assert np.isclose(np.linalg.norm(corners[1] - corners[0]), 52)
    assert np.isclose(np.linalg.norm(corners[2] - corners[1]), 52)
