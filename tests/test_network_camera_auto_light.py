import cv2
import numpy as np

from pt.api import app as api_app


def test_visible_devices_exclude_configured_network_camera_when_offline(monkeypatch):
    monkeypatch.setattr(api_app, "detect_connected_devices", lambda: ["phone1"])

    devices = api_app.visible_device_ids(statuses={"escam1": {"configured": True, "reachable": False}})

    assert devices == ["phone1"]


def test_network_auto_light_uses_fresh_dark_probe_before_skipping(monkeypatch, tmp_path):
    capture_path = tmp_path / "capture_20260609_120000.jpg"
    cv2.imwrite(str(capture_path), np.zeros((24, 24, 3), dtype=np.uint8))
    events = []

    monkeypatch.setattr(api_app, "network_camera_status", lambda camera_id, probe=False: {"reachable": True})
    monkeypatch.setattr(
        api_app,
        "effective_capture_settings",
        lambda camera_id: {
            "light_mode": "auto",
            "active_light_mode": "day",
            "latest_luminance": 120.0,
            "collect_night_frames": False,
        },
    )
    monkeypatch.setattr(api_app, "capture_network_camera", lambda camera_id, filename: str(capture_path))
    monkeypatch.setattr(api_app, "record_light_transition_if_changed", lambda *args, **kwargs: events.append((args, kwargs)))
    monkeypatch.setattr(api_app, "start_night_skip", lambda camera_id, timestamp: events.append(("skip", camera_id, timestamp)))
    monkeypatch.setattr(api_app, "finish_night_skip", lambda *args, **kwargs: events.append(("finish", args, kwargs)))
    monkeypatch.setattr(api_app, "process_latest_captures", lambda *_args, **_kwargs: events.append("analyzed"))
    monkeypatch.setattr(api_app, "assemble_video", lambda *_args, **_kwargs: events.append("video"))
    assert api_app.capture_network_and_analyze("escam1") is True

    assert not capture_path.exists()
    assert events[0][0][2] == "night_ir"
    assert events[0][1]["luma"] == 0.0
    assert any(event[0] == "skip" and event[1] == "escam1" for event in events if isinstance(event, tuple))
    assert "analyzed" not in events
