from pt.api import app as api_app


def test_visible_devices_excludes_unreachable_configured_camera(monkeypatch):
    monkeypatch.setattr(api_app, "detect_connected_devices", lambda: ["phone-usb"])
    statuses = {
        "escam_online": {"configured": True, "reachable": True},
        "escam_offline": {"configured": True, "reachable": False},
    }

    assert api_app.visible_device_ids(statuses) == ["phone-usb", "escam_online"]


def test_visible_devices_probes_when_requested(monkeypatch):
    monkeypatch.setattr(api_app, "detect_connected_devices", lambda: [])
    calls = []

    def statuses(probe=False):
        calls.append(probe)
        return {"escam": {"configured": True, "reachable": False}}

    monkeypatch.setattr(api_app, "network_camera_statuses", statuses)

    assert api_app.visible_device_ids(probe_network=True) == []
    assert calls == [True]
