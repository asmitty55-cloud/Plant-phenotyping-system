from pt.device import adb_transport


def test_normalize_endpoint_accepts_host_and_embedded_port():
    assert adb_transport.normalize_endpoint("192.168.137.42") == "192.168.137.42:5555"
    assert adb_transport.normalize_endpoint("phone.local:37123") == "phone.local:37123"


def test_normalize_endpoint_rejects_invalid_input():
    for value in ("", "phone name"):
        try:
            adb_transport.normalize_endpoint(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{value!r} should be rejected")


def test_adb_devices_includes_only_ready_devices(monkeypatch):
    output = (
        "List of devices attached\n"
        "USB123\tdevice product:x\n"
        "192.168.137.42:5555\tdevice product:y\n"
        "OFFLINE\toffline\n"
    )
    monkeypatch.setattr(adb_transport, "run_adb", lambda args: (output, "", 0))
    assert adb_transport.adb_devices() == ["USB123", "192.168.137.42:5555"]


def test_remember_endpoint_updates_existing_entry(monkeypatch):
    saved = []
    monkeypatch.setattr(
        adb_transport,
        "load_devices",
        lambda: [{"name": "old", "host": "192.168.137.42", "port": 5555, "enabled": False}],
    )
    monkeypatch.setattr(adb_transport, "save_local_devices", lambda devices: saved.extend(devices))

    endpoint = adb_transport.remember_endpoint("192.168.137.42:5555", name="top")

    assert endpoint == "192.168.137.42:5555"
    assert saved == [{
        "name": "top",
        "host": "192.168.137.42",
        "port": 5555,
        "enabled": True,
        "auto_connect": True,
    }]
