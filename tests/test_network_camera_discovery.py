import ipaddress

import pytest
import yaml

from pt.device import network_camera as network_camera
from pt.device import network_camera_discovery as discovery


CAMERA_MAC = "28:f3:66:79:bd:24"


def test_normalize_mac_accepts_common_separators():
    assert discovery.normalize_mac("28-F3-66-79-BD-24") == CAMERA_MAC
    assert discovery.normalize_mac("28f36679bd24") == CAMERA_MAC


def test_normalize_mac_rejects_invalid_value():
    for value in ("not-a-mac", "28:f3:66:79:bd:24xyz"):
        with pytest.raises(ValueError):
            discovery.normalize_mac(value)


def test_network_discovery_requires_unique_rtsp_mac_match(monkeypatch):
    network = ipaddress.ip_network("192.168.137.0/30")
    monkeypatch.setattr(discovery, "scan_host", lambda host: {
        "host": host,
        "open_ports": [554] if host in ("192.168.137.1", "192.168.137.2") else [],
    })
    monkeypatch.setattr(discovery, "read_neighbor_macs", lambda: {
        "192.168.137.1": CAMERA_MAC,
        "192.168.137.2": "dc:74:a8:4e:c3:2e",
    })
    monkeypatch.setattr(discovery, "is_rtsp_server", lambda _host, _port: True)

    result = discovery.discover_camera(CAMERA_MAC, networks=[network])

    assert result["status"] == "matched"
    assert result["match"]["host"] == "192.168.137.1"
    assert len(result["candidates"]) == 2


def test_network_discovery_does_not_match_open_port_without_rtsp(monkeypatch):
    network = ipaddress.ip_network("192.168.137.0/30")
    monkeypatch.setattr(discovery, "scan_host", lambda host: {
        "host": host,
        "open_ports": [554] if host == "192.168.137.1" else [],
    })
    monkeypatch.setattr(discovery, "read_neighbor_macs", lambda: {"192.168.137.1": CAMERA_MAC})
    monkeypatch.setattr(discovery, "is_rtsp_server", lambda _host, _port: False)

    result = discovery.discover_camera(CAMERA_MAC, networks=[network])

    assert result["status"] == "not_found"
    assert result["match"] is None


def test_network_discovery_leaves_ambiguous_mac_unchanged(monkeypatch):
    network = ipaddress.ip_network("192.168.137.0/30")
    monkeypatch.setattr(discovery, "scan_host", lambda host: {"host": host, "open_ports": [554]})
    monkeypatch.setattr(discovery, "read_neighbor_macs", lambda: {
        "192.168.137.1": CAMERA_MAC,
        "192.168.137.2": CAMERA_MAC,
    })
    monkeypatch.setattr(discovery, "is_rtsp_server", lambda _host, _port: True)

    result = discovery.discover_camera(CAMERA_MAC, networks=[network])

    assert result["status"] == "ambiguous"
    assert result["match"] is None


def test_discovered_host_overlay_preserves_local_yaml_and_urls(monkeypatch, tmp_path):
    config_text = (
        "# keep local comments and credentials\n"
        "network_cameras:\n"
        "  - id: escam_penguin_qf521\n"
        "    enabled: true\n"
        "    host: 192.168.137.146\n"
        "    username: admin\n"
        "    password: local-test-only\n"
        "    rtsp_port: 554\n"
        "    stream_url: rtsp://admin:local-test-only@192.168.137.146:554/onvif1\n"
        "    onvif_url: http://192.168.137.146:5000/onvif/device_service\n"
        "    paths: [onvif1, onvif2]\n"
    )
    local_config = tmp_path / "network_cameras.local.yaml"
    local_config.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(network_camera, "LOCAL_CONFIG_PATH", str(local_config))
    monkeypatch.setattr(network_camera, "CONFIG_PATH", str(tmp_path / "sample.yaml"))
    monkeypatch.setattr(network_camera, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(network_camera, "DISCOVERY_STATE_PATH", str(tmp_path / "network_camera_discovery.json"))

    network_camera.set_network_camera_mac("escam_penguin_qf521", CAMERA_MAC)
    network_camera.update_discovered_network_camera("escam_penguin_qf521", {
        "host": "192.168.137.77",
        "mac_address": CAMERA_MAC,
        "rtsp_port": 554,
    })
    camera = network_camera.camera_by_id("escam_penguin_qf521")

    assert local_config.read_text(encoding="utf-8") == config_text
    assert camera["host"] == "192.168.137.77"
    assert camera["stream_url"] == "rtsp://admin:local-test-only@192.168.137.77:554/onvif1"
    assert camera["onvif_url"] == "http://192.168.137.77:5000/onvif/device_service"
    assert camera["paths"] == ["onvif1", "onvif2"]
    assert yaml.safe_load(config_text)["network_cameras"][0]["password"] == "local-test-only"
    assert "local-test-only" not in (tmp_path / "network_camera_discovery.json").read_text(encoding="utf-8")


def test_discovered_host_rejects_unpaired_mac(monkeypatch, tmp_path):
    local_config = tmp_path / "network_cameras.local.yaml"
    local_config.write_text(
        "network_cameras:\n  - id: escam\n    enabled: true\n    host: 192.168.137.10\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(network_camera, "LOCAL_CONFIG_PATH", str(local_config))
    monkeypatch.setattr(network_camera, "CONFIG_PATH", str(tmp_path / "sample.yaml"))
    monkeypatch.setattr(network_camera, "CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(network_camera, "DISCOVERY_STATE_PATH", str(tmp_path / "network_camera_discovery.json"))

    network_camera.set_network_camera_mac("escam", CAMERA_MAC)
    with pytest.raises(ValueError, match="does not match"):
        network_camera.update_discovered_network_camera("escam", {
            "host": "192.168.137.77",
            "mac_address": "dc:74:a8:4e:c3:2e",
            "rtsp_port": 554,
        })


def test_camera_identity_route_requires_same_origin_request():
    from pt.api import app as api_app

    response = api_app.app.test_client().post(
        "/network_camera_discovery/escam/identify",
        json={"mac_address": CAMERA_MAC},
    )

    assert response.status_code == 403


def test_offline_paired_camera_schedules_auto_scan_only_when_probed(monkeypatch):
    from pt.api import app as api_app

    calls = []
    monkeypatch.setattr(api_app, "configured_camera_ids", lambda: ["escam"])
    monkeypatch.setattr(api_app, "network_camera_status", lambda _camera_id, probe=False: {
        "configured": True,
        "reachable": False,
        "mac_address": CAMERA_MAC,
    })
    monkeypatch.setattr(
        api_app,
        "start_network_camera_discovery",
        lambda camera_id, automatic=False: calls.append((camera_id, automatic)),
    )

    api_app.network_camera_statuses(probe=False)
    assert calls == []

    api_app.network_camera_statuses(probe=True)
    assert calls == [("escam", True)]


def test_host_reuse_with_different_mac_schedules_rediscovery(monkeypatch):
    from pt.api import app as api_app

    calls = []
    monkeypatch.setattr(api_app, "configured_camera_ids", lambda: ["escam"])
    monkeypatch.setattr(api_app, "network_camera_statuses", lambda probe=False: {
        "escam": {
            "reachable": True,
            "host": "192.168.137.77",
            "mac_address": CAMERA_MAC,
        },
    })
    monkeypatch.setattr(api_app, "read_neighbor_macs", lambda: {
        "192.168.137.77": "dc:74:a8:4e:c3:2e",
    })
    monkeypatch.setattr(
        api_app,
        "start_network_camera_discovery",
        lambda camera_id, automatic=False: calls.append((camera_id, automatic)),
    )
    monkeypatch.setattr(api_app.time, "sleep", lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        api_app.network_camera_discovery_monitor()

    assert calls == [("escam", True)]
