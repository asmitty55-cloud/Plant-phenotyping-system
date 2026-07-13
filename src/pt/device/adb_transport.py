import os
import re
import subprocess
import threading
import time

import yaml

from pt.core.utils.path_utils import get_data_root


ADB = os.getenv("PT_ADB", "adb")
CONFIG_DIR = os.path.join(get_data_root(), "configs")
CONFIG_PATH = os.path.join(CONFIG_DIR, "android_devices.yaml")
LOCAL_CONFIG_PATH = os.path.join(CONFIG_DIR, "android_devices.local.yaml")
DEFAULT_PORT = 5555
RECONNECT_INTERVAL_SECONDS = 30

_connect_lock = threading.Lock()
_last_reconnect = 0.0


def _subprocess_flags():
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def run_adb(args, timeout=25):
    try:
        result = subprocess.run(
            [ADB] + list(args),
            capture_output=True,
            text=True,
            creationflags=_subprocess_flags(),
            timeout=timeout,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except FileNotFoundError:
        return "", "adb-not-found", 127
    except subprocess.TimeoutExpired:
        return "", f"ADB timeout: {' '.join([ADB] + list(args))}", 124


def normalize_endpoint(host, port=DEFAULT_PORT):
    value = str(host or "").strip()
    if not value:
        raise ValueError("A phone IP or hostname is required.")
    if any(ch.isspace() for ch in value):
        raise ValueError("ADB endpoints cannot contain spaces.")
    if ":" in value:
        endpoint_host, endpoint_port = value.rsplit(":", 1)
        if endpoint_port.isdigit():
            value = endpoint_host
            port = int(endpoint_port)
    port = int(port or DEFAULT_PORT)
    if not 1 <= port <= 65535:
        raise ValueError("ADB port must be between 1 and 65535.")
    return f"{value}:{port}"


def _config_path():
    return LOCAL_CONFIG_PATH if os.path.exists(LOCAL_CONFIG_PATH) else CONFIG_PATH


def load_devices():
    path = _config_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    devices = data.get("android_devices") or []
    return [dict(device) for device in devices if isinstance(device, dict)]


def save_local_devices(devices):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(LOCAL_CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"android_devices": devices},
            handle,
            sort_keys=False,
            default_flow_style=False,
        )


def configured_endpoints():
    endpoints = []
    for device in load_devices():
        if not device.get("enabled", True):
            continue
        host = device.get("host") or device.get("ip")
        if not host:
            continue
        try:
            endpoints.append(normalize_endpoint(host, device.get("port", DEFAULT_PORT)))
        except (TypeError, ValueError):
            continue
    return endpoints


def adb_devices():
    out, err, _ = run_adb(["devices"])
    if err == "adb-not-found":
        return []
    return [
        line.split("\t", 1)[0]
        for line in out.splitlines()[1:]
        if "\tdevice" in line
    ]


def connect(endpoint):
    endpoint = normalize_endpoint(endpoint)
    out, err, code = run_adb(["connect", endpoint], timeout=12)
    message = out or err
    ok = code == 0 and ("connected to" in message.lower() or "already connected" in message.lower())
    return {"ok": ok, "endpoint": endpoint, "message": message}


def disconnect(endpoint):
    endpoint = normalize_endpoint(endpoint)
    out, err, code = run_adb(["disconnect", endpoint], timeout=12)
    return {"ok": code == 0, "endpoint": endpoint, "message": out or err}


def pair(endpoint, pairing_code):
    endpoint = normalize_endpoint(endpoint)
    code = str(pairing_code or "").strip()
    if not code:
        raise ValueError("A wireless-debugging pairing code is required.")
    out, err, returncode = run_adb(["pair", endpoint, code], timeout=20)
    message = out or err
    if "protocol fault" in message.lower():
        message = (
            f"{message}. Use the temporary pairing IP:port shown inside 'Pair device with pairing code', "
            "not the normal wireless-debugging connect port. If the code screen timed out, open a new code. "
            "If it still fails, toggle Wireless debugging off/on or restart the ADB server."
        )
    return {
        "ok": returncode == 0 and "successfully paired" in message.lower(),
        "endpoint": endpoint,
        "message": message,
    }


def prepare_legacy_wifi(usb_serial, port=DEFAULT_PORT):
    serial = str(usb_serial or "").strip()
    if not serial or ":" in serial:
        raise ValueError("Select a USB-connected Android serial for legacy setup.")
    port = int(port or DEFAULT_PORT)
    ip_out, ip_err, ip_code = run_adb(
        ["-s", serial, "shell", "ip", "-f", "inet", "addr", "show", "wlan0"],
        timeout=12,
    )
    match = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)/", ip_out)
    if ip_code != 0 or not match:
        return {
            "ok": False,
            "serial": serial,
            "message": ip_err or "Could not find the phone Wi-Fi address on wlan0.",
        }
    endpoint = normalize_endpoint(match.group(1), port)
    out, err, code = run_adb(["-s", serial, "tcpip", str(port)], timeout=15)
    if code != 0 or "restarting in tcp mode" not in (out or err).lower():
        return {"ok": False, "serial": serial, "endpoint": endpoint, "message": out or err}
    result = {"ok": False, "endpoint": endpoint, "message": "ADB TCP service did not become reachable."}
    for _ in range(4):
        time.sleep(1)
        result = connect(endpoint)
        if result["ok"]:
            break
    result["serial"] = serial
    return result


def remember_endpoint(endpoint, name=""):
    endpoint = normalize_endpoint(endpoint)
    host, port_text = endpoint.rsplit(":", 1)
    devices = load_devices()
    updated = False
    for device in devices:
        existing_host = device.get("host") or device.get("ip")
        if not existing_host:
            continue
        try:
            existing = normalize_endpoint(existing_host, device.get("port", DEFAULT_PORT))
        except (TypeError, ValueError):
            continue
        if existing == endpoint:
            device.update({"host": host, "port": int(port_text), "enabled": True, "auto_connect": True})
            if name:
                device["name"] = str(name).strip()
            updated = True
            break
    if not updated:
        device = {
            "name": str(name).strip() or endpoint,
            "host": host,
            "port": int(port_text),
            "enabled": True,
            "auto_connect": True,
        }
        devices.append(device)
    save_local_devices(devices)
    return endpoint


def forget_endpoint(endpoint):
    endpoint = normalize_endpoint(endpoint)
    kept = []
    for device in load_devices():
        host = device.get("host") or device.get("ip")
        try:
            current = normalize_endpoint(host, device.get("port", DEFAULT_PORT)) if host else ""
        except (TypeError, ValueError):
            current = ""
        if current != endpoint:
            kept.append(device)
    save_local_devices(kept)
    return endpoint


def reconnect_configured(force=False):
    global _last_reconnect
    now = time.monotonic()
    if not force and now - _last_reconnect < RECONNECT_INTERVAL_SECONDS:
        return []
    if not _connect_lock.acquire(blocking=False):
        return []
    try:
        _last_reconnect = now
        online = set(adb_devices())
        results = []
        for device in load_devices():
            if not device.get("enabled", True) or not device.get("auto_connect", True):
                continue
            host = device.get("host") or device.get("ip")
            if not host:
                continue
            try:
                endpoint = normalize_endpoint(host, device.get("port", DEFAULT_PORT))
            except (TypeError, ValueError) as exc:
                results.append({"ok": False, "endpoint": str(host), "message": str(exc)})
                continue
            if endpoint in online:
                results.append({"ok": True, "endpoint": endpoint, "message": "already connected"})
            else:
                results.append(connect(endpoint))
        return results
    finally:
        _connect_lock.release()


def reconnect_configured_async():
    if time.monotonic() - _last_reconnect < RECONNECT_INTERVAL_SECONDS or _connect_lock.locked():
        return
    threading.Thread(target=reconnect_configured, daemon=True).start()


def transport_status():
    online = set(adb_devices())
    configured = []
    for device in load_devices():
        host = device.get("host") or device.get("ip")
        endpoint = None
        error = ""
        if host:
            try:
                endpoint = normalize_endpoint(host, device.get("port", DEFAULT_PORT))
            except (TypeError, ValueError) as exc:
                error = str(exc)
        configured.append({
            **device,
            "endpoint": endpoint,
            "online": bool(endpoint and endpoint in online),
            "error": error,
        })
    return {
        "adb_available": run_adb(["version"], timeout=5)[2] == 0,
        "online": sorted(online),
        "usb": sorted(device for device in online if ":" not in device),
        "wifi": sorted(device for device in online if ":" in device),
        "configured": configured,
        "config_path": LOCAL_CONFIG_PATH,
    }
