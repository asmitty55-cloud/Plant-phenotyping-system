import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from urllib.parse import quote

import cv2
import numpy as np
import yaml

from pt.core.utils.path_utils import get_captures_dir, get_data_root


CONFIG_DIR = os.path.join(get_data_root(), "configs")
CONFIG_PATH = os.path.join(CONFIG_DIR, "network_cameras.yaml")
LOCAL_CONFIG_PATH = os.path.join(CONFIG_DIR, "network_cameras.local.yaml")
DEFAULT_RTSP_PATHS = ("onvif1", "onvif2")
CAMERA_STATUS = {}


def hidden_subprocess_flags():
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def load_network_cameras():
    config_path = LOCAL_CONFIG_PATH if os.path.exists(LOCAL_CONFIG_PATH) else CONFIG_PATH
    if not os.path.exists(config_path):
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cameras = data.get("network_cameras", [])
    return [camera for camera in cameras if camera.get("enabled", True)]


def configured_camera_ids():
    return [camera["id"] for camera in load_network_cameras() if camera.get("id")]


def camera_by_id(camera_id):
    for camera in load_network_cameras():
        if camera.get("id") == camera_id:
            return camera
    return None


def configured_camera(camera_id):
    return camera_by_id(camera_id)


def rtsp_urls(camera):
    if camera.get("stream_url"):
        return [camera["stream_url"]]

    host = camera.get("host")
    if not host:
        return []

    username = camera.get("username", "admin")
    password = camera.get("password", "")
    port = int(camera.get("rtsp_port", 554))
    paths = camera.get("paths") or list(DEFAULT_RTSP_PATHS)
    auth = quote(username, safe="")
    if password:
        auth += ":" + quote(str(password), safe="")
    return [f"rtsp://{auth}@{host}:{port}/{path}" for path in paths]


def live_stream_urls(camera):
    """Return stream URLs that are suitable for diagnostic live preview."""
    explicit_url = camera.get("live_stream_url") or camera.get("mjpeg_url")
    if explicit_url:
        return [explicit_url]
    return rtsp_urls(camera)


def camera_has_live_stream(camera_id):
    camera = camera_by_id(camera_id)
    return bool(camera and live_stream_urls(camera))


def camera_reachable(camera, timeout=1.0):
    host = camera.get("host")
    port = int(camera.get("rtsp_port", 554))
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def network_camera_status(camera_id, probe=False):
    camera = camera_by_id(camera_id)
    if not camera:
        return {"configured": False, "reachable": False, "message": "Unknown camera"}
    cached = dict(CAMERA_STATUS.get(camera_id, {}))
    if probe or "reachable" not in cached:
        cached["reachable"] = camera_reachable(camera)
        cached["checked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cached.update({
        "configured": True,
        "id": camera_id,
        "name": camera.get("name") or camera_id,
        "host": camera.get("host"),
        "rtsp_port": int(camera.get("rtsp_port", 554)),
    })
    CAMERA_STATUS[camera_id] = cached
    return cached


def get_ffmpeg():
    candidates = [
        os.path.join(get_data_root(), "bin", "ffmpeg.exe"),
        "ffmpeg",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-version"],
                capture_output=True,
                text=True,
                creationflags=hidden_subprocess_flags(),
            )
            if result.returncode == 0:
                return candidate
        except OSError:
            continue
    return None


def capture_with_ffmpeg(camera, url, output_path, transport=None):
    ffmpeg = get_ffmpeg()
    if not ffmpeg:
        return False
    transport = transport or camera.get("rtsp_transport", "tcp")
    cmd = [
        ffmpeg,
        "-y",
        "-rtsp_transport",
        transport,
        "-rw_timeout",
        "5000000",
        "-i",
        url,
        "-frames:v",
        "1",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8, creationflags=hidden_subprocess_flags())
    except subprocess.TimeoutExpired:
        return False
    if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return False
    if has_smeared_bottom_half(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass
        return False
    return True


def has_smeared_bottom_half(image_path):
    img = cv2.imread(image_path)
    if img is None or img.shape[0] < 20:
        return False
    h = img.shape[0]
    top = img[: h // 2].astype(np.float32)
    bottom = img[h // 2 :].astype(np.float32)
    top_vertical_variation = float(np.mean(np.std(top, axis=0)))
    bottom_vertical_variation = float(np.mean(np.std(bottom, axis=0)))
    row_diffs = np.mean(np.abs(np.diff(bottom, axis=0)), axis=(1, 2))
    repeated_row_fraction = float(np.mean(row_diffs < 1.5)) if len(row_diffs) else 0.0
    longest_repeated_run = 0
    current_run = 0
    for is_repeated in row_diffs < 1.5:
        current_run = current_run + 1 if is_repeated else 0
        longest_repeated_run = max(longest_repeated_run, current_run)
    longest_repeated_fraction = float(longest_repeated_run / max(1, len(row_diffs)))
    bottom_texture = float(np.mean(np.std(bottom, axis=1)))
    return (
        top_vertical_variation > 5.0
        and (
            bottom_vertical_variation < 2.0
            or repeated_row_fraction > 0.65
            or longest_repeated_fraction > 0.30
            or bottom_texture < 2.0
        )
    )


def capture_network_camera(camera_id, filename=None):
    camera = camera_by_id(camera_id)
    if not camera:
        print(f"[NETWORK_CAMERA] Unknown camera: {camera_id}")
        return None

    if filename is None:
        filename = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        filename += ".jpg"

    output_dir = os.path.join(get_captures_dir(), camera_id)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    status = network_camera_status(camera_id, probe=True)
    if not status.get("reachable"):
        status.update({
            "last_error": f"{camera.get('host')}:{camera.get('rtsp_port', 554)} is unreachable",
            "last_attempt": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        CAMERA_STATUS[camera_id] = status
        print(f"[NETWORK_CAMERA] {camera_id} is configured but unreachable at {status['host']}:{status['rtsp_port']}.")
        return None

    preferred = camera.get("rtsp_transport", "tcp")
    transports = [preferred] + [value for value in ("tcp", "udp") if value != preferred]
    for url in rtsp_urls(camera):
        for transport in transports:
            if capture_with_ffmpeg(camera, url, output_path, transport=transport):
                CAMERA_STATUS[camera_id] = {
                    **status,
                    "reachable": True,
                    "last_capture": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "last_error": "",
                    "transport": transport,
                    "url_path": url.rsplit("/", 1)[-1],
                }
                print(f"[NETWORK_CAMERA] Captured {camera_id}: {output_path}")
                return output_path

    CAMERA_STATUS[camera_id] = {
        **status,
        "last_attempt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_error": "RTSP port is reachable but no configured stream produced a frame.",
    }
    print(f"[NETWORK_CAMERA] Failed to capture {camera_id}; tried {len(rtsp_urls(camera))} stream URL(s).")
    return None


def mjpeg_live_frames(camera_id, fps=3, jpeg_quality=65, max_width=960):
    camera = camera_by_id(camera_id)
    if not camera:
        return

    interval = 1.0 / max(1, min(float(fps), 8.0))
    jpeg_quality = int(max(35, min(int(jpeg_quality), 90)))
    max_width = int(max(320, min(int(max_width), 1920)))

    for url in live_stream_urls(camera):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            continue

        try:
            while True:
                started = time.monotonic()
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.2)
                    continue

                height, width = frame.shape[:2]
                if width > max_width:
                    scale = max_width / float(width)
                    frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)

                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if ok:
                    payload = encoded.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Cache-Control: no-store\r\n\r\n" + payload + b"\r\n"
                    )

                elapsed = time.monotonic() - started
                if elapsed < interval:
                    time.sleep(interval - elapsed)
        finally:
            cap.release()


def _soap_username_token(username, password):
    return f"""
    <wsse:Security s:mustUnderstand="1"
      xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd">
      <wsse:UsernameToken>
        <wsse:Username>{username}</wsse:Username>
        <wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText">{password}</wsse:Password>
      </wsse:UsernameToken>
    </wsse:Security>
    """


def _post_soap(url, username, password, body, timeout=3):
    envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
  xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
  xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Header>{_soap_username_token(username, password)}</s:Header>
  <s:Body>{body}</s:Body>
</s:Envelope>"""
    request = urllib.request.Request(
        url,
        data=envelope.encode("utf-8"),
        headers={"Content-Type": "application/soap+xml; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def ptz_move(camera_id, direction, speed=0.45, duration_ms=350):
    camera = configured_camera(camera_id)
    if not camera:
        return {"status": "error", "message": f"Unknown network camera: {camera_id}"}

    url = camera.get("onvif_url")
    if not url and camera.get("yoosee_bridge_device"):
        return yoosee_bridge_move(camera, direction, duration_ms)
    if not url:
        return {"status": "error", "message": f"No ONVIF/PTZ URL configured for {camera_id}"}

    vectors = {
        "left": (-speed, 0.0),
        "right": (speed, 0.0),
        "up": (0.0, speed),
        "down": (0.0, -speed),
        "up_left": (-speed, speed),
        "up_right": (speed, speed),
        "down_left": (-speed, -speed),
        "down_right": (speed, -speed),
    }
    if direction not in vectors:
        return {"status": "error", "message": f"Unsupported PTZ direction: {direction}"}

    pan, tilt = vectors[direction]
    token = camera.get("ptz_profile_token", "MainStream")
    username = camera.get("username", "admin")
    password = str(camera.get("password", ""))
    duration_ms = int(max(80, min(2000, duration_ms)))

    move_body = f"""
    <tptz:ContinuousMove>
      <tptz:ProfileToken>{token}</tptz:ProfileToken>
      <tptz:Velocity>
        <tt:PanTilt x="{pan:.3f}" y="{tilt:.3f}"/>
      </tptz:Velocity>
      <tptz:Timeout>PT{duration_ms / 1000:.2f}S</tptz:Timeout>
    </tptz:ContinuousMove>
    """
    stop_body = f"""
    <tptz:Stop>
      <tptz:ProfileToken>{token}</tptz:ProfileToken>
      <tptz:PanTilt>true</tptz:PanTilt>
      <tptz:Zoom>false</tptz:Zoom>
    </tptz:Stop>
    """

    try:
        status, _ = _post_soap(url, username, password, move_body)
        time.sleep(duration_ms / 1000)
        _post_soap(url, username, password, stop_body)
        return {"status": "ok", "http_status": status, "direction": direction}
    except urllib.error.URLError as e:
        if camera.get("yoosee_bridge_device"):
            fallback = yoosee_bridge_move(camera, direction, duration_ms)
            fallback["fallback_reason"] = str(e.reason)
            return fallback
        return {"status": "error", "message": str(e.reason)}
    except Exception as e:
        if camera.get("yoosee_bridge_device"):
            fallback = yoosee_bridge_move(camera, direction, duration_ms)
            fallback["fallback_reason"] = str(e)
            return fallback
        return {"status": "error", "message": str(e)}


def ptz_stop(camera_id):
    camera = configured_camera(camera_id)
    if not camera or not camera.get("onvif_url"):
        return {"status": "error", "message": f"No ONVIF/PTZ URL configured for {camera_id}"}
    token = camera.get("ptz_profile_token", "MainStream")
    body = f"""
    <tptz:Stop>
      <tptz:ProfileToken>{token}</tptz:ProfileToken>
      <tptz:PanTilt>true</tptz:PanTilt>
      <tptz:Zoom>true</tptz:Zoom>
    </tptz:Stop>
    """
    try:
        status, _ = _post_soap(camera["onvif_url"], camera.get("username", "admin"), str(camera.get("password", "")), body)
        return {"status": "ok", "http_status": status}
    except Exception as e:
        if camera.get("yoosee_bridge_device"):
            return {"status": "ok", "method": "yoosee_bridge", "message": "No-op stop; bridge moves are short swipes."}
        return {"status": "error", "message": str(e)}


def yoosee_bridge_move(camera, direction, duration_ms=350):
    device_id = camera.get("yoosee_bridge_device")
    joystick = camera.get("yoosee_joystick") or {}
    center = joystick.get("center", [360, 910])
    targets = {
        "up": joystick.get("up", [360, 803]),
        "down": joystick.get("down", [360, 1017]),
        "left": joystick.get("left", [253, 910]),
        "right": joystick.get("right", [467, 910]),
        "up_left": [joystick.get("left", [253, 910])[0], joystick.get("up", [360, 803])[1]],
        "up_right": [joystick.get("right", [467, 910])[0], joystick.get("up", [360, 803])[1]],
        "down_left": [joystick.get("left", [253, 910])[0], joystick.get("down", [360, 1017])[1]],
        "down_right": [joystick.get("right", [467, 910])[0], joystick.get("down", [360, 1017])[1]],
    }
    target = targets.get(direction)
    if not device_id or not target:
        return {"status": "error", "message": f"Yoosee bridge not configured for {direction}"}

    duration_ms = int(max(120, min(1200, duration_ms)))
    cmd = [
        "adb",
        "-s",
        device_id,
        "shell",
        "input",
        "swipe",
        str(int(center[0])),
        str(int(center[1])),
        str(int(target[0])),
        str(int(target[1])),
        str(duration_ms),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, creationflags=hidden_subprocess_flags())
    if result.returncode == 0:
        return {"status": "ok", "method": "yoosee_bridge", "direction": direction}
    return {"status": "error", "method": "yoosee_bridge", "message": result.stderr or result.stdout or "ADB swipe failed"}
