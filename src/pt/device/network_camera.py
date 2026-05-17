import os
import subprocess
import time
import urllib.error
import urllib.request
from urllib.parse import quote

import cv2
import yaml

from pt.core.utils.path_utils import get_captures_dir, get_data_root


CONFIG_DIR = os.path.join(get_data_root(), "configs")
CONFIG_PATH = os.path.join(CONFIG_DIR, "network_cameras.yaml")
LOCAL_CONFIG_PATH = os.path.join(CONFIG_DIR, "network_cameras.local.yaml")
DEFAULT_RTSP_PATHS = ("onvif1", "onvif2")


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


def get_ffmpeg():
    candidates = [
        os.path.join(get_data_root(), "bin", "ffmpeg.exe"),
        "ffmpeg",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for candidate in candidates:
        try:
            result = subprocess.run([candidate, "-version"], capture_output=True, text=True)
            if result.returncode == 0:
                return candidate
        except OSError:
            continue
    return None


def capture_with_ffmpeg(camera, url, output_path):
    ffmpeg = get_ffmpeg()
    if not ffmpeg:
        return False
    transport = camera.get("rtsp_transport", "udp")
    cmd = [
        ffmpeg,
        "-y",
        "-rtsp_transport",
        transport,
        "-i",
        url,
        "-frames:v",
        "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0


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

    for url in rtsp_urls(camera):
        if capture_with_ffmpeg(camera, url, output_path):
            print(f"[NETWORK_CAMERA] Captured {camera_id}: {output_path}")
            return output_path

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                continue

            frame = None
            for _ in range(12):
                ok, candidate = cap.read()
                if ok and candidate is not None:
                    frame = candidate
                    break

            if frame is not None and cv2.imwrite(output_path, frame):
                print(f"[NETWORK_CAMERA] Captured {camera_id}: {output_path}")
                return output_path
        finally:
            cap.release()

    print(f"[NETWORK_CAMERA] Failed to capture {camera_id}; tried {len(rtsp_urls(camera))} stream URL(s).")
    return None


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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        return {"status": "ok", "method": "yoosee_bridge", "direction": direction}
    return {"status": "error", "method": "yoosee_bridge", "message": result.stderr or result.stdout or "ADB swipe failed"}
