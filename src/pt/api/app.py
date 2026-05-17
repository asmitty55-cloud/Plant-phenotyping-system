import os
import re
import time
import threading
import subprocess
import json
import shutil
import cv2
from datetime import datetime, timezone
from flask import Flask, render_template_string, jsonify, send_from_directory, request

from pt.core.analysis import process_latest_captures
from pt.core.analysis.calibration_store import calib_store
from pt.core.analysis.volumetric import calibrate_extrinsics, reconstruct_visual_hull
from pt.core.utils.path_utils import get_captures_dir, get_data_root
from pt.device.calibration.phone_interrogate import interrogate_phone
from pt.device.calibration.phone_logger import PhoneLogger
from pt.device.capture_service import capture
from pt.device.network_camera import capture_network_camera, configured_camera_ids, ptz_move, ptz_stop


# Get paths
DATA_ROOT = get_data_root()
CAPTURES_DIR = get_captures_dir()
VIDEOS_DIR = os.path.join(DATA_ROOT, "videos")
DEBUG_DIR = os.path.join(DATA_ROOT, "debug")
profiles_file = os.path.join(DEBUG_DIR, "profiles.json")
device_settings_file = os.path.join(DEBUG_DIR, "device_settings.json")
legacy_profiles_file = os.path.join(os.getcwd(), "debug", "profiles.json")
MEDIA_EXTENSIONS = (".jpg", ".jpeg", ".png", ".mp4", ".mov")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CAPTURE_IMAGE_RE = re.compile(r"^capture_\d{8}_\d{6}\.(jpg|jpeg|png)$", re.IGNORECASE)
DEFAULT_REMOTE_DIR = "/sdcard/PTCaptures"

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

app = Flask(__name__)

# Global state
phone_profiles = {}
device_settings = {}
last_capture = {}
timelapse_running = True # Auto-start
timelapse_interval = 180 # Increased from 120 for legacy hardware stability
interrogation_in_progress = set()
device_locks = {}

DEFAULT_DEVICE_SETTINGS = {
    "light_mode": "auto",
    "zoom_percent": 0,
    "delay_ms": 5000,
    "exposure_compensation": 0,
    "iso": "auto",
    "focus_mode": "continuous-picture",
    "antibanding": "60hz",
}
DAY_CAPTURE_PROFILE = {
    "delay_ms": 5000,
    "exposure_compensation": 0,
    "iso": "auto",
}
NIGHT_CAPTURE_PROFILE = {
    "delay_ms": 8000,
    "exposure_compensation": 4,
    "iso": "800",
}
NIGHT_LUMA_THRESHOLD = 45.0


def _is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def _is_capture_image(filename):
    return bool(CAPTURE_IMAGE_RE.match(filename))


def _device_lock(device_id):
    if device_id not in device_locks:
        device_locks[device_id] = threading.Lock()
    return device_locks[device_id]


def _latest_capture_file(files):
    captures = sorted([f for f in files if _is_capture_image(f)])
    return captures[-1] if captures else None


def _latest_reference_file(files):
    latest_capture = _latest_capture_file(files)
    if latest_capture:
        return latest_capture
    images = sorted([f for f in files if _is_image_file(f)])
    return images[-1] if images else None

def load_profiles():
    phone_profiles.clear()
    for candidate in (legacy_profiles_file, profiles_file):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, 'r') as f:
                loaded = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        for device_id, profile in loaded.items():
            current = phone_profiles.get(device_id)
            if current is None or _profile_rank(profile) > _profile_rank(current):
                phone_profiles[device_id] = profile
    if phone_profiles:
        save_profiles()


def _profile_rank(profile):
    if not profile:
        return 0
    score = 0
    if profile.get("save_folder"):
        score += 1
    if profile.get("shutter_success"):
        score += 2
    return score

def save_profiles():
    os.makedirs(os.path.dirname(profiles_file), exist_ok=True)
    with open(profiles_file, 'w') as f:
        json.dump(phone_profiles, f, indent=4)


def load_device_settings():
    device_settings.clear()
    if not os.path.exists(device_settings_file):
        return
    try:
        with open(device_settings_file, "r") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(loaded, dict):
        device_settings.update(loaded)


def save_device_settings():
    os.makedirs(os.path.dirname(device_settings_file), exist_ok=True)
    with open(device_settings_file, "w") as f:
        json.dump(device_settings, f, indent=4)


def settings_for_device(device_id):
    settings = dict(DEFAULT_DEVICE_SETTINGS)
    settings.update(device_settings.get(device_id, {}))
    if settings.get("light_mode") not in ("auto", "day", "night"):
        settings["light_mode"] = "auto"
    settings["zoom_percent"] = int(max(0, min(100, settings.get("zoom_percent", 0))))
    settings["delay_ms"] = int(max(500, min(15000, settings.get("delay_ms", 5000))))
    settings["exposure_compensation"] = int(max(-12, min(12, settings.get("exposure_compensation", 0))))
    if str(settings.get("iso", "auto")) not in ("auto", "100", "200", "400", "800", "1600"):
        settings["iso"] = "auto"
    return settings


def latest_luminance(device_id):
    device_dir = os.path.join(CAPTURES_DIR, device_id)
    if not os.path.exists(device_dir):
        return None
    latest = _latest_capture_file(os.listdir(device_dir))
    if not latest:
        return None
    frame = cv2.imread(os.path.join(device_dir, latest), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None
    return float(frame.mean())


def sensed_light_mode(device_id):
    luma = latest_luminance(device_id)
    if luma is None:
        return "day", None
    return ("night" if luma < NIGHT_LUMA_THRESHOLD else "day"), luma


def effective_capture_settings(device_id):
    settings = settings_for_device(device_id)
    sensed_mode, luma = sensed_light_mode(device_id)
    active_mode = sensed_mode if settings["light_mode"] == "auto" else settings["light_mode"]
    profile = NIGHT_CAPTURE_PROFILE if active_mode == "night" else DAY_CAPTURE_PROFILE
    effective = dict(settings)
    effective.update(profile)
    effective["light_mode"] = settings["light_mode"]
    effective["active_light_mode"] = active_mode
    effective["latest_luminance"] = luma
    return effective


def settings_response(device_id):
    settings = settings_for_device(device_id)
    sensed_mode, luma = sensed_light_mode(device_id)
    active_mode = sensed_mode if settings["light_mode"] == "auto" else settings["light_mode"]
    response = dict(settings)
    response["active_light_mode"] = active_mode
    response["latest_luminance"] = luma
    return response

def run_adb(cmd):
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = proc.communicate()
        return out.strip(), err.strip()
    except FileNotFoundError:
        return "", "adb-not-found"

def detect_connected_devices():
    out, err = run_adb(["adb", "devices"])
    if err == "adb-not-found":
        return []
    lines = out.split("\n")[1:]
    devices = []
    for line in lines:
        if "\tdevice" in line:
            devices.append(line.split("\t")[0])
    return devices


def ensure_device_profile(device_id):
    profile = phone_profiles.get(device_id)
    if profile and profile.get("save_folder") and profile.get("shutter_success"):
        return profile

    if device_id in interrogation_in_progress:
        return None

    interrogation_in_progress.add(device_id)
    try:
        profile = interrogate_phone(device_id)
        if profile and profile.get("save_folder"):
            phone_profiles[device_id] = profile
            save_profiles()
            return profile
        return profile
    finally:
        interrogation_in_progress.discard(device_id)


def ensure_all_device_profiles(devices):
    for device_id in devices:
        ensure_device_profile(device_id)


def _state_path(local_device_dir):
    return os.path.join(local_device_dir, ".sync_state.json")


def _load_sync_state(local_device_dir):
    path = _state_path(local_device_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_sync_state(local_device_dir, state):
    state["last_sync_at"] = datetime.now(timezone.utc).isoformat()
    with open(_state_path(local_device_dir), "w") as f:
        json.dump(state, f, indent=4)


def _remote_dir_for(device_id):
    return phone_profiles.get(device_id, {}).get("save_folder") or DEFAULT_REMOTE_DIR


def _remote_media_files(device_id, remote_dir):
    out, _ = run_adb(["adb", "-s", device_id, "shell", f"ls \"{remote_dir}\""])
    files = []
    for name in out.splitlines():
        clean = name.strip()
        if clean and clean.lower().endswith(MEDIA_EXTENSIONS):
            files.append(clean)
    return sorted(files)


def _pull_remote_file(device_id, remote_dir, filename, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    remote_path = f"{remote_dir}/{filename}"
    local_path = os.path.join(local_dir, filename)
    out, err = run_adb(["adb", "-s", device_id, "pull", remote_path, local_path])
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    print(f"[SYNC] Failed to pull {remote_path} from {device_id}: {err or out}")
    return None


def _delete_remote_file(device_id, remote_dir, filename):
    run_adb(["adb", "-s", device_id, "shell", f"rm \"{remote_dir}/{filename}\""])


def sync_device(device_id, logger):
    """
    Research-grade sync logic:
    1. First connection: full media backup, then clear device except latest image.
    2. Later connections: differential image sync, then clear device except latest image.
    """
    remote_dir = _remote_dir_for(device_id)
    local_device_dir = os.path.join(CAPTURES_DIR, device_id)
    os.makedirs(local_device_dir, exist_ok=True)
    state = _load_sync_state(local_device_dir)
    is_first_sync = not state.get("initial_backup_complete")

    # Ensure remote directory exists
    run_adb(["adb", "-s", device_id, "shell", f"mkdir -p {remote_dir}"])

    remote_files = _remote_media_files(device_id, remote_dir)
    if not remote_files:
        logger.log("No files found on device to sync.", major=False)
        state["initial_backup_complete"] = True
        _save_sync_state(local_device_dir, state)
        return

    latest_file = _latest_reference_file(remote_files) or remote_files[-1]

    if is_first_sync:
        logger.log(f"New device {device_id} detected. Starting full backup of {len(remote_files)} files.", major=True)
        backup_dir = os.path.join(local_device_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)

        for f in remote_files:
            _pull_remote_file(device_id, remote_dir, f, backup_dir)

        if _is_capture_image(latest_file):
            backup_latest = os.path.join(backup_dir, latest_file)
            if os.path.exists(backup_latest):
                shutil.copy2(backup_latest, os.path.join(local_device_dir, latest_file))
        logger.log(f"Initial sync complete. Backup in {backup_dir}.", major=True)
    else:
        local_files = sorted([f for f in os.listdir(local_device_dir) if _is_capture_image(f)])
        last_local = local_files[-1] if local_files else ""

        to_pull = [
            f for f in remote_files
            if _is_capture_image(f) and f > last_local
        ]
        if to_pull:
            logger.log(f"Differential sync: {len(to_pull)} new files found for {device_id}.", major=True)
            for f in to_pull:
                _pull_remote_file(device_id, remote_dir, f, local_device_dir)
        else:
            logger.log(f"No new files for {device_id}.", major=False)

    for f in remote_files:
        if f != latest_file:
            _delete_remote_file(device_id, remote_dir, f)

    state.update({
        "initial_backup_complete": True,
        "last_remote_file": latest_file,
        "remote_dir": remote_dir,
    })
    _save_sync_state(local_device_dir, state)
    
    logger.log(f"Sync finished for {device_id}. Reference file {latest_file} remains on device.", major=False)


def capture_and_sync(device_id):
    lock = _device_lock(device_id)
    if not lock.acquire(blocking=False):
        print(f"[CAPTURE] {device_id} already has a capture in progress.")
        return False

    try:
        ensure_device_profile(device_id)
        logger = PhoneLogger(device_id)
        local_device_dir = os.path.join(CAPTURES_DIR, device_id)
        if not _load_sync_state(local_device_dir).get("initial_backup_complete"):
            sync_device(device_id, logger)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"

        settings = effective_capture_settings(device_id)
        logger.log(
            f"Triggering capture: {filename} light={settings['active_light_mode']} mode={settings['light_mode']} luma={settings['latest_luminance']} zoom={settings['zoom_percent']}% focus={settings['focus_mode']} antibanding={settings['antibanding']}",
            major=True,
        )
        if capture.capture_on_device(
            device_id,
            filename,
            zoom_percent=settings["zoom_percent"],
            delay=settings["delay_ms"],
            exposure=settings["exposure_compensation"],
            iso=settings["iso"],
            focus_mode=settings["focus_mode"],
            antibanding=settings["antibanding"],
        ):
            sync_device(device_id, logger)
            last_capture[device_id] = time.strftime("%Y-%m-%d %H:%M:%S")

            try:
                process_latest_captures(CAPTURES_DIR)
            except Exception as e:
                print(f"[ANALYSIS] Error: {e}")

            assemble_video(device_id)
            return True
        return False
    finally:
        lock.release()


def capture_network_and_analyze(camera_id):
    lock = _device_lock(camera_id)
    if not lock.acquire(blocking=False):
        print(f"[NETWORK_CAMERA] {camera_id} already has a capture in progress.")
        return False
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        if capture_network_camera(camera_id, filename):
            last_capture[camera_id] = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                process_latest_captures(CAPTURES_DIR)
            except Exception as e:
                print(f"[ANALYSIS] Error: {e}")
            assemble_video(camera_id)
            return True
        return False
    finally:
        lock.release()

def get_ffmpeg():
    # Attempt to find ffmpeg in common locations
    candidates = [
        "ffmpeg",
        os.path.join(DATA_ROOT, "bin", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\pt\bin\ffmpeg.exe"
    ]
    for c in candidates:
        try:
            result = subprocess.run([c, "-version"], capture_output=True, text=True)
            if result.returncode == 0:
                return c
        except: continue
    return None


def _ffmpeg_concat_path(path):
    return os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")


def assemble_video(device_id):
    ffmpeg = get_ffmpeg()
    if not ffmpeg:
        print(f"[VIDEO] FFmpeg not found. Cannot assemble video for {device_id}.")
        return

    device_dir = os.path.join(CAPTURES_DIR, device_id)
    if not os.path.exists(device_dir):
        print(f"[VIDEO] No capture directory for {device_id}: {device_dir}")
        return

    images = sorted([f for f in os.listdir(device_dir) if _is_capture_image(f)])
    if len(images) < 2:
        print(f"[VIDEO] Need at least 2 frames for {device_id}; found {len(images)} in {device_dir}.")
        return

    list_file = os.path.join(device_dir, "file_list.txt")
    with open(list_file, "w") as f:
        for img in images:
            img_path = _ffmpeg_concat_path(os.path.join(device_dir, img))
            f.write(f"file '{img_path}'\n")
            f.write("duration 0.1\n")
        img_path = _ffmpeg_concat_path(os.path.join(device_dir, images[-1]))
        f.write(f"file '{img_path}'\n")

    output_file = os.path.join(VIDEOS_DIR, f"{device_id}.mp4")
    # ffmpeg concat demuxer command
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", output_file]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"[VIDEO] Updated timelapse video for {device_id}: {len(images)} frames -> {output_file}")
        else:
            print(f"[VIDEO] FFmpeg completed but output is missing or empty for {device_id}: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"[VIDEO] FFmpeg error for {device_id}: {e.stderr}")

def timelapse_loop():
    print("[TIMELAPSE] Starting background loop...")
    while True:
        if timelapse_running:
            devices = detect_connected_devices()
            ensure_all_device_profiles(devices)
            for d in devices:
                try:
                    capture_and_sync(d)
                except Exception as e:
                    print(f"Error on device {d}: {e}")
            for camera_id in configured_camera_ids():
                try:
                    capture_network_and_analyze(camera_id)
                except Exception as e:
                    print(f"Error on network camera {camera_id}: {e}")
        time.sleep(timelapse_interval)

@app.route("/calibrate_multicam")
def run_multicam_calibration():
    devices = detect_connected_devices()
    device_images = {}
    for d in devices:
        device_dir = os.path.join(CAPTURES_DIR, d)
        if os.path.exists(device_dir):
            latest = _latest_capture_file(os.listdir(device_dir))
            if latest:
                device_images[d] = os.path.join(device_dir, latest)

    if not device_images:
        return "No images found", 400

    results = calibrate_extrinsics(device_images)
    return jsonify({"status": "ok", "calibrated": list(results.keys())})

@app.route("/ignore_region/<device_id>", methods=["POST"])
def add_ignore(device_id):
    import flask
    data = flask.request.json
    # data expected: [x1, y1, x2, y2]
    calib_store.add_ignore_region(device_id, data)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "device": device_id, "stats": stats.get(device_id, {})})

@app.route("/clear_ignore/<device_id>")
def clear_ignore(device_id):
    calib_store.clear_ignore_regions(device_id)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "device": device_id, "stats": stats.get(device_id, {})})

@app.route("/marker_size", methods=["POST"])
def set_marker_size():
    import flask
    data = flask.request.json
    # data: { marker_id, size_mm, device_id (optional) }
    calib_store.set_marker_size(data["marker_id"], data["size_mm"], data.get("device_id"))
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "stats": stats.get(data.get("device_id"), {}) if data.get("device_id") else {}})

@app.route("/charuco_target", methods=["POST"])
def set_charuco_target():
    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id")
    calib_store.set_charuco_target(data, device_id)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "stats": stats.get(device_id, {}) if device_id else {}})

@app.route("/reconstruct")
def run_reconstruction():
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    if not os.path.exists(stats_file):
        return "No stats found", 404

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    device_masks = {}
    for device_id, info in stats.items():
        # This is a bit tricky as masks are not saved to disk by default
        # We might need to re-run analysis or save masks
        device_dir = os.path.join(CAPTURES_DIR, device_id)
        latest = _latest_capture_file(os.listdir(device_dir))
        if latest:
            from pt.core.analysis.image_analysis import analyze_image
            res = analyze_image(os.path.join(device_dir, latest), device_id=device_id)
            if res and "mask" in res:
                device_masks[device_id] = res["mask"]

    if len(device_masks) < 2:
        return "Need at least 2 masks for reconstruction", 400

    vol_data = reconstruct_visual_hull(device_masks)
    if vol_data.get("status") != "ok":
        return jsonify(vol_data), 400

    # Save to a global reconstruction log
    rec_file = os.path.join(DATA_ROOT, "reconstruction.json")
    history = []
    if os.path.exists(rec_file):
        try:
            with open(rec_file, 'r') as f: history = json.load(f)
        except: pass

    vol_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append(vol_data)
    with open(rec_file, 'w') as f: json.dump(history[-100:], f, indent=4)

    return jsonify(vol_data)

@app.route("/")
def dashboard():
    devices = detect_connected_devices() + configured_camera_ids()
    # Load debug logs for each device
    logs = {}
    for d in devices:
        log_path = os.path.join(DEBUG_DIR, f"{d}.txt")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                # Get last 10 lines
                logs[d] = "".join(f.readlines()[-10:])
        else:
            logs[d] = "No logs yet."

    return render_template_string(OBSERVATORY_DASHBOARD_HTML, devices=devices, last=last_capture, running=timelapse_running, logs=logs)

@app.route("/video/<device_id>")
def serve_video(device_id):
    return send_from_directory(VIDEOS_DIR, f"{device_id}.mp4")

@app.route("/stats")
def get_stats():
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/device_settings")
def get_device_settings():
    devices = detect_connected_devices()
    known = sorted(set(devices) | set(device_settings.keys()))
    return jsonify({device_id: settings_response(device_id) for device_id in known})


@app.route("/device_settings/<device_id>", methods=["POST"])
def update_device_settings(device_id):
    data = request.get_json(silent=True) or {}
    current = settings_for_device(device_id)
    if data.get("light_mode") in ("auto", "day", "night"):
        current["light_mode"] = data["light_mode"]
    if "zoom_percent" in data:
        current["zoom_percent"] = int(max(0, min(100, int(data["zoom_percent"]))))
    if "delay_ms" in data:
        current["delay_ms"] = int(max(500, min(15000, int(data["delay_ms"]))))
    if "exposure_compensation" in data:
        current["exposure_compensation"] = int(max(-12, min(12, int(data["exposure_compensation"]))))
    if str(data.get("iso")) in ("auto", "100", "200", "400", "800", "1600"):
        current["iso"] = str(data["iso"])
    if data.get("focus_mode") in ("continuous-picture", "auto", "infinity", "macro", "fixed"):
        current["focus_mode"] = data["focus_mode"]
    if data.get("antibanding") in ("off", "50hz", "60hz", "auto"):
        current["antibanding"] = data["antibanding"]
    device_settings[device_id] = current
    save_device_settings()
    return jsonify(settings_response(device_id))

@app.route("/analysis_debug/<device_id>")
def serve_analysis_debug(device_id):
    debug_dir = os.path.join(DATA_ROOT, "analysis_debug")
    device_dir = os.path.join(CAPTURES_DIR, device_id)
    if not os.path.exists(device_dir): return "Not found", 404
    latest = _latest_capture_file(os.listdir(device_dir))
    if not latest: return "No images", 404
    debug_path = os.path.join(debug_dir, latest)
    if not os.path.exists(debug_path): return "No debug image", 404
    return send_from_directory(debug_dir, latest)

@app.route("/last_frame/<device_id>")
def serve_last_frame(device_id):
    device_dir = os.path.join(CAPTURES_DIR, device_id)
    if not os.path.exists(device_dir): return "Not found", 404
    latest = _latest_capture_file(os.listdir(device_dir))
    if not latest: return "No images", 404
    return send_from_directory(device_dir, latest)

@app.route("/capture/<device_id>")
def manual_capture(device_id):
    if device_id in configured_camera_ids():
        if capture_network_and_analyze(device_id): return "OK"
        return "Failed", 500
    if capture_and_sync(device_id): return "OK"
    return "Failed", 500


@app.route("/ptz/<camera_id>/<direction>", methods=["POST"])
def move_network_camera(camera_id, direction):
    if camera_id not in configured_camera_ids():
        return jsonify({"status": "error", "message": "Unknown network camera"}), 404
    data = request.get_json(silent=True) or {}
    result = ptz_move(
        camera_id,
        direction,
        speed=float(data.get("speed", 0.45)),
        duration_ms=int(data.get("duration_ms", 350)),
    )
    return jsonify(result), (200 if result.get("status") == "ok" else 502)


@app.route("/ptz/<camera_id>/stop", methods=["POST"])
def stop_network_camera(camera_id):
    if camera_id not in configured_camera_ids():
        return jsonify({"status": "error", "message": "Unknown network camera"}), 404
    result = ptz_stop(camera_id)
    return jsonify(result), (200 if result.get("status") == "ok" else 502)

@app.route("/interrogate/<device_id>")
def run_interrogate(device_id):
    profile = interrogate_phone(device_id)
    if profile and profile.get("save_folder"):
        capture.install_apk(device_id)
        phone_profiles[device_id] = profile
        save_profiles()
        return jsonify(profile)
    return "Failed", 500

@app.route("/timelapse/stop")
def stop_timelapse():
    global timelapse_running
    timelapse_running = False
    return "Stopped"

@app.route("/timelapse/start")
def start_timelapse():
    global timelapse_running
    if not timelapse_running:
        timelapse_running = True
    return "Started"

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Plant Timelapse Research Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 20px; margin: 0; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 25px; }
        .card { background: #151515; border-radius: 12px; padding: 20px; border: 1px solid #333; box-shadow: 0 4px 15px rgba(0,0,0,0.5); display: flex; flex-direction: column; }
        video, img { width: 100%; border-radius: 8px; background: #000; margin-top: 10px; border: 1px solid #222; }
        canvas { width: 100% !important; height: 150px !important; margin-top: 10px; }
        .controls { margin-top: 20px; display: flex; gap: 12px; }
        button { flex: 1; padding: 14px; cursor: pointer; background: #252525; color: white; border: 1px solid #444; border-radius: 6px; font-weight: bold; transition: 0.2s; }
        button:hover { background: #353535; border-color: #666; }
        .status-bar { margin-bottom: 30px; padding: 15px 25px; border-radius: 8px; display: flex; align-items: center; justify-content: space-between; background: #1a1a1a; border: 1px solid #333; }
        .running { color: #74c69d; }
        .stopped { color: #ffb3c1; }
        h1 { margin: 0; font-size: 1.8em; }
        h3 { margin-top: 0; color: #fff; border-bottom: 1px solid #333; padding-bottom: 10px; }
        .label { font-size: 0.75em; color: #666; text-transform: uppercase; margin-top: 15px; letter-spacing: 1px; font-weight: bold; }
        .fastest-badge { background: #f39c12; color: #000; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8em; margin-left: 10px; animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 0.8; } 50% { opacity: 1; } 100% { opacity: 0.8; } }
    </style>
</head>
<body>
    <div class="status-bar">
        <h1>Plant Timelapse <span style="color:#4facfe">Research</span></h1>
        <div style="display: flex; gap: 10px; align-items: center;">
            <button onclick="calibrateMulti()" style="background:#2c3e50; padding: 8px 15px;">Calibrate 3D</button>
            <button onclick="reconstruct3D()" style="background:#27ae60; padding: 8px 15px;">Run Volumetric</button>
            <div class="{{ 'running' if running else 'stopped' }}" style="margin-left:10px">
                 ● SYSTEM {{ 'RUNNING' if running else 'STOPPED' }}
            </div>
            <button onclick="fetch('/timelapse/start').then(()=>location.reload())" style="padding: 8px 15px;">START</button>
            <button onclick="fetch('/timelapse/stop').then(()=>location.reload())" style="padding: 8px 15px;">STOP</button>
        </div>
    </div>

    <div class="grid">
        {% for d in devices %}
        <div class="card">
            <h3 id="title-{{ d }}">{{ d }}</h3>
            <p>Last Sync: <span class="timestamp">{{ last.get(d, 'Initializing...') }}</span></p>

            <div class="label">Timelapse Loop</div>
            <video id="vid-{{d}}" controls loop autoplay muted>
                <source src="/video/{{d}}" type="video/mp4">
            </video>

            <div class="controls">
                <button onclick="fetch('/capture/{{d}}').then(()=>location.reload())">Force Capture</button>
                <button onclick="document.getElementById('analysis-{{d}}').src='/last_frame/{{d}}?t='+Date.now()">Refresh Frame</button>
                <button onclick="clearIgnore('{{d}}')">Clear Masks</button>
            </div>

            <div class="label">Realtime Frame (ArUco Detection)</div>
            <div style="position:relative">
                <img id="analysis-{{d}}" src="/analysis_debug/{{d}}"
                     onclick="handleImageClick(event, '{{d}}')"
                     onload="this.style.display='block'"
                     onerror="this.onerror=null; this.src='/last_frame/{{d}}'">
                <div id="selection-{{d}}" style="position:absolute; border: 2px dashed #f1c40f; pointer-events:none; display:none;"></div>
            </div>

            <div class="label">Plant Statistics & Growth Trend</div>
            <div id="stats-{{d}}" style="font-family: monospace; font-size: 0.9em; background: #000; padding: 10px; border-radius: 4px; border: 1px solid #333; margin-bottom: 10px;">
                Loading telemetry...
            </div>
            <canvas id="chart-{{d}}" style="height: 150px; background: #111; border-radius: 4px;"></canvas>

            <div class="label">System Debug Log</div>
            <div class="debug-log" id="log-{{d}}" style="font-family: monospace; font-size: 0.8em; background: #000; padding: 10px; border-radius: 4px; border: 1px solid #333; height: 100px; overflow-y: auto; white-space: pre-wrap; margin-top: 5px;">{{ logs[d] }}</div>
        </div>
        {% endfor %}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const charts = {};
        let selectionStart = null;

        function calibrateMulti() {
            if(confirm("Calibrate 3D coordinate system using current frames?")) {
                fetch('/calibrate_multicam').then(r => r.json()).then(d => alert("Calibrated: " + d.calibrated.join(", ")));
            }
        }

        function reconstruct3D() {
            fetch('/reconstruct').then(r => r.json()).then(d => {
                alert("Volumetric Reconstruction: " + (d.volume_mm3 / 1000).toFixed(2) + " cm³");
            });
        }

        function clearIgnore(deviceId) {
            fetch('/clear_ignore/' + deviceId).then(() => alert("Ignore regions cleared"));
        }

        function handleImageClick(event, deviceId) {
            const img = event.target;
            const rect = img.getBoundingClientRect();
            const x = (event.clientX - rect.left) * (img.naturalWidth / rect.width);
            const y = (event.clientY - rect.top) * (img.naturalHeight / rect.height);

            if (!selectionStart) {
                selectionStart = { x, y, screenX: event.clientX, screenY: event.clientY };
                const sel = document.getElementById('selection-' + deviceId);
                sel.style.left = (event.clientX - rect.left) + 'px';
                sel.style.top = (event.clientY - rect.top) + 'px';
                sel.style.width = '0px';
                sel.style.height = '0px';
                sel.style.display = 'block';
            } else {
                const x2 = x;
                const y2 = y;
                const region = [
                    Math.min(selectionStart.x, x2),
                    Math.min(selectionStart.y, y2),
                    Math.max(selectionStart.x, x2),
                    Math.max(selectionStart.y, y2)
                ].map(Math.round);

                if (confirm("Ignore this region for " + deviceId + "?")) {
                    fetch('/ignore_region/' + deviceId, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(region)
                    }).then(() => {
                        selectionStart = null;
                        document.getElementById('selection-' + deviceId).style.display = 'none';
                        alert("Region added to ignore list");
                    });
                } else {
                    selectionStart = null;
                    document.getElementById('selection-' + deviceId).style.display = 'none';
                }
            }
        }

        function updateStats() {
            fetch('/stats').then(r => {
                if (!r.ok) throw new Error("Stats fetch failed");
                return r.json();
            }).then(stats => {
                if (!stats || Object.keys(stats).length === 0) return;

                for (const [id, info] of Object.entries(stats)) {
                    const cardTitle = document.querySelector(`.card h3#title-${id}`);
                    if (cardTitle && info.is_fastest) {
                        if (!cardTitle.innerHTML.includes('FASTEST')) {
                            cardTitle.innerHTML += ' <span class="fastest-badge">🔥 FASTEST GROWER</span>';
                        }
                    }

                    const el = document.getElementById('stats-' + id);
                    if (el && info.data) {
                        const data = info.data;
                        const area = (data.plant_area_mm2 !== undefined && data.plant_area_mm2 !== null)
                                     ? data.plant_area_mm2.toFixed(1) + ' mm²'
                                     : '0.0 mm²';
                        const rate = info.growth_rate_mm2_hr ? info.growth_rate_mm2_hr.toFixed(2) + ' mm²/hr' : '0.00 mm²/hr';
                        const coverage = data.canopy_coverage !== undefined && data.canopy_coverage !== null
                                     ? (data.canopy_coverage * 100).toFixed(1) + '%'
                                     : 'N/A';
                        const color = data.color_metrics || {};
                        const deficiency = info.nutrient_deficiency || {};
                        const deficiencyText = deficiency.status === 'ready'
                                     ? `${deficiency.severity.toUpperCase()} (${(deficiency.score || 0).toFixed(2)}) ${deficiency.flags && deficiency.flags.length ? deficiency.flags.join(', ') : 'stable'}`
                                     : 'Collecting baseline';
                        el.innerHTML = `
                            Markers Found: ${data.markers_found || 0}<br>
                            Scale: ${data.scale_px_per_mm ? data.scale_px_per_mm.toFixed(2) + ' px/mm' : 'N/A'}<br>
                            Canopy Area: ${area}<br>
                            Canopy Coverage: ${coverage}<br>
                            Green Index: ${color.green_index !== undefined ? color.green_index.toFixed(3) : 'N/A'}<br>
                            Chlorosis Ratio: ${color.chlorosis_ratio !== undefined ? color.chlorosis_ratio.toFixed(3) : 'N/A'}<br>
                            Deficiency: ${deficiencyText}<br>
                            Growth Rate: ${rate}<br>
                            Last Update: ${info.timestamp || 'Never'}
                        `;
                    }

                    // Update Chart
                    if (info.history && info.history.length > 0) {
                        const ctx = document.getElementById('chart-' + id);
                        if (ctx) {
                            // Only plot points that have valid area data
                            const validHistory = info.history.filter(h =>
                                h.area !== undefined && h.area !== null && h.area > 0
                            );

                            if (validHistory.length === 0) continue;

                            const labels = validHistory.map(h => h.timestamp ? h.timestamp.split(' ')[1] : '');
                            const values = validHistory.map(h => h.area);

                            if (!charts[id]) {
                                charts[id] = new Chart(ctx, {
                                    type: 'line',
                                    data: {
                                        labels: labels,
                                        datasets: [{
                                            label: 'Area (mm²)',
                                            data: values,
                                            borderColor: '#74c69d',
                                            backgroundColor: 'rgba(116, 198, 157, 0.1)',
                                            fill: true,
                                            tension: 0.3,
                                            pointRadius: 2
                                        }]
                                    },
                                    options: {
                                        responsive: true,
                                        maintainAspectRatio: false,
                                        animation: false,
                                        plugins: { legend: { display: false } },
                                        scales: {
                                            y: { grid: { color: '#222' }, ticks: { color: '#666' } },
                                            x: { grid: { display: false }, ticks: { color: '#666' } }
                                        }
                                    }
                                });
                            } else {
                                charts[id].data.labels = labels;
                                charts[id].data.datasets[0].data = values;
                                charts[id].update('none');
                            }
                        }
                    }
                }
            }).catch(e => console.error("Update Stats Error:", e));
        }

        setInterval(() => {
            document.querySelectorAll('img').forEach(img => {
                const base = img.src.split('?')[0];
                img.src = base + '?t=' + Date.now();
            });
            document.querySelectorAll('video').forEach(v => { if(v.paused) v.play().catch(()=>{}); });
            updateStats();
        }, 15000);
        updateStats();
    </script>
</body>
</html>
"""

OBSERVATORY_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Plant Observatory</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        :root {
            --bg: #0d1110; --panel: #151b19; --panel-2: #101514; --line: #29322f;
            --text: #edf4ef; --muted: #8c9a94; --green: #7ac77f; --cyan: #77b7c5;
            --amber: #d8ad5f; --red: #df7d7d;
        }
        * { box-sizing: border-box; }
        body { margin: 0; background: var(--bg); color: var(--text); font-family: "Segoe UI", system-ui, sans-serif; }
        .shell { min-height: 100vh; display: grid; grid-template-columns: 248px 1fr; }
        .sidebar { border-right: 1px solid var(--line); background: #0f1413; padding: 24px 18px; position: sticky; top: 0; height: 100vh; }
        .brand { font-size: 1.2rem; font-weight: 700; margin-bottom: 4px; }
        .subtitle, .muted { color: var(--muted); }
        .subtitle { font-size: .82rem; margin-bottom: 28px; }
        .nav-section { color: #6f7c76; font-size: .72rem; text-transform: uppercase; margin: 22px 0 8px; }
        .nav-item { display: flex; align-items: center; justify-content: space-between; padding: 10px 12px; border-radius: 6px; color: #c4cec8; margin-bottom: 4px; cursor: pointer; border: 1px solid transparent; }
        .nav-item.active { background: #1b2421; color: var(--text); border: 1px solid #31413b; }
        .pill { font-size: .72rem; color: #0d1110; background: var(--green); padding: 2px 7px; border-radius: 999px; font-weight: 700; }
        .main { padding: 24px; min-width: 0; }
        .topbar { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
        h1, h2, h3 { margin: 0; letter-spacing: 0; }
        h1 { font-size: 1.7rem; }
        h2 { font-size: 1rem; }
        h3 { font-size: .98rem; }
        .context { color: var(--muted); margin-top: 5px; }
        .actions { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
        button { min-height: 38px; padding: 0 13px; border-radius: 6px; border: 1px solid #3a4541; background: #1a211f; color: var(--text); cursor: pointer; font-weight: 650; }
        button:hover { border-color: #5b6963; background: #202927; }
        button.primary { background: #24452d; border-color: #40794c; }
        button.warn { background: #3a2b1d; border-color: #73562b; }
        .status-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }
        .running .status-dot { background: var(--green); }
        .stopped .status-dot { background: var(--red); }
        .summary { display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 12px; margin-bottom: 18px; }
        .metric, .panel, .device-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
        .metric { padding: 14px; min-height: 86px; }
        .metric .label { color: var(--muted); font-size: .76rem; text-transform: uppercase; }
        .metric .value { font-size: 1.45rem; margin-top: 8px; font-weight: 700; }
        .metric .hint { color: var(--muted); font-size: .78rem; margin-top: 3px; }
        .workspace { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr); gap: 16px; align-items: start; }
        .panel { padding: 16px; }
        .panel-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }
        .camera-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
        .device-card { padding: 14px; min-width: 0; }
        .device-title { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 10px; }
        .device-id { overflow-wrap: anywhere; font-weight: 700; }
        .badge { color: #111; background: var(--amber); border-radius: 999px; padding: 3px 8px; font-size: .72rem; font-weight: 800; white-space: nowrap; }
        img, video { width: 100%; border-radius: 6px; background: #050706; border: 1px solid #26302c; display: block; }
        video { aspect-ratio: 16 / 9; object-fit: contain; }
        img { max-height: 390px; object-fit: contain; }
        .controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 10px 0 14px; }
        .mask-active { outline: 2px solid var(--amber); cursor: crosshair; }
        .image-wrap { margin-top: 10px; position: relative; }
        .calib-overlay { position: absolute; inset: 0; pointer-events: none; }
        .marker-hotspot {
            position: absolute;
            border: 2px solid var(--red);
            background: rgba(223, 125, 125, .16);
            color: #fff;
            border-radius: 6px;
            transform: translate(-50%, -50%);
            pointer-events: auto;
            cursor: pointer;
            min-width: 28px;
            min-height: 28px;
            display: grid;
            place-items: center;
            font-size: .72rem;
            font-weight: 800;
        }
        .charuco-hotspot { border-style: dashed; background: rgba(223, 125, 125, .08); transform: none; }
        .calibration-popover {
            position: fixed;
            z-index: 20;
            min-width: 260px;
            background: #121816;
            border: 1px solid #495650;
            border-radius: 8px;
            padding: 14px;
            box-shadow: 0 18px 50px rgba(0,0,0,.45);
        }
        .calibration-popover label { display: block; color: var(--muted); font-size: .76rem; margin-top: 10px; }
        .calibration-popover input { width: 100%; min-height: 34px; border-radius: 6px; border: 1px solid #3a4541; background: #0b100f; color: var(--text); padding: 0 8px; }
        .popover-actions { display: flex; gap: 8px; margin-top: 12px; }
        .ptz-pad { display: grid; grid-template-columns: repeat(3, 42px); gap: 6px; justify-content: center; margin: 12px 0; }
        .ptz-pad button { min-height: 36px; padding: 0; }
        .view-section { display: none; }
        .view-section.active { display: block; }
        .settings-row { display: grid; grid-template-columns: 90px 1fr 54px; align-items: center; gap: 10px; margin-top: 10px; }
        select, input[type="range"] { width: 100%; accent-color: var(--green); }
        select { min-height: 34px; border-radius: 6px; border: 1px solid #3a4541; background: #101514; color: var(--text); padding: 0 8px; }
        .telemetry { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
        .telemetry div { background: var(--panel-2); border: 1px solid #222c28; border-radius: 6px; padding: 10px; min-height: 58px; }
        .telemetry span { display: block; color: var(--muted); font-size: .72rem; text-transform: uppercase; }
        .telemetry strong { display: block; margin-top: 5px; overflow-wrap: anywhere; }
        canvas { width: 100% !important; height: 210px !important; }
        .timeline { display: grid; gap: 10px; margin-top: 12px; }
        .event { display: grid; grid-template-columns: 82px 1fr; gap: 10px; border-top: 1px solid #26302c; padding-top: 10px; color: #c9d3ce; }
        .event time { color: var(--muted); font-size: .78rem; }
        .log { font-family: Consolas, monospace; font-size: .76rem; white-space: pre-wrap; background: #070908; border: 1px solid #202824; border-radius: 6px; padding: 10px; max-height: 130px; overflow: auto; color: #aab7b0; }
        .empty { border: 1px dashed #3a4541; border-radius: 8px; padding: 28px; color: var(--muted); text-align: center; }
        @media (max-width: 980px) {
            .shell { grid-template-columns: 1fr; }
            .sidebar { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
            .workspace, .summary { grid-template-columns: 1fr; }
            .topbar { flex-direction: column; }
            .actions { justify-content: flex-start; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <aside class="sidebar">
            <div class="brand">Plant Observatory</div>
            <div class="subtitle">Phenotyping control and biological state</div>
            <div class="nav-section">Workspace</div>
            <div class="nav-item active" data-view="mission" onclick="showSection('mission', this)">Mission Control <span class="pill">{{ devices|length }}</span></div>
            <div class="nav-item" data-view="experiments" onclick="showSection('experiments', this)">Active Experiments</div>
            <div class="nav-item" data-view="plants" onclick="showSection('plants', this)">Plants</div>
            <div class="nav-item" data-view="analytics" onclick="showSection('analytics', this)">Analytics</div>
            <div class="nav-section">Operations</div>
            <div class="nav-item" data-view="timelapses" onclick="showSection('timelapses', this)">Timelapses</div>
            <div class="nav-item" data-view="calibration" onclick="showSection('calibration', this)">Calibration</div>
            <div class="nav-item" data-view="health" onclick="showSection('health', this)">Device Health</div>
            <div class="nav-item" data-view="settings" onclick="showSection('settings', this)">Settings</div>
        </aside>
        <main class="main">
            <div class="topbar">
                <div>
                    <h1>Mission Control</h1>
                    <div class="context">Live capture, canopy analysis, calibration, and trait ranking</div>
                </div>
                <div class="actions">
                    <button class="primary" onclick="fetch('/timelapse/start').then(()=>location.reload())">Start Loop</button>
                    <button class="warn" onclick="fetch('/timelapse/stop').then(()=>location.reload())">Stop Loop</button>
                    <button onclick="calibrateMulti()">Calibrate 3D</button>
                    <button onclick="reconstruct3D()">Run Volumetric</button>
                </div>
            </div>
            <section class="summary">
                <div class="metric"><div class="label">System</div><div class="value {{ 'running' if running else 'stopped' }}"><span class="status-dot"></span>{{ 'Running' if running else 'Stopped' }}</div><div class="hint">capture interval 180s</div></div>
                <div class="metric"><div class="label">Cameras Online</div><div class="value">{{ devices|length }}</div><div class="hint">ADB devices detected</div></div>
                <div class="metric"><div class="label">Last Capture</div><div class="value" id="last-capture-summary">--</div><div class="hint">latest synced frame</div></div>
                <div class="metric"><div class="label">Fastest Plant</div><div class="value" id="fastest-summary">--</div><div class="hint">by area gain</div></div>
                <div class="metric"><div class="label">Calibration</div><div class="value" id="calibration-summary">--</div><div class="hint">marker detection state</div></div>
            </section>
            <section class="panel" id="operation-result" style="display:none; margin-bottom:16px"></section>
            <div class="workspace view-section active" id="view-mission">
                <section class="panel">
                    <div class="panel-head"><div><h2>Camera Feeds</h2><div class="muted">Latest frame with segmentation overlay and capture controls</div></div></div>
                    {% if devices %}
                    <div class="camera-grid">
                        {% for d in devices %}
                        <article class="device-card" data-device="{{ d }}">
                            <div class="device-title"><div class="device-id" id="title-{{ d }}">{{ d }}</div><span class="badge" id="badge-{{ d }}" style="display:none">Fastest</span></div>
                            <div class="muted">Last sync: <span id="last-{{ d }}">{{ last.get(d, 'Initializing') }}</span></div>
                            <div class="image-wrap" id="image-wrap-{{d}}">
                                <img id="analysis-{{d}}" src="/analysis_debug/{{d}}" onload="renderCalibrationOverlays('{{d}}', (latestStats['{{d}}'] || {}).data || {})" onpointerdown="startIgnoreDrag(event, '{{d}}')" onpointermove="moveIgnoreDrag(event, '{{d}}')" onpointerup="finishIgnoreDrag(event, '{{d}}')" onpointercancel="cancelIgnoreDrag('{{d}}')" onerror="this.onerror=null; this.src='/last_frame/{{d}}'">
                                <div class="calib-overlay" id="calib-overlay-{{d}}"></div>
                                <div id="selection-{{d}}" style="position:absolute; border: 2px dashed var(--amber); pointer-events:none; display:none;"></div>
                            </div>
                            <div class="controls">
                                <button onclick="fetch('/capture/{{d}}').then(()=>location.reload())">Capture</button>
                                <button onclick="refreshFrame('{{d}}')">Refresh</button>
                                <button onclick="enableIgnoreMode('{{d}}')">Ignore Region</button>
                                <button onclick="clearIgnore('{{d}}')">Clear Ignores</button>
                            </div>
                            {% if d.startswith('escam_') %}
                            <div class="ptz-pad">
                                <button onclick="ptzMove('{{d}}','up_left')">↖</button>
                                <button onclick="ptzMove('{{d}}','up')">↑</button>
                                <button onclick="ptzMove('{{d}}','up_right')">↗</button>
                                <button onclick="ptzMove('{{d}}','left')">←</button>
                                <button onclick="ptzStop('{{d}}')">■</button>
                                <button onclick="ptzMove('{{d}}','right')">→</button>
                                <button onclick="ptzMove('{{d}}','down_left')">↙</button>
                                <button onclick="ptzMove('{{d}}','down')">↓</button>
                                <button onclick="ptzMove('{{d}}','down_right')">↘</button>
                            </div>
                            {% endif %}
                            <div class="settings-row">
                                <label class="muted" for="zoom-{{d}}">Zoom</label>
                                <input id="zoom-{{d}}" type="range" min="0" max="100" value="0" oninput="previewZoom('{{d}}', this.value)" onchange="saveDeviceSetting('{{d}}', 'zoom_percent', this.value)">
                                <strong id="zoom-value-{{d}}">0%</strong>
                            </div>
                            <div class="settings-row">
                                <label class="muted" for="delay-{{d}}">Settle</label>
                                <input id="delay-{{d}}" type="range" min="500" max="15000" step="500" value="5000" oninput="previewDelay('{{d}}', this.value)" onchange="saveDeviceSetting('{{d}}', 'delay_ms', this.value)">
                                <strong id="delay-value-{{d}}">5.0s</strong>
                            </div>
                            <div class="settings-row">
                                <label class="muted" for="antibanding-{{d}}">Banding</label>
                                <select id="antibanding-{{d}}" onchange="saveDeviceSetting('{{d}}', 'antibanding', this.value)">
                                    <option value="60hz">60 Hz</option>
                                    <option value="50hz">50 Hz</option>
                                    <option value="auto">Auto</option>
                                    <option value="off">Off</option>
                                </select>
                                <strong></strong>
                            </div>
                            <div class="settings-row">
                                <label class="muted" for="focus-{{d}}">Focus</label>
                                <select id="focus-{{d}}" onchange="saveDeviceSetting('{{d}}', 'focus_mode', this.value)">
                                    <option value="continuous-picture">Continuous</option>
                                    <option value="auto">Auto</option>
                                    <option value="macro">Macro</option>
                                    <option value="infinity">Infinity</option>
                                    <option value="fixed">Fixed</option>
                                </select>
                                <strong></strong>
                            </div>
                            <div class="settings-row">
                                <label class="muted" for="exposure-{{d}}">Exposure</label>
                                <input id="exposure-{{d}}" type="range" min="-12" max="12" step="1" value="0" oninput="previewExposure('{{d}}', this.value)" onchange="saveDeviceSetting('{{d}}', 'exposure_compensation', this.value)">
                                <strong id="exposure-value-{{d}}">0</strong>
                            </div>
                            <div class="settings-row">
                                <label class="muted" for="iso-{{d}}">ISO</label>
                                <select id="iso-{{d}}" onchange="saveDeviceSetting('{{d}}', 'iso', this.value)">
                                    <option value="auto">Auto</option>
                                    <option value="100">100</option>
                                    <option value="200">200</option>
                                    <option value="400">400</option>
                                    <option value="800">800</option>
                                    <option value="1600">1600</option>
                                </select>
                                <strong></strong>
                            </div>
                            <div class="controls">
                                <button id="profile-toggle-{{d}}" onclick="toggleLightProfile('{{d}}')">Night Profile</button>
                                <button id="auto-light-{{d}}" onclick="setAutoLight('{{d}}')">Auto Light</button>
                            </div>
                            <video id="vid-{{d}}" controls loop autoplay muted><source src="/video/{{d}}" type="video/mp4"></video>
                            <div class="telemetry" id="stats-{{d}}">
                                <div><span>Markers</span><strong>--</strong></div><div><span>Canopy</span><strong>--</strong></div>
                                <div><span>Growth</span><strong>--</strong></div><div><span>Health</span><strong>Collecting</strong></div>
                            </div>
                        </article>
                        {% endfor %}
                    </div>
                    {% else %}
                    <div class="empty">No ADB cameras are online. Connect the Galaxy S5 Neo and keep USB debugging enabled.</div>
                    {% endif %}
                </section>
                <aside class="panel">
                    <div class="panel-head"><div><h2>Growth Analytics</h2><div class="muted">Canopy area over time</div></div></div>
                    <canvas id="fleet-chart"></canvas>
                    <div class="timeline" id="event-timeline"><div class="event"><time>Now</time><div>Waiting for capture telemetry.</div></div></div>
                    <div style="margin-top:16px">
                        <h3>Device Health</h3>
                        {% for d in devices %}
                        <div style="margin-top:10px"><div class="muted">{{ d }}</div><div class="log" id="log-{{d}}">{{ logs[d] }}</div></div>
                        {% endfor %}
                    </div>
                </aside>
            </div>
            <section class="panel view-section" id="view-experiments"><h2>Active Experiments</h2><div class="timeline" id="experiments-list"><div class="event"><time>Live</time><div>Current capture loop is the active experiment. Experiment grouping is ready for plant/genotype metadata next.</div></div></div></section>
            <section class="panel view-section" id="view-plants"><h2>Plants</h2><div class="timeline" id="plants-list"></div></section>
            <section class="panel view-section" id="view-analytics">
                <h2>Analytics</h2>
                <div class="muted" style="margin-top:6px">Trait ranking, segmentation features, and circadian motion signals</div>
                <canvas id="analytics-chart"></canvas>
                <div class="timeline" id="trait-ranking"></div>
                <div class="timeline">
                    <div class="event"><time>Segment</time><div><strong>Feature extraction framework</strong><br>Plant segmentation, mask extraction, contour analysis, green pixel analysis, canopy metrics, and shape descriptors.</div></div>
                    <div class="event"><time>Derived</time><div>Projected leaf area, convex hull area, centroid, canopy density, Excess Green, and entropy.</div></div>
                    <div class="event"><time>Circadian</time><div><strong>Movement rhythm analysis</strong><br>Leaf angle changes, centroid movement, canopy width oscillation, and posture variation across photoperiod cycles.</div></div>
                    <div class="event"><time>Motion</time><div>Planned optical flow and motion field analysis for circadian entrainment, stress disruption, and photoperiod response.</div></div>
                </div>
            </section>
            <section class="panel view-section" id="view-timelapses"><h2>Timelapses</h2><div class="timeline" id="timelapse-list"></div></section>
            <section class="panel view-section" id="view-calibration"><h2>Calibration</h2><div class="timeline" id="calibration-list"><div class="event"><time>Ready</time><div>Use Calibrate 3D to estimate camera poses from current ChArUco detections. Results now appear in the operation panel above Mission Control.</div></div></div></section>
            <section class="panel view-section" id="view-health"><h2>Device Health</h2><div class="timeline" id="health-list"></div></section>
            <section class="panel view-section" id="view-settings"><h2>Settings</h2><div class="timeline"><div class="event"><time>Camera</time><div>Per-device zoom, anti-banding, and focus controls live on each camera card. Capture again after changing them so the APK applies the new settings.</div></div></div></section>
        </main>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const charts = {};
        const ignoreState = {};
        let latestStats = {};
        function fmt(value, digits = 1) { return Number.isFinite(value) ? value.toFixed(digits) : '--'; }
        function showOperation(title, detail, tone = 'normal') {
            const el = document.getElementById('operation-result');
            el.style.display = 'block';
            el.style.borderColor = tone === 'bad' ? 'var(--red)' : tone === 'warn' ? 'var(--amber)' : 'var(--line)';
            el.innerHTML = `<h2>${title}</h2><div class="muted" style="margin-top:6px">${detail}</div>`;
        }
        function showSection(name, item) {
            document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            const section = document.getElementById('view-' + name);
            if (section) section.classList.add('active');
            if (item) item.classList.add('active');
        }
        function calibrateMulti() {
            if (confirm("Calibrate the multi-camera coordinate system using current frames?")) {
                showOperation('Calibration running', 'Looking for ChArUco corners in the current frames...');
                fetch('/calibrate_multicam').then(async r => {
                    const body = r.headers.get('content-type')?.includes('application/json') ? await r.json() : { error: await r.text() };
                    if (!r.ok) throw new Error(body.error || body.message || 'Calibration failed');
                    return body;
                }).then(d => {
                    const calibrated = d.calibrated || [];
                    showOperation('Calibration complete', calibrated.length ? `Camera poses saved for: ${calibrated.join(', ')}` : 'No cameras produced a valid ChArUco pose. Check marker visibility, focus, and lighting.', calibrated.length ? 'normal' : 'warn');
                }).catch(e => showOperation('Calibration failed', e.message, 'bad'));
            }
        }
        function reconstruct3D() {
            showOperation('Volumetric reconstruction running', 'Projecting silhouettes into the calibrated volume...');
            fetch('/reconstruct').then(async r => {
                const body = r.headers.get('content-type')?.includes('application/json') ? await r.json() : { error: await r.text() };
                if (!r.ok) throw new Error(body.error || body.message || 'Reconstruction failed');
                return body;
            }).then(d => {
                const cm3 = d.volume_mm3 ? (d.volume_mm3 / 1000).toFixed(2) : '0.00';
                showOperation('Volumetric reconstruction complete', `Volume: ${cm3} cm3. Occupied voxels: ${d.occupied_voxels || 0}. Grid: ${(d.grid_shape || []).join(' x ')}`);
            }).catch(e => showOperation('Reconstruction failed', e.message, 'bad'));
        }
        function refreshFrame(deviceId) { document.getElementById('analysis-' + deviceId).src = '/analysis_debug/' + deviceId + '?t=' + Date.now(); }
        function clearIgnore(deviceId) {
            fetch('/clear_ignore/' + deviceId).then(() => {
                showOperation('Ignore regions cleared', `${deviceId} will use the full frame on the next analysis pass.`);
                refreshFrame(deviceId);
                updateStats();
            });
        }
        function enableIgnoreMode(deviceId) {
            ignoreState[deviceId] = { enabled: true, dragging: false };
            document.getElementById('analysis-' + deviceId).classList.add('mask-active');
            showOperation('Ignore-region mode', `Drag across the ${deviceId} image to mark erroneous green/artifact space. The region is saved when you release.`);
        }
        function imagePoint(event, img) {
            const rect = img.getBoundingClientRect();
            return {
                x: (event.clientX - rect.left) * (img.naturalWidth / rect.width),
                y: (event.clientY - rect.top) * (img.naturalHeight / rect.height),
                sx: event.clientX - rect.left,
                sy: event.clientY - rect.top
            };
        }
        function startIgnoreDrag(event, deviceId) {
            const state = ignoreState[deviceId];
            if (!state || !state.enabled) return;
            event.preventDefault();
            const img = event.target;
            img.setPointerCapture(event.pointerId);
            const point = imagePoint(event, img);
            const sel = document.getElementById('selection-' + deviceId);
            state.dragging = true;
            state.start = point;
            state.current = point;
            sel.style.left = point.sx + 'px';
            sel.style.top = point.sy + 'px';
            sel.style.width = '0px';
            sel.style.height = '0px';
            sel.style.display = 'block';
        }
        function moveIgnoreDrag(event, deviceId) {
            const state = ignoreState[deviceId];
            if (!state || !state.dragging) return;
            event.preventDefault();
            const point = imagePoint(event, event.target);
            state.current = point;
            const sel = document.getElementById('selection-' + deviceId);
            sel.style.left = Math.min(state.start.sx, point.sx) + 'px';
            sel.style.top = Math.min(state.start.sy, point.sy) + 'px';
            sel.style.width = Math.abs(point.sx - state.start.sx) + 'px';
            sel.style.height = Math.abs(point.sy - state.start.sy) + 'px';
        }
        function finishIgnoreDrag(event, deviceId) {
            const state = ignoreState[deviceId];
            if (!state || !state.dragging) return;
            event.preventDefault();
            moveIgnoreDrag(event, deviceId);
            const region = [
                Math.min(state.start.x, state.current.x),
                Math.min(state.start.y, state.current.y),
                Math.max(state.start.x, state.current.x),
                Math.max(state.start.y, state.current.y)
            ].map(Math.round);
            cancelIgnoreDrag(deviceId);
            if (Math.abs(region[2] - region[0]) < 8 || Math.abs(region[3] - region[1]) < 8) {
                showOperation('Ignore region skipped', 'The drawn box was too small to save.', 'warn');
                return;
            }
            fetch('/ignore_region/' + deviceId, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(region) })
                .then(() => {
                    showOperation('Ignore region saved', `${deviceId}: [${region.join(', ')}]. The overlay has been regenerated without that region.`);
                    refreshFrame(deviceId);
                    updateStats();
                });
        }
        function cancelIgnoreDrag(deviceId) {
            const state = ignoreState[deviceId] || {};
            state.enabled = false;
            state.dragging = false;
            ignoreState[deviceId] = state;
            const img = document.getElementById('analysis-' + deviceId);
            if (img) img.classList.remove('mask-active');
            const sel = document.getElementById('selection-' + deviceId);
            if (sel) sel.style.display = 'none';
        }
        function previewZoom(deviceId, value) {
            document.getElementById('zoom-value-' + deviceId).textContent = `${value}%`;
        }
        function previewDelay(deviceId, value) {
            document.getElementById('delay-value-' + deviceId).textContent = `${(Number(value) / 1000).toFixed(1)}s`;
        }
        function previewExposure(deviceId, value) {
            document.getElementById('exposure-value-' + deviceId).textContent = value;
        }
        function saveDeviceSetting(deviceId, key, value) {
            const payload = {};
            payload[key] = (key === 'zoom_percent' || key === 'delay_ms' || key === 'exposure_compensation') ? Number(value) : value;
            fetch('/device_settings/' + deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(r => r.json()).then(settings => {
                applyDeviceSettings(deviceId, settings);
                showOperation('Camera setting saved', `${deviceId}: zoom ${settings.zoom_percent}%, settle ${(settings.delay_ms / 1000).toFixed(1)}s, exposure ${settings.exposure_compensation}, ISO ${settings.iso}, focus ${settings.focus_mode}, anti-banding ${settings.antibanding}. It applies on the next capture.`);
            });
        }
        function saveDeviceSettings(deviceId, payload) {
            return fetch('/device_settings/' + deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(r => r.json()).then(settings => {
                applyDeviceSettings(deviceId, settings);
                return settings;
            });
        }
        function toggleLightProfile(deviceId) {
            const current = document.getElementById('profile-toggle-' + deviceId)?.dataset.activeMode || 'day';
            const nextMode = current === 'night' ? 'day' : 'night';
            const profile = nextMode === 'night' ? {
                light_mode: 'night',
                zoom_percent: 0,
                delay_ms: 8000,
                exposure_compensation: 4,
                iso: '800',
                focus_mode: 'continuous-picture',
                antibanding: '60hz'
            } : {
                light_mode: 'day',
                delay_ms: 5000,
                exposure_compensation: 0,
                iso: 'auto',
                focus_mode: 'continuous-picture',
                antibanding: '60hz'
            };
            saveDeviceSettings(deviceId, profile).then(settings => {
                const label = nextMode === 'night' ? 'Night profile saved' : 'Day profile saved';
                showOperation(label, `${deviceId}: now in ${nextMode} mode. You can still fine-tune exposure, ISO, settle, focus, and zoom individually.`);
            });
        }
        function setAutoLight(deviceId) {
            saveDeviceSettings(deviceId, { light_mode: 'auto' }).then(settings => {
                showOperation('Auto light sensing enabled', `${deviceId}: latest brightness ${settings.latest_luminance === null ? 'unknown' : settings.latest_luminance.toFixed(1)}; active profile is ${settings.active_light_mode}.`);
            });
        }
        function ptzMove(cameraId, direction) {
            fetch(`/ptz/${cameraId}/${direction}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ speed: 0.45, duration_ms: 350 })
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'PTZ command failed');
                showOperation('Camera moved', `${cameraId}: ${direction}`);
                setTimeout(() => refreshFrame(cameraId), 700);
            }).catch(e => showOperation('PTZ failed', `${cameraId}: ${e.message}`, 'bad'));
        }
        function ptzStop(cameraId) {
            fetch(`/ptz/${cameraId}/stop`, { method: 'POST' })
                .then(r => r.json())
                .then(body => showOperation('Camera stop sent', `${cameraId}: ${body.status}`))
                .catch(e => showOperation('PTZ stop failed', `${cameraId}: ${e.message}`, 'bad'));
        }
        function applyDeviceSettings(deviceId, settings) {
            const zoom = document.getElementById('zoom-' + deviceId);
            const zoomValue = document.getElementById('zoom-value-' + deviceId);
            const delay = document.getElementById('delay-' + deviceId);
            const delayValue = document.getElementById('delay-value-' + deviceId);
            const exposure = document.getElementById('exposure-' + deviceId);
            const exposureValue = document.getElementById('exposure-value-' + deviceId);
            const iso = document.getElementById('iso-' + deviceId);
            const focus = document.getElementById('focus-' + deviceId);
            const antibanding = document.getElementById('antibanding-' + deviceId);
            const toggle = document.getElementById('profile-toggle-' + deviceId);
            const auto = document.getElementById('auto-light-' + deviceId);
            if (zoom) zoom.value = settings.zoom_percent || 0;
            if (zoomValue) zoomValue.textContent = `${settings.zoom_percent || 0}%`;
            if (delay) delay.value = settings.delay_ms || 5000;
            if (delayValue) delayValue.textContent = `${((settings.delay_ms || 5000) / 1000).toFixed(1)}s`;
            if (exposure) exposure.value = settings.exposure_compensation || 0;
            if (exposureValue) exposureValue.textContent = settings.exposure_compensation || 0;
            if (iso) iso.value = settings.iso || 'auto';
            if (focus) focus.value = settings.focus_mode || 'continuous-picture';
            if (antibanding) antibanding.value = settings.antibanding || '60hz';
            if (toggle) {
                toggle.dataset.activeMode = settings.active_light_mode || settings.light_mode || 'day';
                toggle.textContent = (settings.active_light_mode || 'day') === 'night' ? 'Day Profile' : 'Night Profile';
            }
            if (auto) {
                const luma = settings.latest_luminance === null || settings.latest_luminance === undefined ? '--' : Number(settings.latest_luminance).toFixed(0);
                auto.textContent = settings.light_mode === 'auto' ? `Auto: ${(settings.active_light_mode || 'day').toUpperCase()} (${luma})` : 'Auto Light';
            }
        }
        function loadDeviceSettings() {
            fetch('/device_settings').then(r => r.json()).then(all => {
                Object.entries(all).forEach(([deviceId, settings]) => applyDeviceSettings(deviceId, settings));
            });
        }
        function markerBounds(marker) {
            const points = marker.corners || [];
            if (!points.length) return { x: marker.center[0] - 16, y: marker.center[1] - 16, width: 32, height: 32 };
            const xs = points.map(p => p[0]);
            const ys = points.map(p => p[1]);
            const minX = Math.min(...xs), maxX = Math.max(...xs);
            const minY = Math.min(...ys), maxY = Math.max(...ys);
            return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
        }
        function imageScale(deviceId) {
            const img = document.getElementById('analysis-' + deviceId);
            const rect = img.getBoundingClientRect();
            return {
                sx: rect.width / Math.max(1, img.naturalWidth),
                sy: rect.height / Math.max(1, img.naturalHeight)
            };
        }
        function renderCalibrationOverlays(deviceId, data) {
            const overlay = document.getElementById('calib-overlay-' + deviceId);
            const img = document.getElementById('analysis-' + deviceId);
            if (!overlay || !img || !img.naturalWidth) return;
            const scale = imageScale(deviceId);
            const markers = data.markers || [];
            const markerHtml = markers.map(marker => {
                const b = markerBounds(marker);
                const width = Math.max(28, b.width * scale.sx);
                const height = Math.max(28, b.height * scale.sy);
                const left = marker.center[0] * scale.sx;
                const top = marker.center[1] * scale.sy;
                return `<button class="marker-hotspot" style="left:${left}px;top:${top}px;width:${width}px;height:${height}px" onclick='openMarkerEditor(event, "${deviceId}", ${JSON.stringify(marker)})'>${marker.id}</button>`;
            }).join('');
            let charucoHtml = '';
            if (data.charuco_bbox) {
                const b = data.charuco_bbox;
                charucoHtml = `<button class="marker-hotspot charuco-hotspot" style="left:${b.x * scale.sx}px;top:${b.y * scale.sy}px;width:${Math.max(36, b.width * scale.sx)}px;height:${Math.max(36, b.height * scale.sy)}px" onclick='openCharucoEditor(event, "${deviceId}", ${JSON.stringify(data.charuco_target || data.calibration_target || {})})'>ChArUco</button>`;
            }
            overlay.innerHTML = markerHtml + charucoHtml;
        }
        function closeCalibrationPopover() {
            const existing = document.getElementById('calibration-popover');
            if (existing) existing.remove();
        }
        function placePopover(popover, event) {
            document.body.appendChild(popover);
            const x = Math.min(window.innerWidth - popover.offsetWidth - 12, event.clientX + 12);
            const y = Math.min(window.innerHeight - popover.offsetHeight - 12, event.clientY + 12);
            popover.style.left = Math.max(12, x) + 'px';
            popover.style.top = Math.max(12, y) + 'px';
        }
        function openMarkerEditor(event, deviceId, marker) {
            event.preventDefault();
            event.stopPropagation();
            closeCalibrationPopover();
            const pop = document.createElement('div');
            pop.className = 'calibration-popover';
            pop.id = 'calibration-popover';
            pop.innerHTML = `
                <h3>ArUco marker ${marker.id}</h3>
                <div class="muted">${deviceId}</div>
                <label>Marker size (mm)</label>
                <input id="marker-size-input" type="number" step="0.01" min="1" value="${marker.size_mm || 41.18}">
                <div class="popover-actions">
                    <button class="primary" onclick="saveMarkerSize('${deviceId}', ${marker.id})">Save</button>
                    <button onclick="closeCalibrationPopover()">Cancel</button>
                </div>
            `;
            placePopover(pop, event);
        }
        function openCharucoEditor(event, deviceId, target) {
            event.preventDefault();
            event.stopPropagation();
            closeCalibrationPopover();
            const pop = document.createElement('div');
            pop.className = 'calibration-popover';
            pop.id = 'calibration-popover';
            pop.innerHTML = `
                <h3>ChArUco board</h3>
                <div class="muted">${deviceId}</div>
                <label>Square size (mm)</label>
                <input id="charuco-square-input" type="number" step="0.01" min="1" value="${target.square_size_mm || 51.28}">
                <label>Marker size (mm)</label>
                <input id="charuco-marker-input" type="number" step="0.01" min="1" value="${target.marker_size_mm || 41.18}">
                <div class="popover-actions">
                    <button class="primary" onclick="saveCharucoTarget('${deviceId}')">Save</button>
                    <button onclick="closeCalibrationPopover()">Cancel</button>
                </div>
            `;
            placePopover(pop, event);
        }
        function saveMarkerSize(deviceId, markerId) {
            const size = Number(document.getElementById('marker-size-input').value);
            fetch('/marker_size', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ device_id: deviceId, marker_id: markerId, size_mm: size })
            }).then(() => {
                closeCalibrationPopover();
                showOperation('Marker size saved', `${deviceId} marker ${markerId}: ${size} mm. Analysis has been regenerated with this device-specific size.`);
                refreshFrame(deviceId);
                updateStats();
            });
        }
        function saveCharucoTarget(deviceId) {
            const square = Number(document.getElementById('charuco-square-input').value);
            const marker = Number(document.getElementById('charuco-marker-input').value);
            fetch('/charuco_target', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ device_id: deviceId, square_size_mm: square, marker_size_mm: marker })
            }).then(() => {
                closeCalibrationPopover();
                showOperation('ChArUco dimensions saved', `${deviceId}: square ${square} mm, marker ${marker} mm. Analysis has been regenerated with this device-specific board size.`);
                refreshFrame(deviceId);
                updateStats();
            });
        }
        function renderFleetChart(stats) {
            const ctx = document.getElementById('fleet-chart');
            if (!ctx) return;
            const colors = ['#7ac77f', '#77b7c5', '#d8ad5f', '#df7d7d'];
            const datasets = Object.entries(stats).map(([id, info], idx) => {
                const history = (info.history || []).filter(h => h.area !== null && h.area !== undefined);
                return { label: id, data: history.map(h => ({ x: h.timestamp || '', y: h.area || 0 })), borderColor: colors[idx % colors.length], backgroundColor: 'transparent', tension: 0.25, pointRadius: 2 };
            });
            if (!charts.fleet) {
                charts.fleet = new Chart(ctx, {
                    type: 'line',
                    data: { datasets },
                    options: { responsive: true, maintainAspectRatio: false, parsing: { xAxisKey: 'x', yAxisKey: 'y' }, plugins: { legend: { labels: { color: '#cbd6d0', boxWidth: 10 } } }, scales: { y: { grid: { color: '#27312d' }, ticks: { color: '#8c9a94' } }, x: { grid: { display: false }, ticks: { color: '#8c9a94', maxRotation: 0 } } } }
                });
            } else {
                charts.fleet.data.datasets = datasets;
                charts.fleet.update('none');
            }
        }
        function updateStats() {
            fetch('/stats').then(r => r.json()).then(stats => {
                latestStats = stats || {};
                const entries = Object.entries(stats || {});
                if (entries.length === 0) return;
                let fastest = null, calibrated = 0, latest = '';
                const events = [];
                for (const [id, info] of entries) {
                    const data = info.data || {};
                    const color = data.color_metrics || {};
                    const deficiency = info.nutrient_deficiency || {};
                    const rate = Number(info.growth_rate_mm2_hr || 0);
                    const area = Number(data.plant_area_mm2 || data.canopy_area_mm2 || 0);
                    const coverage = data.canopy_coverage !== undefined ? Number(data.canopy_coverage) * 100 : NaN;
                    const health = deficiency.status === 'ready' ? `${(deficiency.severity || 'none').toUpperCase()} ${(deficiency.score || 0).toFixed(2)}` : 'Baseline';
                    if (data.markers_found > 0) calibrated += 1;
                    if (info.is_fastest) fastest = id;
                    if (info.timestamp && info.timestamp > latest) latest = info.timestamp;
                    const lastEl = document.getElementById('last-' + id);
                    if (lastEl) lastEl.textContent = info.timestamp || 'Initializing';
                    const badge = document.getElementById('badge-' + id);
                    if (badge) badge.style.display = info.is_fastest ? 'inline-block' : 'none';
                    const statsEl = document.getElementById('stats-' + id);
                    if (statsEl) {
                        statsEl.innerHTML = `
                            <div><span>Markers</span><strong>${data.markers_found || 0} detected</strong></div>
                            <div><span>Canopy</span><strong>${fmt(area)} mm2 / ${fmt(coverage)}%</strong></div>
                            <div><span>Growth</span><strong>${fmt(rate, 2)} mm2/hr</strong></div>
                            <div><span>Health</span><strong>${health}</strong></div>
                            <div><span>Scale</span><strong>${data.scale_px_per_mm ? fmt(data.scale_px_per_mm, 2) + ' px/mm' : '--'}</strong></div>
                            <div><span>Green Index</span><strong>${color.green_index !== undefined ? fmt(color.green_index, 3) : '--'}</strong></div>
                        `;
                    }
                    renderCalibrationOverlays(id, data);
                    const history = info.history || [];
                    if (history.length) {
                        const h = history[history.length - 1];
                        events.push({ time: (h.timestamp || '').split(' ')[1] || 'Recent', text: `${id}: canopy ${fmt(Number(h.area || 0))} mm2` });
                    }
                }
                document.getElementById('last-capture-summary').textContent = latest ? latest.split(' ')[1] : '--';
                document.getElementById('fastest-summary').textContent = fastest || '--';
                document.getElementById('calibration-summary').textContent = calibrated + '/' + entries.length;
                renderFleetChart(stats);
                const timeline = document.getElementById('event-timeline');
                if (timeline) timeline.innerHTML = events.slice(-6).reverse().map(e => `<div class="event"><time>${e.time}</time><div>${e.text}</div></div>`).join('');
                const plants = document.getElementById('plants-list');
                if (plants) plants.innerHTML = entries.map(([id, info]) => `<div class="event"><time>${(info.timestamp || '').split(' ')[1] || '--'}</time><div><strong>${id}</strong><br>Area ${fmt(Number((info.data || {}).plant_area_mm2 || 0))} mm2, growth ${fmt(Number(info.growth_rate_mm2_hr || 0), 2)} mm2/hr</div></div>`).join('');
                const ranking = document.getElementById('trait-ranking');
                if (ranking) ranking.innerHTML = entries.slice().sort((a, b) => Number(b[1].growth_rate_mm2_hr || 0) - Number(a[1].growth_rate_mm2_hr || 0)).map(([id, info], idx) => `<div class="event"><time>#${idx + 1}</time><div>${id}: ${fmt(Number(info.growth_rate_mm2_hr || 0), 2)} mm2/hr</div></div>`).join('');
                const timelapses = document.getElementById('timelapse-list');
                if (timelapses) timelapses.innerHTML = entries.map(([id]) => `<div class="event"><time>Video</time><div>${id}<br><video controls src="/video/${id}" style="margin-top:8px"></video></div></div>`).join('');
                const health = document.getElementById('health-list');
                if (health) health.innerHTML = entries.map(([id, info]) => `<div class="event"><time>${(info.timestamp || '').split(' ')[1] || '--'}</time><div>${id}: ${((info.data || {}).markers_found || 0)} markers, health ${(info.nutrient_deficiency || {}).severity || 'collecting'}</div></div>`).join('');
                const calibration = document.getElementById('calibration-list');
                if (calibration) calibration.innerHTML = entries.map(([id, info]) => `<div class="event"><time>${((info.data || {}).markers_found || 0) ? 'Seen' : 'Missing'}</time><div>${id}: ${((info.data || {}).markers_found || 0)} markers, scale ${((info.data || {}).scale_px_per_mm || '--')} px/mm</div></div>`).join('');
            }).catch(e => console.error("Update Stats Error:", e));
        }
        setInterval(() => {
            document.querySelectorAll('img[id^="analysis-"]').forEach(img => { img.src = img.src.split('?')[0] + '?t=' + Date.now(); });
            document.querySelectorAll('video').forEach(v => { if (v.paused) v.play().catch(()=>{}); });
            updateStats();
        }, 15000);
        loadDeviceSettings();
        updateStats();
        window.addEventListener('resize', () => {
            Object.entries(latestStats || {}).forEach(([id, info]) => renderCalibrationOverlays(id, (info || {}).data || {}));
        });
    </script>
</body>
</html>
"""

def run_app():
    load_profiles()
    load_device_settings()
    # Start the timelapse thread immediately
    t = threading.Thread(target=timelapse_loop, daemon=True)
    t.start()
    print(f"[SERVER] Data root: {DATA_ROOT}")
    print(f"[SERVER] Captures: {CAPTURES_DIR}")
    print(f"[SERVER] Videos: {VIDEOS_DIR}")
    print("[SERVER] Starting Flask on port 5000...")
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    run_app()
