import os
import re
import time
import threading
import subprocess
import json
import shutil
import cv2
import numpy as np
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template_string, jsonify, send_from_directory, request, Response, stream_with_context

from pt.core.analysis import process_latest_captures
from pt.core.analysis.calibration_store import calib_store
import pt.core.analysis.metric_store as metric_store
from pt.core.analysis.segmentation_store import segmentation_store
from pt.core.analysis.volumetric import calibrate_extrinsics, reconstruct_visual_hull
from pt.core.experiments import add_event, load_events, load_experiments, load_plant_records, upsert_experiment, upsert_plant_record
from pt.core.utils.path_utils import get_captures_dir, get_data_root
from pt.device.calibration.phone_interrogate import interrogate_phone
from pt.device.calibration.phone_logger import PhoneLogger
from pt.device import adb_transport
from pt.device.capture_service import capture
from pt.device.network_camera import (
    camera_has_live_stream,
    capture_network_camera,
    configured_camera_ids,
    mjpeg_live_frames,
    network_camera_status,
    ptz_move,
    ptz_stop,
)


# Get paths
DATA_ROOT = get_data_root()
CAPTURES_DIR = get_captures_dir()
VIDEOS_DIR = os.path.join(DATA_ROOT, "videos")
WORKING_VIDEOS_DIR = os.path.join(VIDEOS_DIR, "_program_working_do_not_open_while_running")
DEBUG_DIR = os.path.join(DATA_ROOT, "debug")
profiles_file = os.path.join(DEBUG_DIR, "profiles.json")
device_settings_file = os.path.join(DEBUG_DIR, "device_settings.json")
device_aliases_file = os.path.join(DEBUG_DIR, "device_aliases.json")
device_metadata_file = os.path.join(DEBUG_DIR, "device_metadata.json")
video_manifest_file = os.path.join(DEBUG_DIR, "video_manifest.json")
night_skip_state_file = os.path.join(DEBUG_DIR, "night_skip_state.json")
legacy_profiles_file = os.path.join(os.getcwd(), "debug", "profiles.json")
MEDIA_EXTENSIONS = (".jpg", ".jpeg", ".png", ".mp4", ".mov")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CAPTURE_IMAGE_RE = re.compile(r"^capture_\d{8}_\d{6}\.(jpg|jpeg|png)$", re.IGNORECASE)
NIGHT_PLACEHOLDER_RE = re.compile(r"^night_\d{8}_\d{6}\.(jpg|jpeg|png)$", re.IGNORECASE)
DEFAULT_REMOTE_DIR = "/sdcard/PTCaptures"
VIDEO_ASSEMBLY_FRAME_STEP = 5

os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(WORKING_VIDEOS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

app = Flask(__name__)

# Global state
phone_profiles = {}
device_settings = {}
device_aliases = {}
device_metadata = {}
last_capture = {}
timelapse_running = True # Auto-start
timelapse_interval = 180 # Increased from 120 for legacy hardware stability
interrogation_in_progress = set()
device_locks = {}
live_stream_lock = threading.Lock()
active_live_stream_device = None
video_manifest = {}
video_locks = {}
night_skip_started = {}
operation_jobs = {}
operation_jobs_lock = threading.Lock()
activity_lines = []
activity_lock = threading.Lock()

DEFAULT_DEVICE_SETTINGS = {
    "light_mode": "auto",
    "profile_name": "auto",
    "zoom_percent": 0,
    "delay_ms": 5000,
    "exposure_compensation": 0,
    "iso": "auto",
    "focus_mode": "continuous-picture",
    "antibanding": "60hz",
    "white_balance": "daylight",
    "display_rotation_deg": 0,
    "collect_night_frames": True,
    "measurement_locked": False,
    "measurement_locked_at": "",
    "expected_marker_count": 4,
    "measurement_profile_score": 0.0,
    "measurement_profile_summary": "",
}


def update_operation(operation_id, message, progress=None, status=None, result=None):
    if not operation_id:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    with operation_jobs_lock:
        job = operation_jobs.setdefault(operation_id, {
            "id": operation_id,
            "status": "running",
            "progress": 0,
            "lines": [],
            "result": None,
        })
        if message:
            job["lines"].append(f"[{timestamp}] {message}")
            job["lines"] = job["lines"][-120:]
        if progress is not None:
            job["progress"] = int(max(0, min(100, progress)))
        if status:
            job["status"] = status
        if result is not None:
            job["result"] = result
        job["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def operation_snapshot(operation_id):
    with operation_jobs_lock:
        job = operation_jobs.get(operation_id)
        return dict(job) if job else None


def log_activity(message, device_id=None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = f"{device_id}: " if device_id else ""
    line = f"[{timestamp}] {prefix}{message}"
    with activity_lock:
        activity_lines.append(line)
        del activity_lines[:-160]
    print(f"[ACTIVITY] {prefix}{message}")


def activity_snapshot():
    with activity_lock:
        return list(activity_lines)


DAY_CAPTURE_PROFILE = {
    "delay_ms": 5000,
    "exposure_compensation": 0,
    "iso": "auto",
    "white_balance": "daylight",
}
NIGHT_IR_CAPTURE_PROFILE = {
    "delay_ms": 8000,
    "exposure_compensation": 4,
    "iso": "1600",
    "white_balance": "daylight",
}
NIGHT_LUMA_THRESHOLD = 45.0


def _is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def _is_capture_image(filename):
    return bool(CAPTURE_IMAGE_RE.match(filename))


def _is_night_placeholder(filename):
    return bool(NIGHT_PLACEHOLDER_RE.match(filename))


def _is_video_frame(filename):
    return _is_capture_image(filename) or _is_night_placeholder(filename)


def _frame_timestamp(filename):
    match = re.match(r"^(capture|night)_(\d{8})_(\d{6})\.(jpg|jpeg|png)$", filename, re.IGNORECASE)
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(2)}_{match.group(3)}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _frame_files_for_device(device_id, start_dt=None, end_dt=None):
    device_dir = os.path.join(CAPTURES_DIR, device_id)
    if not os.path.exists(device_dir):
        return []
    frames = []
    for filename in os.listdir(device_dir):
        if not _is_video_frame(filename):
            continue
        stamp = _frame_timestamp(filename)
        if stamp is None:
            continue
        if start_dt and stamp < start_dt:
            continue
        if end_dt and stamp > end_dt:
            continue
        frames.append((stamp, filename))
    return [filename for _, filename in sorted(frames)]


def _device_lock(device_id):
    if device_id not in device_locks:
        device_locks[device_id] = threading.Lock()
    return device_locks[device_id]


def _video_lock(device_id):
    if device_id not in video_locks:
        video_locks[device_id] = threading.Lock()
    return video_locks[device_id]


def _safe_video_id(device_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", device_id)


def _video_current_name(device_id):
    return f"{_safe_video_id(device_id)}.mp4"


def _video_current_path(device_id):
    return os.path.join(WORKING_VIDEOS_DIR, _video_current_name(device_id))


def _video_playback_name(device_id):
    return f"{_safe_video_id(device_id)}_playback.mp4"


def _video_playback_path(device_id):
    return os.path.join(VIDEOS_DIR, _video_playback_name(device_id))


def _custom_video_name(device_id, start_label, end_label):
    return f"{_safe_video_id(device_id)}_custom_{start_label}_{end_label}.mp4"


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
    if profile.get("capabilities") and not profile.get("capabilities", {}).get("error"):
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


def load_device_aliases():
    device_aliases.clear()
    if not os.path.exists(device_aliases_file):
        return
    try:
        with open(device_aliases_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(loaded, dict):
        device_aliases.update({str(k): str(v) for k, v in loaded.items() if str(v).strip()})


def save_device_aliases():
    os.makedirs(os.path.dirname(device_aliases_file), exist_ok=True)
    with open(device_aliases_file, "w", encoding="utf-8") as f:
        json.dump(device_aliases, f, indent=2)


def load_device_metadata():
    device_metadata.clear()
    if not os.path.exists(device_metadata_file):
        return
    try:
        with open(device_metadata_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(loaded, dict):
        device_metadata.update(loaded)


def save_device_metadata():
    os.makedirs(os.path.dirname(device_metadata_file), exist_ok=True)
    with open(device_metadata_file, "w", encoding="utf-8") as f:
        json.dump(device_metadata, f, indent=2)


def load_video_manifest():
    video_manifest.clear()
    if not os.path.exists(video_manifest_file):
        return
    try:
        with open(video_manifest_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(loaded, dict):
        for device_id in loaded:
            video_manifest[device_id] = _video_playback_name(device_id)


def save_video_manifest():
    os.makedirs(os.path.dirname(video_manifest_file), exist_ok=True)
    with open(video_manifest_file, "w", encoding="utf-8") as f:
        json.dump(video_manifest, f, indent=2)


def settings_for_device(device_id):
    settings = dict(DEFAULT_DEVICE_SETTINGS)
    settings.update(device_settings.get(device_id, {}))
    if settings.get("light_mode") == "night":
        settings["light_mode"] = "night_ir"
    if settings.get("profile_name") == "night":
        settings["profile_name"] = "night_ir"
    if settings.get("light_mode") not in ("auto", "day", "night_ir"):
        settings["light_mode"] = "auto"
    if settings.get("profile_name") not in ("auto", "day", "wide_day", "night_ir"):
        settings["profile_name"] = settings["light_mode"] if settings["light_mode"] in ("day", "night_ir") else "auto"
    settings["zoom_percent"] = int(max(0, min(100, settings.get("zoom_percent", 0))))
    settings["delay_ms"] = int(max(500, min(15000, settings.get("delay_ms", 5000))))
    settings["exposure_compensation"] = int(max(-12, min(12, settings.get("exposure_compensation", 0))))
    settings["display_rotation_deg"] = float(settings.get("display_rotation_deg", 0.0)) % 360.0
    settings["collect_night_frames"] = bool(settings.get("collect_night_frames", True))
    settings["measurement_locked"] = bool(settings.get("measurement_locked", False))
    settings["measurement_locked_at"] = str(settings.get("measurement_locked_at") or "")
    settings["expected_marker_count"] = int(max(1, min(50, settings.get("expected_marker_count", 4))))
    settings["measurement_profile_score"] = float(settings.get("measurement_profile_score") or 0.0)
    settings["measurement_profile_summary"] = str(settings.get("measurement_profile_summary") or "")
    if str(settings.get("iso", "auto")) not in ("auto", "100", "200", "400", "800", "1600"):
        settings["iso"] = "auto"
    if settings.get("white_balance") not in ("auto", "daylight", "cloudy-daylight", "fluorescent", "incandescent", "shade", "twilight", "warm-fluorescent"):
        settings["white_balance"] = "daylight"
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
    return ("night_ir" if luma < NIGHT_LUMA_THRESHOLD else "day"), luma


def effective_capture_settings(device_id):
    settings = settings_for_device(device_id)
    sensed_mode, luma = sensed_light_mode(device_id)
    active_mode = sensed_mode if settings["light_mode"] == "auto" else settings["light_mode"]
    profile = NIGHT_IR_CAPTURE_PROFILE if active_mode == "night_ir" else DAY_CAPTURE_PROFILE
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


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def load_night_skip_state():
    night_skip_started.clear()
    if not os.path.exists(night_skip_state_file):
        return
    try:
        with open(night_skip_state_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(loaded, dict):
        night_skip_started.update(loaded)


def save_night_skip_state():
    os.makedirs(os.path.dirname(night_skip_state_file), exist_ok=True)
    with open(night_skip_state_file, "w", encoding="utf-8") as f:
        json.dump(night_skip_started, f, indent=2)


def start_night_skip(device_id, timestamp):
    if device_id not in night_skip_started:
        night_skip_started[device_id] = {
            "started_at": time.time(),
            "timestamp": timestamp,
        }
        save_night_skip_state()


def finish_night_skip(device_id, logger=None):
    state = night_skip_started.pop(device_id, None)
    if not state:
        return None
    save_night_skip_state()
    started_at = float(state.get("started_at") or time.time())
    timestamp = state.get("timestamp") or time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
    duration = max(0.0, time.time() - started_at)
    placeholder = create_night_placeholder(device_id, timestamp, duration)
    if logger:
        logger.log(f"Night interval summarized in {placeholder}: {_format_duration(duration)}.", major=True)
    return placeholder


def create_night_placeholder(device_id, timestamp, duration_seconds):
    device_dir = os.path.join(CAPTURES_DIR, device_id)
    os.makedirs(device_dir, exist_ok=True)
    filename = f"night_{timestamp}.jpg"
    path = os.path.join(device_dir, filename)
    latest_capture = _latest_capture_file(os.listdir(device_dir))
    frame = cv2.imread(os.path.join(device_dir, latest_capture), cv2.IMREAD_COLOR) if latest_capture else None
    if frame is None:
        frame = cv2.UMat(720, 1280, cv2.CV_8UC3).get()
    else:
        frame = cv2.resize(frame, (1280, 720))
        frame = (frame * 0.12).astype("uint8")
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (1280, 720), (8, 14, 12), -1)
    frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
    lines = [
        "Night interval",
        f"{device_aliases.get(device_id, device_id)}",
        f"Duration: {_format_duration(duration_seconds)}",
        "Night frame collection disabled",
    ]
    y = 230
    for idx, text in enumerate(lines):
        scale = 1.7 if idx == 0 else 1.0
        thickness = 3 if idx == 0 else 2
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        x = max(24, (1280 - size[0]) // 2)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (188, 214, 198), thickness, cv2.LINE_AA)
        y += 78 if idx == 0 else 52
    cv2.imwrite(path, frame)
    return filename

def run_adb(cmd):
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )
        out, err = proc.communicate()
        return out.strip(), err.strip()
    except FileNotFoundError:
        return "", "adb-not-found"

def detect_connected_devices():
    adb_transport.reconnect_configured_async()
    return adb_transport.adb_devices()


def ensure_device_profile(device_id):
    profile = phone_profiles.get(device_id)
    if (
        profile
        and profile.get("save_folder")
        and profile.get("shutter_success")
        and profile.get("capabilities")
        and not profile.get("capabilities", {}).get("error")
    ):
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
        log_activity("No remote files found during sync.", device_id)
        state["initial_backup_complete"] = True
        _save_sync_state(local_device_dir, state)
        return

    latest_file = _latest_reference_file(remote_files) or remote_files[-1]

    if is_first_sync:
        logger.log(f"New device {device_id} detected. Starting full backup of {len(remote_files)} files.", major=True)
        log_activity(f"Starting first sync backup of {len(remote_files)} files.", device_id)
        backup_dir = os.path.join(local_device_dir, "backup")
        os.makedirs(backup_dir, exist_ok=True)

        for f in remote_files:
            _pull_remote_file(device_id, remote_dir, f, backup_dir)

        if _is_capture_image(latest_file):
            backup_latest = os.path.join(backup_dir, latest_file)
            if os.path.exists(backup_latest):
                shutil.copy2(backup_latest, os.path.join(local_device_dir, latest_file))
        logger.log(f"Initial sync complete. Backup in {backup_dir}.", major=True)
        log_activity("Initial sync complete.", device_id)
    else:
        local_files = sorted([f for f in os.listdir(local_device_dir) if _is_capture_image(f)])
        last_local = local_files[-1] if local_files else ""

        to_pull = [
            f for f in remote_files
            if _is_capture_image(f) and f > last_local
        ]
        if to_pull:
            logger.log(f"Differential sync: {len(to_pull)} new files found for {device_id}.", major=True)
            log_activity(f"Differential sync found {len(to_pull)} new file(s).", device_id)
            for f in to_pull:
                log_activity(f"Pulling {f} from device.", device_id)
                _pull_remote_file(device_id, remote_dir, f, local_device_dir)
        else:
            logger.log(f"No new files for {device_id}.", major=False)
            log_activity("No new files found during sync.", device_id)

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
    log_activity(f"Sync finished. Reference file {latest_file} remains on device.", device_id)


def capture_and_sync(device_id):
    lock = _device_lock(device_id)
    if not lock.acquire(blocking=False):
        print(f"[CAPTURE] {device_id} already has a capture in progress.")
        log_activity("Capture skipped because another capture is already running.", device_id)
        return False

    try:
        log_activity("Preparing Android capture profile.", device_id)
        ensure_device_profile(device_id)
        logger = PhoneLogger(device_id)
        local_device_dir = os.path.join(CAPTURES_DIR, device_id)
        if not _load_sync_state(local_device_dir).get("initial_backup_complete"):
            sync_device(device_id, logger)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"

        settings = effective_capture_settings(device_id)
        logger.log(
            f"Triggering capture: {filename} light={settings['active_light_mode']} mode={settings['light_mode']} luma={settings['latest_luminance']} collect_night={settings['collect_night_frames']} zoom={settings['zoom_percent']}% focus={settings['focus_mode']} white_balance={settings['white_balance']} antibanding={settings['antibanding']}",
            major=True,
        )
        log_activity(
            f"Launching camera capture {filename}; focus={settings['focus_mode']} white_balance={settings['white_balance']} antibanding={settings['antibanding']}.",
            device_id,
        )
        if settings["active_light_mode"] == "night_ir" and not settings["collect_night_frames"]:
            start_night_skip(device_id, timestamp)
            last_capture[device_id] = f"Night skipped: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            logger.log("Night capture skipped; waiting for daylight to write one summarized timelapse frame.", major=True)
            log_activity("Night capture skipped; waiting to summarize this night interval.", device_id)
            return True

        finish_night_skip(device_id, logger)
        log_activity("Sending capture command to phone.", device_id)
        if capture.capture_on_device(
            device_id,
            filename,
            zoom_percent=settings["zoom_percent"],
            delay=settings["delay_ms"],
            exposure=settings["exposure_compensation"],
            iso=settings["iso"],
            focus_mode=settings["focus_mode"],
            antibanding=settings["antibanding"],
            white_balance=settings["white_balance"],
        ):
            log_activity("Capture complete; starting device sync.", device_id)
            sync_device(device_id, logger)
            last_capture[device_id] = time.strftime("%Y-%m-%d %H:%M:%S")

            try:
                log_activity("Analyzing greenspace, ChArUco scale, color reference, and movement.", device_id)
                process_latest_captures(CAPTURES_DIR)
                log_activity("Analysis complete.", device_id)
            except Exception as e:
                print(f"[ANALYSIS] Error: {e}")
                log_activity(f"Analysis failed: {e}", device_id)

            log_activity("Updating timelapse video if enough new frames are available.", device_id)
            assemble_video(device_id)
            return True
        log_activity("Capture command failed or timed out.", device_id)
        return False
    finally:
        lock.release()


def capture_network_and_analyze(camera_id):
    lock = _device_lock(camera_id)
    if not lock.acquire(blocking=False):
        print(f"[NETWORK_CAMERA] {camera_id} already has a capture in progress.")
        log_activity("Network camera capture skipped because a capture is already running.", camera_id)
        return False
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        settings = effective_capture_settings(camera_id)
        if settings["active_light_mode"] == "night_ir" and not settings["collect_night_frames"]:
            start_night_skip(camera_id, timestamp)
            last_capture[camera_id] = f"Night skipped: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            log_activity("Night network capture skipped; waiting to summarize this interval.", camera_id)
            return True
        finish_night_skip(camera_id)
        log_activity(f"Requesting network camera frame {filename}.", camera_id)
        if capture_network_camera(camera_id, filename):
            last_capture[camera_id] = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                log_activity("Analyzing network frame greenspace, markers, color, and movement.", camera_id)
                process_latest_captures(CAPTURES_DIR)
                log_activity("Network frame analysis complete.", camera_id)
            except Exception as e:
                print(f"[ANALYSIS] Error: {e}")
                log_activity(f"Network frame analysis failed: {e}", camera_id)
            log_activity("Updating network camera timelapse if needed.", camera_id)
            assemble_video(camera_id)
            return True
        log_activity("Network camera did not provide a fresh frame.", camera_id)
        return False
    finally:
        lock.release()

def get_ffmpeg():
    # Attempt to find ffmpeg in common locations
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    candidates = [
        "ffmpeg",
        os.path.join(DATA_ROOT, "bin", "ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\pt\bin\ffmpeg.exe"
    ]
    for c in candidates:
        try:
            result = subprocess.run([c, "-version"], capture_output=True, text=True, creationflags=creationflags)
            if result.returncode == 0:
                return c
        except: continue
    return None


def _ffmpeg_concat_path(path):
    return os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")


def _render_video_from_frames(device_id, images, output_file):
    ffmpeg = get_ffmpeg()
    if not ffmpeg:
        return False, "FFmpeg not found."

    device_dir = os.path.join(CAPTURES_DIR, device_id)
    if not os.path.exists(device_dir):
        return False, f"No capture directory for {device_id}: {device_dir}"

    if len(images) < 2:
        return False, f"Need at least 2 frames; found {len(images)}."

    build_dir = os.path.join(VIDEOS_DIR, ".build")
    os.makedirs(build_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = _safe_video_id(device_id)
    list_file = os.path.join(build_dir, f"{safe_id}_{stamp}.txt")
    temp_output_file = os.path.join(build_dir, f"{safe_id}_{stamp}.tmp.mp4")
    with open(list_file, "w", encoding="utf-8") as f:
        for img in images:
            img_path = _ffmpeg_concat_path(os.path.join(device_dir, img))
            f.write(f"file '{img_path}'\n")
            f.write("duration 1.0\n" if _is_night_placeholder(img) else "duration 0.1\n")
        img_path = _ffmpeg_concat_path(os.path.join(device_dir, images[-1]))
        f.write(f"file '{img_path}'\n")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-vf",
        "scale=in_range=pc:out_range=tv,fps=10,format=yuv420p",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "24",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        temp_output_file,
    ]

    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=creationflags)
        if os.path.exists(temp_output_file) and os.path.getsize(temp_output_file) > 0:
            os.replace(temp_output_file, output_file)
            return True, result.stderr or "ok"
        return False, f"FFmpeg completed but output is missing or empty: {result.stderr}"
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    finally:
        if os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
            except OSError:
                pass
        if os.path.exists(list_file):
            try:
                os.remove(list_file)
            except OSError:
                pass


def _refresh_playback_mirror(device_id):
    current_file = _video_current_path(device_id)
    playback_file = _video_playback_path(device_id)
    if not os.path.exists(current_file):
        return False

    build_dir = os.path.join(VIDEOS_DIR, ".build")
    os.makedirs(build_dir, exist_ok=True)
    temp_playback = os.path.join(build_dir, f"{_safe_video_id(device_id)}_playback.tmp.mp4")
    try:
        shutil.copy2(current_file, temp_playback)
        os.replace(temp_playback, playback_file)
        video_manifest[device_id] = _video_playback_name(device_id)
        save_video_manifest()
        return True
    except OSError as e:
        print(f"[VIDEO] Playback mirror is locked for {device_id}; canonical video updated and mirror will catch up later: {e}")
        return False
    finally:
        if os.path.exists(temp_playback):
            try:
                os.remove(temp_playback)
            except OSError:
                pass


def assemble_video(device_id):
    with _video_lock(device_id):
        device_dir = os.path.join(CAPTURES_DIR, device_id)
        if not os.path.exists(device_dir):
            print(f"[VIDEO] No capture directory for {device_id}: {device_dir}")
            return

        images = _frame_files_for_device(device_id)
        if len(images) < 2:
            print(f"[VIDEO] Need at least 2 frames for {device_id}; found {len(images)} in {device_dir}.")
            return

        output_file = _video_current_path(device_id)
        playback_file = _video_playback_path(device_id)
        if os.path.exists(output_file) and len(images) % VIDEO_ASSEMBLY_FRAME_STEP != 0:
            if not os.path.exists(playback_file):
                _refresh_playback_mirror(device_id)
            print(f"[VIDEO] Skipping timelapse rebuild for {device_id}: {len(images)} frames; next update at a {VIDEO_ASSEMBLY_FRAME_STEP}-frame boundary.")
            return

        ok, message = _render_video_from_frames(device_id, images, output_file)
        if ok:
            mirror_ok = _refresh_playback_mirror(device_id)
            mirror_note = "playback mirror updated" if mirror_ok else "playback mirror deferred"
            print(f"[VIDEO] Updated timelapse video for {device_id}: {len(images)} frames -> {output_file} ({mirror_note})")
        else:
            print(f"[VIDEO] FFmpeg error for {device_id}: {message}")

def timelapse_loop():
    print("[TIMELAPSE] Starting background loop...")
    while True:
        time.sleep(timelapse_interval)
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

@app.route("/calibrate_multicam")
def run_multicam_calibration():
    selected = request.args.get("devices", "")
    devices = [d for d in selected.split(",") if d] if selected else detect_connected_devices() + configured_camera_ids()
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
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    data = request.get_json(silent=True)
    if isinstance(data, dict) and data.get("polygon"):
        calib_store.add_ignore_polygon(device_id, data["polygon"])
    elif isinstance(data, dict) and data.get("region"):
        calib_store.add_ignore_region(device_id, data["region"])
    else:
        calib_store.add_ignore_region(device_id, data)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "device": device_id, "stats": stats.get(device_id, {})})

@app.route("/clear_ignore/<device_id>")
def clear_ignore(device_id):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    calib_store.clear_ignore_regions(device_id)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "device": device_id, "stats": stats.get(device_id, {})})


@app.route("/segments")
def list_segments():
    return jsonify(segmentation_store.list())


@app.route("/segments/<device_id>", methods=["POST"])
def add_segment(device_id):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    data = request.get_json(silent=True) or {}
    region = data.get("region")
    polygon = data.get("polygon")
    if polygon:
        if not isinstance(polygon, list) or len(polygon) < 3:
            return jsonify({"status": "error", "message": "polygon must contain at least 3 points"}), 400
    elif not isinstance(region, list) or len(region) != 4:
        return jsonify({"status": "error", "message": "region must be [x1,y1,x2,y2]"}), 400
    name = (data.get("name") or f"{device_id} segment").strip()
    segment = segmentation_store.add(device_id, name, region=region, polygon=polygon)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "segment": segment, "stats": stats.get(device_id, {})})


@app.route("/segments/<device_id>", methods=["DELETE"])
def clear_segments(device_id):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    deleted = segmentation_store.clear(device_id) if hasattr(segmentation_store, "clear") else 0
    if not deleted:
        for segment in list(segmentation_store.list().get(device_id, [])):
            deleted += segmentation_store.delete(device_id, segment.get("id"))
    return jsonify({"status": "ok", "device": device_id, "deleted": deleted})


@app.route("/segments/<device_id>/<segment_id>", methods=["DELETE"])
def delete_segment(device_id, segment_id):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    deleted = segmentation_store.delete(device_id, segment_id)
    return jsonify({"status": "ok", "device": device_id, "segment_id": segment_id, "deleted": deleted})


@app.route("/manual_marker/<device_id>", methods=["POST"])
def add_manual_marker(device_id):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    data = request.get_json(silent=True) or {}
    corners = data.get("corners")
    if not isinstance(corners, list) or len(corners) != 4:
        return jsonify({"status": "error", "message": "corners must contain four [x,y] points"}), 400
    marker = calib_store.add_manual_marker(
        device_id,
        corners,
        float(data.get("size_mm") or 60.0),
        data.get("marker_id") or "manual",
    )
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "marker": marker, "stats": stats.get(device_id, {})})


@app.route("/manual_marker/<device_id>/clear", methods=["POST"])
def clear_manual_markers(device_id):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    calib_store.clear_manual_markers(device_id)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "device": device_id, "stats": stats.get(device_id, {})})


@app.route("/manual_markers")
def list_manual_markers():
    return jsonify(calib_store.data.get("manual_markers", {}))


@app.route("/manual_marker/<device_id>/<uid>", methods=["DELETE"])
def delete_manual_marker(device_id, uid):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    calib_store.delete_manual_marker(device_id, uid)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "device": device_id, "stats": stats.get(device_id, {})})


@app.route("/color_boards")
def list_color_boards():
    return jsonify(calib_store.data.get("color_boards", {}))


@app.route("/color_reference/<device_id>", methods=["GET", "POST"])
def color_reference(device_id):
    if request.method == "GET":
        return jsonify(calib_store.get_color_reference(device_id))
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)

    data = request.get_json(silent=True) or {}
    baseline = None
    if data.get("set_baseline"):
        stats = load_dashboard_stats()
        device_data = (stats.get(device_id) or {}).get("data") or {}
        color_boards = device_data.get("color_boards") or []
        if not any((board.get("patches") or []) for board in color_boards):
            return jsonify({"status": "error", "message": "No sampled color patches are available on the current frame."}), 400
        baseline = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": (stats.get(device_id) or {}).get("filename"),
            "color_boards": color_boards,
        }
    reference = calib_store.set_color_reference(
        device_id,
        enabled=data.get("enabled") if "enabled" in data else None,
        mode=data.get("mode") if "mode" in data else None,
        baseline=baseline,
    )
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "reference": reference, "stats": stats.get(device_id, {})})


@app.route("/color_board/<device_id>", methods=["POST"])
def add_color_patch(device_id):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    data = request.get_json(silent=True) or {}
    polygon = data.get("polygon") or []
    if not isinstance(polygon, list) or len(polygon) < 3:
        return jsonify({"status": "error", "message": "Color patches need at least three polygon points."}), 400
    patch = calib_store.add_color_patch(
        device_id,
        data.get("board_name") or "Color Board",
        data.get("patch_name") or "Patch",
        data.get("code") or "",
        data.get("role") or "",
        polygon,
    )
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "patch": patch, "device": device_id, "stats": stats.get(device_id, {})})


@app.route("/color_board/<device_id>/clear", methods=["POST"])
def clear_color_board(device_id):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    calib_store.clear_color_boards(device_id)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "device": device_id, "stats": stats.get(device_id, {})})


@app.route("/color_board/<device_id>/<board_name>/<uid>", methods=["DELETE"])
def delete_color_patch(device_id, board_name, uid):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    calib_store.delete_color_patch(device_id, board_name, uid)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "device": device_id, "stats": stats.get(device_id, {})})

@app.route("/marker_size", methods=["POST"])
def set_marker_size():
    import flask
    data = flask.request.json
    if data.get("device_id") and measurement_setup_locked(data.get("device_id")):
        return locked_setup_response(data.get("device_id"))
    # data: { marker_id, size_mm, device_id (optional) }
    calib_store.set_marker_size(data["marker_id"], data["size_mm"], data.get("device_id"))
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "stats": stats.get(data.get("device_id"), {}) if data.get("device_id") else {}})

@app.route("/charuco_target", methods=["POST"])
def set_charuco_target():
    data = request.get_json(silent=True) or {}
    device_id = data.get("device_id")
    if device_id and measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    calib_store.set_charuco_target(data, device_id)
    stats = process_latest_captures(CAPTURES_DIR)
    return jsonify({"status": "ok", "stats": stats.get(device_id, {}) if device_id else {}})


@app.route("/calibrate_aruco/<device_id>", methods=["POST"])
def calibrate_aruco(device_id):
    if measurement_setup_locked(device_id):
        return locked_setup_response(device_id)
    payload = request.get_json(silent=True) or {}
    operation_id = payload.get("operation_id")
    update_operation(operation_id, f"{device_id}: analyzing the latest frame for scale.", 10)
    stats = process_latest_captures(CAPTURES_DIR)
    info = stats.get(device_id) or {}
    data = info.get("data") or {}
    scale = data.get("scale_px_per_mm")
    if not scale:
        update_operation(operation_id, f"{device_id}: no valid marker scale was found.", 100, "error")
        return jsonify({"status": "error", "message": "No valid ArUco/ChArUco scale found for the selected device."}), 400

    update_operation(operation_id, f"{device_id}: detected {float(scale):.4f} px/mm from {data.get('scale_source') or 'marker geometry'}.", 65)
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
    else:
        saved = stats
    saved.setdefault(device_id, {}).update({
        "stable_scale_px_per_mm": float(scale),
        "stable_scale_source": data.get("scale_source") or "current_marker_detection",
        "stable_scale_calibrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=4)
    update_operation(operation_id, f"{device_id}: stable scale saved.", 100, "complete")
    return jsonify({"status": "ok", "device": device_id, "scale_px_per_mm": float(scale), "source": data.get("scale_source")})


@app.route("/operations/<operation_id>")
def get_operation(operation_id):
    snapshot = operation_snapshot(operation_id)
    if snapshot is None:
        return jsonify({"status": "missing", "progress": 0, "lines": []}), 404
    return jsonify(snapshot)


@app.route("/activity")
def get_activity():
    return jsonify({"lines": activity_snapshot()})


@app.route("/device_log/<device_id>")
def get_device_log(device_id):
    log_path = os.path.join(DEBUG_DIR, f"{device_id}.txt")
    if not os.path.exists(log_path):
        return jsonify({"device_id": device_id, "log": "No log yet."})
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        return jsonify({"device_id": device_id, "log": "".join(f.readlines()[-80:])})


def measurement_setup_locked(device_id):
    return bool(settings_for_device(device_id).get("measurement_locked"))


def locked_setup_response(device_id):
    return jsonify({
        "status": "locked",
        "message": "Measurement setup is locked. Unlock Setup before changing camera, calibration, segmentation, or color settings. Rotation stays editable because it is display-only.",
        "device": device_id,
    }), 423


@app.route("/lock_setup/<device_id>", methods=["POST"])
def lock_setup(device_id):
    payload = request.get_json(silent=True) or {}
    operation_id = payload.get("operation_id")
    lock = bool(payload.get("lock", True))
    if not lock:
        current = settings_for_device(device_id)
        current["measurement_locked"] = False
        current["measurement_locked_at"] = ""
        device_settings[device_id] = current
        save_device_settings()
        update_operation(operation_id, f"{device_id}: setup unlocked.", 100, "complete")
        return jsonify({"status": "ok", "settings": settings_response(device_id), "scale": None})

    update_operation(operation_id, f"{device_id}: checking marker geometry.", 10)
    stats = process_latest_captures(CAPTURES_DIR)
    info = stats.get(device_id) or {}
    data = info.get("data") or {}
    scale = data.get("scale_px_per_mm")
    if not scale:
        update_operation(operation_id, f"{device_id}: lock stopped because no valid scale is visible.", 100, "error")
        return jsonify({"status": "error", "message": "Cannot lock setup until a valid ArUco/ChArUco scale is visible."}), 400

    update_operation(operation_id, f"{device_id}: accepting scale {float(scale):.4f} px/mm.", 45)
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    saved = stats
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except (OSError, json.JSONDecodeError):
            saved = stats
    saved.setdefault(device_id, {}).update({
        "stable_scale_px_per_mm": float(scale),
        "stable_scale_source": data.get("scale_source") or "current_marker_detection",
        "stable_scale_calibrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=4)

    update_operation(operation_id, f"{device_id}: checking color-reference patches.", 65)
    sampled_boards = data.get("color_boards") or []
    native_patches = [
        patch
        for board in sampled_boards
        for patch in (board.get("patches") or [])
        if patch.get("source") == "charuco_native" and patch.get("mean_rgb")
    ]
    color_note = "manual color calibration remains available"
    if native_patches:
        baseline = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": info.get("filename"),
            "color_boards": sampled_boards,
            "source": "charuco_native",
        }
        calib_store.set_color_reference(device_id, enabled=True, mode="advanced", baseline=baseline)
        color_note = f"native ChArUco color baseline accepted from {len(native_patches)} patches"

    update_operation(operation_id, f"{device_id}: {color_note}.", 80)
    current = settings_for_device(device_id)
    current["measurement_locked"] = True
    current["measurement_locked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device_settings[device_id] = current
    save_device_settings()
    result = {
        "status": "ok",
        "settings": settings_response(device_id),
        "scale": float(scale),
        "scale_source": data.get("scale_source"),
        "native_color_patches": len(native_patches),
    }
    update_operation(operation_id, f"{device_id}: setup lock complete.", 100, "complete", result)
    return jsonify(result)

@app.route("/reconstruct")
def run_reconstruction():
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    if not os.path.exists(stats_file):
        return "No stats found", 404

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    device_masks = {}
    selected = request.args.get("devices", "")
    allowed_devices = set([d for d in selected.split(",") if d]) if selected else None
    for device_id, info in stats.items():
        if allowed_devices and device_id not in allowed_devices:
            continue
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
    network_stream_ids = {device_id for device_id in configured_camera_ids() if camera_has_live_stream(device_id)}
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

    return render_template_string(
        OBSERVATORY_DASHBOARD_HTML,
        devices=devices,
        last=last_capture,
        running=timelapse_running,
        logs=logs,
        network_stream_ids=network_stream_ids,
        device_aliases=device_aliases,
        device_metadata=device_metadata,
    )

@app.route("/video/<device_id>")
def serve_video(device_id):
    filename = video_manifest.get(device_id, _video_playback_name(device_id))
    if not os.path.exists(os.path.join(VIDEOS_DIR, filename)):
        current_path = _video_current_path(device_id)
        if os.path.exists(current_path):
            _refresh_playback_mirror(device_id)
        filename = _video_playback_name(device_id)
    response = send_from_directory(VIDEOS_DIR, filename, conditional=True)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route("/video_custom/<filename>")
def serve_custom_video(filename):
    if not filename.endswith(".mp4") or "/" in filename or "\\" in filename:
        return "Invalid video filename", 400
    response = send_from_directory(VIDEOS_DIR, filename, conditional=True)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route("/show_location/<kind>/<device_id>", methods=["POST"])
def show_location(kind, device_id):
    if kind == "video":
        path = _video_playback_path(device_id)
    else:
        path = os.path.join(CAPTURES_DIR, device_id)
    target = path if os.path.isdir(path) else os.path.dirname(path)
    if os.name == "nt" and os.path.exists(target):
        if kind == "video" and os.path.exists(path):
            subprocess.Popen(["explorer.exe", "/select,", path], creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            subprocess.Popen(["explorer.exe", target], creationflags=subprocess.CREATE_NO_WINDOW)
    return jsonify({"status": "ok", "path": path if kind == "video" else target})


def load_dashboard_stats():
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    stats = {}
    if os.path.exists(stats_file):
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
        except (OSError, json.JSONDecodeError):
            stats = {}

    for device_id in metric_store.list_devices():
        stats.setdefault(device_id, {})
        history = metric_store.history_for_device(device_id)
        if history:
            stats[device_id]["history"] = history
            latest = metric_store.latest_for_device(device_id)
            if latest:
                stats[device_id]["timestamp"] = latest.get("timestamp")
                stats[device_id]["filename"] = latest.get("filename")
                stats[device_id]["growth_rate_mm2_hr"] = latest.get("growth_rate_mm2_hr") or 0.0
                stats[device_id].setdefault("data", {})
                stats[device_id]["data"].update({
                    "plant_area_mm2": latest.get("area"),
                    "canopy_area_mm2": latest.get("area"),
                    "scale_px_per_mm": latest.get("scale"),
                    "scale_rejected": latest.get("scale_rejected"),
                    "canopy_coverage": latest.get("canopy_coverage"),
                    "color_metrics": latest.get("color_metrics") or {},
                    "color_correction": latest.get("color_correction") or {},
                    "movement": latest.get("movement") or {},
                })
    return stats


def _parse_local_datetime(value):
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


@app.route("/custom_timelapse/<device_id>", methods=["POST"])
def custom_timelapse(device_id):
    payload = request.get_json(silent=True) or request.form or {}
    start_dt = _parse_local_datetime(payload.get("start"))
    end_dt = _parse_local_datetime(payload.get("end"))
    if not start_dt or not end_dt:
        return jsonify({"status": "error", "message": "Choose a start and end date/time."}), 400
    if end_dt <= start_dt:
        return jsonify({"status": "error", "message": "End must be after start."}), 400

    frames = _frame_files_for_device(device_id, start_dt, end_dt)
    if len(frames) < 2:
        return jsonify({"status": "error", "message": f"Only {len(frames)} frame(s) found in that range."}), 400

    start_label = start_dt.strftime("%Y%m%d_%H%M")
    end_label = end_dt.strftime("%Y%m%d_%H%M")
    filename = _custom_video_name(device_id, start_label, end_label)
    output_file = os.path.join(VIDEOS_DIR, filename)
    ok, message = _render_video_from_frames(device_id, frames, output_file)
    if not ok:
        return jsonify({"status": "error", "message": message}), 500
    return jsonify({
        "status": "ok",
        "device": device_id,
        "frames": len(frames),
        "filename": filename,
        "url": f"/video_custom/{filename}",
        "path": output_file,
    })


@app.route("/reset_timelapse/<device_id>", methods=["POST"])
def reset_timelapse(device_id):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = os.path.join(DATA_ROOT, "archive")
    os.makedirs(archive_root, exist_ok=True)

    device_capture_dir = os.path.join(CAPTURES_DIR, device_id)
    if os.path.exists(device_capture_dir):
        archived_captures = os.path.join(archive_root, "captures", f"{device_id}_{stamp}")
        os.makedirs(os.path.dirname(archived_captures), exist_ok=True)
        shutil.move(device_capture_dir, archived_captures)
        os.makedirs(device_capture_dir, exist_ok=True)

    archived_video_dir = os.path.join(archive_root, "videos")
    for video_path in (_video_current_path(device_id), _video_playback_path(device_id)):
        if os.path.exists(video_path):
            os.makedirs(archived_video_dir, exist_ok=True)
            base, ext = os.path.splitext(os.path.basename(video_path))
            shutil.move(video_path, os.path.join(archived_video_dir, f"{base}_{stamp}{ext}"))
    if device_id in video_manifest:
        video_manifest[device_id] = _video_playback_name(device_id)
        save_video_manifest()

    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    if os.path.exists(stats_file):
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        if device_id in stats:
            stats[device_id]["archived_history"] = stats[device_id].get("history", [])
            stats[device_id]["history"] = []
            stats[device_id]["growth_rate_mm2_hr"] = 0.0
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4)
    metric_store.reset_device_history(device_id)

    return jsonify({"status": "ok", "device": device_id, "archive": os.path.join(archive_root, "captures", f"{device_id}_{stamp}")})


def _capture_timestamp(filename):
    match = re.match(r"capture_(\d{8})_(\d{6})\.", filename, re.IGNORECASE)
    if not match:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    raw = match.group(1) + match.group(2)
    return datetime.strptime(raw, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")


def _video_device_id(filename):
    stem = os.path.splitext(filename)[0]
    for suffix in ("_playback", "_current"):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _available_video_imports():
    if not os.path.exists(VIDEOS_DIR):
        return []
    videos = []
    for name in sorted(os.listdir(VIDEOS_DIR)):
        path = os.path.join(VIDEOS_DIR, name)
        if not os.path.isfile(path) or os.path.splitext(name)[1].lower() not in (".mp4", ".mov", ".avi", ".mkv"):
            continue
        videos.append({
            "name": name,
            "device_id": _video_device_id(name),
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 1),
            "modified": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S"),
        })
    return videos


def _import_video_frames(video_name, device_id=None, sample_seconds=60, max_frames=240):
    safe_name = os.path.basename(video_name)
    video_path = os.path.join(VIDEOS_DIR, safe_name)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {safe_name}")
    device_id = device_id or _video_device_id(safe_name)
    out_dir = os.path.join(CAPTURES_DIR, device_id)
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {safe_name}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = frame_count / fps if fps else 0
    if duration_seconds <= 0:
        duration_seconds = max_frames * sample_seconds
    step = max(1, int(sample_seconds * fps))
    base_time = datetime.fromtimestamp(os.path.getmtime(video_path)) - timedelta(seconds=duration_seconds)
    written = 0
    frame_idx = 0
    while written < max_frames and frame_idx < max(1, frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = base_time + timedelta(seconds=frame_idx / fps)
        filename = f"capture_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(out_dir, filename), frame)
        written += 1
        frame_idx += step
    cap.release()
    return {"device_id": device_id, "video": safe_name, "frames": written, "sample_seconds": sample_seconds}


def _import_video_job(video_name, device_id=None, sample_seconds=60, max_frames=240):
    try:
        metric_store.set_backfill_status(
            running=1,
            started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            finished_at=None,
            current_device=device_id or _video_device_id(video_name),
            processed=0,
            total=0,
            message=f"extracting frames from {video_name}",
        )
        result = _import_video_frames(video_name, device_id=device_id, sample_seconds=sample_seconds, max_frames=max_frames)
        log_activity(f"Imported {result['frames']} retroactive frames from {result['video']} for {result['device_id']}.")
        _backfill_metric_history()
    except Exception as exc:
        metric_store.set_backfill_status(
            running=0,
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message=f"video import failed: {exc}",
        )


def _dominant_frequency_hz(history):
    samples = []
    for entry in history or []:
        movement = entry.get("movement") or {}
        centroid = movement.get("centroid_px")
        value = None
        if isinstance(centroid, list) and len(centroid) == 2:
            value = float(centroid[1])
        elif movement.get("displacement_mm") is not None:
            value = float(movement.get("displacement_mm") or 0)
        timestamp = None
        try:
            timestamp = datetime.strptime(entry.get("timestamp") or "", "%Y-%m-%d %H:%M:%S").timestamp()
        except (TypeError, ValueError):
            pass
        if timestamp and value is not None and np.isfinite(value):
            samples.append((timestamp, value))
    if len(samples) < 8:
        return None
    samples.sort()
    times = np.asarray([item[0] for item in samples], dtype=np.float64)
    values = np.asarray([item[1] for item in samples], dtype=np.float64)
    span = float(times[-1] - times[0])
    if span < 2 * 3600:
        return None
    deltas = np.diff(times)
    step = float(np.median(deltas[deltas > 0])) if np.any(deltas > 0) else 0
    if step <= 0:
        return None
    step = max(60.0, min(step, 3600.0))
    uniform_times = np.arange(times[0], times[-1] + step, step)
    if len(uniform_times) < 8:
        return None
    uniform_values = np.interp(uniform_times, times, values)
    uniform_values = uniform_values - np.mean(uniform_values)
    amplitude = float(np.std(uniform_values))
    if amplitude <= 0:
        return None
    spectrum = np.abs(np.fft.rfft(uniform_values))
    freqs = np.fft.rfftfreq(len(uniform_values), d=step)
    min_freq = 1.0 / (48 * 3600)
    max_freq = 1.0 / (2 * 3600)
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(mask):
        return None
    masked_spectrum = spectrum[mask]
    masked_freqs = freqs[mask]
    index = int(np.argmax(masked_spectrum))
    freq = float(masked_freqs[index])
    strength = float(masked_spectrum[index] / max(1e-9, np.sum(masked_spectrum)))
    return {
        "frequency_hz": freq,
        "period_hours": (1.0 / freq) / 3600.0 if freq > 0 else None,
        "amplitude_px": amplitude,
        "strength": strength,
        "samples": len(samples),
        "span_hours": span / 3600.0,
        "fingerprint": f"{freq:.9g}Hz:{amplitude:.2f}px:{strength:.2f}",
    }


def _biology_state():
    stats = load_dashboard_stats()
    rhythms = {}
    alerts = []
    for device_id, info in stats.items():
        history = info.get("history") or []
        rhythm = _dominant_frequency_hz(history)
        if rhythm:
            rhythms[device_id] = rhythm
            midpoint = len(history) // 2
            if midpoint >= 8:
                old = _dominant_frequency_hz(history[:midpoint])
                new = _dominant_frequency_hz(history[midpoint:])
                if old and new and old.get("frequency_hz"):
                    shift = abs(new["frequency_hz"] - old["frequency_hz"]) / old["frequency_hz"]
                    if shift >= 0.25:
                        alerts.append({
                            "severity": "warn",
                            "type": "rhythm_shift",
                            "device_id": device_id,
                            "message": f"Circadian rhythm shifted {shift * 100:.0f}% ({old['frequency_hz']:.3g} Hz to {new['frequency_hz']:.3g} Hz).",
                        })
        recent = [entry for entry in history[-12:] if not entry.get("ignored")]
        baseline = [entry for entry in history[-60:-12] if not entry.get("ignored")]
        recent_green = [value for value in [
            ((entry.get("color_metrics") or {}).get("green_index"))
            for entry in recent
        ] if isinstance(value, (int, float))]
        baseline_green = [value for value in [
            ((entry.get("color_metrics") or {}).get("green_index"))
            for entry in baseline
        ] if isinstance(value, (int, float))]
        if recent_green and baseline_green:
            recent_avg = float(np.mean(recent_green))
            baseline_avg = float(np.mean(baseline_green))
            if baseline_avg and recent_avg < baseline_avg * 0.85:
                alerts.append({
                    "severity": "warn",
                    "type": "green_index_drop",
                    "device_id": device_id,
                    "message": f"Green index dropped {((baseline_avg - recent_avg) / abs(baseline_avg)) * 100:.0f}% from recent baseline.",
                })
        latest = (history[-1] if history else {}).get("movement") or {}
        previous = (history[-6] if len(history) >= 6 else {}).get("movement") or {}
        latest_centroid = latest.get("centroid_px")
        previous_centroid = previous.get("centroid_px")
        if isinstance(latest_centroid, list) and isinstance(previous_centroid, list) and len(latest_centroid) == 2 and len(previous_centroid) == 2:
            y_shift = float(latest_centroid[1]) - float(previous_centroid[1])
            speed = float(latest.get("speed_mm_hr") or 0)
            if y_shift > 10 and speed > 0.5:
                alerts.append({
                    "severity": "warn",
                    "type": "droop",
                    "device_id": device_id,
                    "message": f"Canopy centroid moved downward {y_shift:.0f}px with {speed:.2f} mm/hr movement speed.",
                })
    return {"rhythms": rhythms, "alerts": alerts[-100:]}


def _backfill_metric_history():
    from pt.core.analysis.image_analysis import analyze_image, is_capture_image

    jobs = []
    for device_id in sorted(os.listdir(CAPTURES_DIR)) if os.path.exists(CAPTURES_DIR) else []:
        device_path = os.path.join(CAPTURES_DIR, device_id)
        if not os.path.isdir(device_path):
            continue
        files = sorted(f for f in os.listdir(device_path) if is_capture_image(f))
        jobs.extend((device_id, device_path, filename) for filename in files)

    metric_store.set_backfill_status(
        running=1,
        started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        finished_at=None,
        current_device=None,
        processed=0,
        total=len(jobs),
        message="starting",
    )
    previous = {}
    processed = 0
    try:
        for device_id, device_path, filename in jobs:
            timestamp = _capture_timestamp(filename)
            metric_store.set_backfill_status(
                current_device=device_id,
                processed=processed,
                total=len(jobs),
                message=f"analyzing {filename}",
            )
            result = analyze_image(os.path.join(device_path, filename), device_id=device_id)
            if result:
                area = float(result.get("plant_area_mm2") or 0)
                growth = 0.0
                movement = {
                    "centroid_px": (result.get("canopy_bounding_box") or {}).get("center"),
                    "displacement_mm": 0.0,
                    "speed_mm_hr": 0.0,
                }
                prev = previous.get(device_id)
                if prev:
                    dt = (
                        datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                        - datetime.strptime(prev["timestamp"], "%Y-%m-%d %H:%M:%S")
                    ).total_seconds() / 3600.0
                    if dt > 0.01:
                        growth = (area - prev["area"]) / dt
                        active_scale = float(result.get("scale_px_per_mm") or 0)
                        current_centroid = movement.get("centroid_px")
                        previous_centroid = prev.get("centroid_px")
                        if current_centroid and previous_centroid and active_scale > 0:
                            displacement_px = float(np.linalg.norm(
                                np.asarray(current_centroid, dtype=np.float32)
                                - np.asarray(previous_centroid, dtype=np.float32)
                            ))
                            movement["displacement_mm"] = displacement_px / active_scale
                            movement["speed_mm_hr"] = movement["displacement_mm"] / dt
                entry = {
                    "timestamp": timestamp,
                    "filename": filename,
                    "scale": float(result["scale_px_per_mm"]) if result.get("scale_px_per_mm") else None,
                    "detected_scale": float(result["scale_px_per_mm"]) if result.get("scale_px_per_mm") else None,
                    "scale_rejected": bool(result.get("scale_rejected")),
                    "area": area,
                    "growth_rate_mm2_hr": growth,
                    "segments": result.get("segments", []),
                    "canopy_coverage": result.get("canopy_coverage"),
                    "color_metrics": result.get("color_metrics"),
                    "movement": movement,
                    "nutrient_deficiency": {},
                }
                metric_store.upsert_history_point(device_id, entry)
                previous[device_id] = {"timestamp": timestamp, "area": area, "centroid_px": movement.get("centroid_px")}
            processed += 1
        for device_id in {job[0] for job in jobs}:
            metric_store.refresh_rollups(device_id)
        metric_store.set_backfill_status(
            running=0,
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            processed=processed,
            total=len(jobs),
            current_device=None,
            message="complete",
        )
    except Exception as exc:
        metric_store.set_backfill_status(
            running=0,
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            processed=processed,
            total=len(jobs),
            message=f"failed: {exc}",
        )


@app.route("/metrics/clear", methods=["POST"])
def clear_metric_history():
    rows = metric_store.clear_all_history()
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    if os.path.exists(stats_file):
        os.remove(stats_file)
    return jsonify({"status": "ok", "cleared_rows": rows, "message": "Derived metric history cleared; raw captures were kept."})


@app.route("/metrics/backfill", methods=["POST"])
def start_metric_backfill():
    status = metric_store.get_backfill_status()
    if status.get("running"):
        return jsonify(status)
    threading.Thread(target=_backfill_metric_history, daemon=True).start()
    return jsonify({"status": "ok", "message": "Backfill started"})


@app.route("/metrics/backfill")
def metric_backfill_status():
    return jsonify(metric_store.get_backfill_status())


@app.route("/video_imports")
def list_video_imports():
    return jsonify({"videos": _available_video_imports()})


@app.route("/video_imports/import", methods=["POST"])
def import_video_for_analysis():
    data = request.get_json(silent=True) or {}
    video_name = os.path.basename(str(data.get("video") or ""))
    if not video_name:
        return jsonify({"status": "error", "message": "Choose a video to import."}), 400
    status = metric_store.get_backfill_status()
    if status.get("running"):
        return jsonify(status), 409
    try:
        sample_seconds = int(max(1, min(3600, int(data.get("sample_seconds") or 60))))
        max_frames = int(max(1, min(2000, int(data.get("max_frames") or 240))))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Sample seconds and max frames must be numbers."}), 400
    device_id = str(data.get("device_id") or _video_device_id(video_name)).strip()
    threading.Thread(target=_import_video_job, args=(video_name, device_id, sample_seconds, max_frames), daemon=True).start()
    return jsonify({"status": "ok", "message": f"Import started for {video_name}", "device_id": device_id})


@app.route("/stats")
def get_stats():
    return jsonify(load_dashboard_stats())


@app.route("/device_aliases")
def get_device_aliases():
    return jsonify(device_aliases)


@app.route("/device_aliases/<device_id>", methods=["POST"])
def update_device_alias(device_id):
    data = request.get_json(silent=True) or {}
    alias = str(data.get("alias") or "").strip()
    if alias:
        device_aliases[device_id] = alias[:120]
    else:
        device_aliases.pop(device_id, None)
    save_device_aliases()
    return jsonify({"status": "ok", "device": device_id, "alias": device_aliases.get(device_id, "")})


@app.route("/device_metadata")
def get_device_metadata():
    return jsonify(device_metadata)


@app.route("/device_metadata/<device_id>", methods=["POST"])
def update_device_metadata(device_id):
    data = request.get_json(silent=True) or {}
    current = dict(device_metadata.get(device_id, {}))
    if "chamber" in data:
        current["chamber"] = str(data.get("chamber") or "").strip()[:80]
    if "role" in data:
        role = str(data.get("role") or "").strip()
        current["role"] = role if role in ("tray", "wall", "overview", "night_vision", "calibration", "other") else "other"
    if "notes" in data:
        current["notes"] = str(data.get("notes") or "").strip()[:240]
    device_metadata[device_id] = current
    save_device_metadata()
    return jsonify({"status": "ok", "device": device_id, "metadata": current})


@app.route("/ignore_growth_point/<device_id>", methods=["POST"])
def ignore_growth_point(device_id):
    data = request.get_json(silent=True) or {}
    metric_changed = metric_store.ignore_point(
        device_id,
        timestamp=data.get("timestamp"),
        filename=data.get("filename"),
        segment_id=data.get("segment_id"),
    )
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    if not os.path.exists(stats_file):
        return jsonify({"status": "ok", "changed": bool(metric_changed)})
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)
    history = (stats.get(device_id) or {}).get("history", [])
    timestamp = data.get("timestamp")
    filename = data.get("filename")
    segment_id = data.get("segment_id")
    changed = False
    for entry in history:
        if (timestamp and entry.get("timestamp") == timestamp) or (filename and entry.get("filename") == filename):
            if segment_id:
                ignored = set(entry.get("ignored_segments", []))
                ignored.add(segment_id)
                entry["ignored_segments"] = sorted(ignored)
            else:
                entry["ignored"] = True
            changed = True
    if changed:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
    return jsonify({"status": "ok", "changed": changed or bool(metric_changed)})


@app.route("/delete_growth_point/<device_id>", methods=["POST"])
def delete_growth_point(device_id):
    data = request.get_json(silent=True) or {}
    metric_changed = metric_store.delete_point(
        device_id,
        timestamp=data.get("timestamp"),
        filename=data.get("filename"),
        segment_id=data.get("segment_id"),
    )
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    if not os.path.exists(stats_file):
        return jsonify({"status": "ok", "changed": bool(metric_changed)})
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)
    history = (stats.get(device_id) or {}).get("history", [])
    timestamp = data.get("timestamp")
    filename = data.get("filename")
    segment_id = data.get("segment_id")
    changed = False
    if segment_id:
        for entry in history:
            if (timestamp and entry.get("timestamp") == timestamp) or (filename and entry.get("filename") == filename):
                before = len(entry.get("segments", []))
                entry["segments"] = [s for s in entry.get("segments", []) if s.get("id") != segment_id]
                changed = changed or len(entry["segments"]) != before
    else:
        remaining = [
            entry for entry in history
            if not ((timestamp and entry.get("timestamp") == timestamp) or (filename and entry.get("filename") == filename))
        ]
        changed = len(remaining) != len(history)
        if device_id in stats:
            stats[device_id]["history"] = remaining
    if changed:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
    return jsonify({"status": "ok", "changed": changed or bool(metric_changed)})


@app.route("/device_settings")
def get_device_settings():
    devices = detect_connected_devices() + configured_camera_ids()
    known = sorted(set(devices) | set(device_settings.keys()))
    return jsonify({device_id: settings_response(device_id) for device_id in known})


@app.route("/adb_transport")
def get_adb_transport():
    return jsonify(adb_transport.transport_status())


@app.route("/adb_transport/connect", methods=["POST"])
def connect_adb_transport():
    data = request.get_json(silent=True) or {}
    try:
        endpoint = adb_transport.normalize_endpoint(data.get("endpoint"), data.get("port", 5555))
        result = adb_transport.connect(endpoint)
        if result["ok"] and data.get("remember", True):
            adb_transport.remember_endpoint(endpoint, data.get("name", ""))
        return jsonify(result), (200 if result["ok"] else 502)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.route("/adb_transport/disconnect", methods=["POST"])
def disconnect_adb_transport():
    data = request.get_json(silent=True) or {}
    try:
        result = adb_transport.disconnect(data.get("endpoint"))
        if data.get("forget"):
            adb_transport.forget_endpoint(data.get("endpoint"))
        return jsonify(result), (200 if result["ok"] else 502)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.route("/adb_transport/pair", methods=["POST"])
def pair_adb_transport():
    data = request.get_json(silent=True) or {}
    try:
        result = adb_transport.pair(data.get("endpoint"), data.get("pairing_code"))
        return jsonify(result), (200 if result["ok"] else 502)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.route("/adb_transport/prepare_legacy", methods=["POST"])
def prepare_legacy_adb_transport():
    data = request.get_json(silent=True) or {}
    try:
        result = adb_transport.prepare_legacy_wifi(data.get("serial"), data.get("port", 5555))
        if result["ok"] and data.get("remember", True):
            adb_transport.remember_endpoint(result["endpoint"], data.get("name", ""))
        return jsonify(result), (200 if result["ok"] else 502)
    except (TypeError, ValueError) as exc:
        return jsonify({"ok": False, "message": str(exc)}), 400


@app.route("/adb_transport/reconnect", methods=["POST"])
def reconnect_adb_transports():
    results = adb_transport.reconnect_configured(force=True)
    return jsonify({"ok": all(result.get("ok") for result in results), "results": results})


@app.route("/device_capabilities")
def get_device_capabilities():
    return jsonify({
        device_id: (profile or {}).get("capabilities", {})
        for device_id, profile in phone_profiles.items()
    })


@app.route("/network_camera_status")
def get_network_camera_status():
    probe = request.args.get("probe", "0") == "1"
    return jsonify({
        camera_id: network_camera_status(camera_id, probe=probe)
        for camera_id in configured_camera_ids()
    })


@app.route("/experiments", methods=["GET", "POST"])
def experiments_api():
    if request.method == "GET":
        return jsonify({
            "experiments": load_experiments(),
            "events": load_events(),
            "plants": load_plant_records(),
            "biology": _biology_state(),
        })
    try:
        return jsonify({"status": "ok", "experiment": upsert_experiment(request.get_json(silent=True) or {})})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/experiment_events", methods=["POST"])
def experiment_events_api():
    try:
        return jsonify({"status": "ok", "event": add_event(request.get_json(silent=True) or {})})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/plant_records", methods=["GET", "POST"])
def plant_records_api():
    if request.method == "GET":
        return jsonify({"plants": load_plant_records()})
    try:
        return jsonify({"status": "ok", "plant": upsert_plant_record(request.get_json(silent=True) or {})})
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/biology_state")
def biology_state_api():
    return jsonify(_biology_state())


@app.route("/device_settings/<device_id>", methods=["POST"])
def update_device_settings(device_id):
    data = request.get_json(silent=True) or {}
    current = settings_for_device(device_id)
    if current.get("measurement_locked"):
        allowed_locked_keys = {"display_rotation_deg"}
        unlocking = set(data.keys()) == {"measurement_locked"} and data.get("measurement_locked") is False
        visual_only = set(data.keys()).issubset(allowed_locked_keys)
        if not unlocking and not visual_only:
            return jsonify({
                "status": "locked",
                "message": "Measurement setup is locked. Unlock Setup before changing camera, calibration, segmentation, or color settings. Rotation stays editable because it is display-only.",
                "settings": settings_response(device_id),
            }), 423
    if data.get("light_mode") in ("auto", "day", "night_ir"):
        current["light_mode"] = data["light_mode"]
    if data.get("profile_name") in ("auto", "day", "wide_day", "night_ir"):
        current["profile_name"] = data["profile_name"]
    if "zoom_percent" in data:
        current["zoom_percent"] = int(max(0, min(100, int(data["zoom_percent"]))))
    if "delay_ms" in data:
        current["delay_ms"] = int(max(500, min(15000, int(data["delay_ms"]))))
    if "exposure_compensation" in data:
        current["exposure_compensation"] = int(max(-12, min(12, int(data["exposure_compensation"]))))
    if "display_rotation_deg" in data:
        current["display_rotation_deg"] = float(data["display_rotation_deg"]) % 360.0
    if "collect_night_frames" in data:
        current["collect_night_frames"] = bool(data["collect_night_frames"])
    if "measurement_locked" in data:
        current["measurement_locked"] = bool(data["measurement_locked"])
        current["measurement_locked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if current["measurement_locked"] else ""
    if "expected_marker_count" in data:
        try:
            current["expected_marker_count"] = int(max(1, min(50, int(data["expected_marker_count"]))))
        except (TypeError, ValueError):
            pass
    if str(data.get("iso")) in ("auto", "100", "200", "400", "800", "1600"):
        current["iso"] = str(data["iso"])
    if data.get("focus_mode") in ("continuous-picture", "auto", "infinity", "macro", "fixed"):
        current["focus_mode"] = data["focus_mode"]
    if data.get("antibanding") in ("off", "50hz", "60hz", "auto"):
        current["antibanding"] = data["antibanding"]
    if data.get("white_balance") in ("auto", "daylight", "cloudy-daylight", "fluorescent", "incandescent", "shade", "twilight", "warm-fluorescent"):
        current["white_balance"] = data["white_balance"]
    device_settings[device_id] = current
    save_device_settings()
    return jsonify(settings_response(device_id))


def latest_marker_count(device_id):
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    if not os.path.exists(stats_file):
        return 0
    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return int(((stats.get(device_id) or {}).get("data") or {}).get("markers_found") or 0)
    except (OSError, ValueError, TypeError):
        return 0


def latest_analysis_record(device_id):
    stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
    if not os.path.exists(stats_file):
        return {}
    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return stats.get(device_id) or {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def latest_capture_path(device_id):
    device_dir = os.path.join(CAPTURES_DIR, device_id)
    if not os.path.exists(device_dir):
        return None
    latest = _latest_capture_file(os.listdir(device_dir))
    return os.path.join(device_dir, latest) if latest else None


def image_quality_metrics(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR) if image_path else None
    if img is None:
        return {"sharpness": 0.0, "brightness": 0.0, "clipped_fraction": 1.0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    clipped = float(((gray <= 4) | (gray >= 251)).mean())
    return {"sharpness": sharpness, "brightness": brightness, "clipped_fraction": clipped}


def score_measurement_trial(analysis, quality, expected_markers, previous_trial=None):
    data = analysis.get("data") or {}
    markers = int(data.get("markers_found") or 0)
    marker_score = min(markers / max(1, expected_markers), 1.0)

    sharpness = float(quality.get("sharpness") or 0.0)
    sharpness_score = min(sharpness / 260.0, 1.0)

    scale = data.get("scale_px_per_mm")
    scale_score = 0.0
    if scale not in (None, 0):
        scale_score = 0.55
        if not data.get("scale_rejected"):
            scale_score += 0.25
        if previous_trial and previous_trial.get("scale"):
            ratio = min(float(scale), float(previous_trial["scale"])) / max(float(scale), float(previous_trial["scale"]))
            scale_score += max(0.0, min(0.2, (ratio - 0.75) / 0.25 * 0.2))

    color = data.get("color_metrics") or {}
    green_index = color.get("green_index")
    canopy = float(data.get("canopy_area_mm2") or data.get("plant_area_mm2") or 0.0)
    green_score = 0.0
    if green_index is not None and canopy > 0:
        green_score = min(max(float(green_index), 0.0) / 0.18, 1.0)

    brightness = float(quality.get("brightness") or 0.0)
    clipped = float(quality.get("clipped_fraction") or 0.0)
    exposure_score = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0)
    exposure_score *= max(0.0, 1.0 - clipped * 8.0)

    score = (
        marker_score * 0.40
        + sharpness_score * 0.20
        + scale_score * 0.20
        + green_score * 0.10
        + exposure_score * 0.10
    )
    return {
        "score": round(float(score), 4),
        "markers": markers,
        "expected_markers": expected_markers,
        "marker_score": round(float(marker_score), 4),
        "sharpness": round(sharpness, 2),
        "sharpness_score": round(float(sharpness_score), 4),
        "scale": float(scale) if scale not in (None, 0) else None,
        "scale_score": round(float(scale_score), 4),
        "scale_rejected": bool(data.get("scale_rejected")),
        "green_index": float(green_index) if green_index is not None else None,
        "green_score": round(float(green_score), 4),
        "canopy_area_mm2": canopy,
        "brightness": round(brightness, 2),
        "clipped_fraction": round(clipped, 5),
        "exposure_score": round(float(exposure_score), 4),
    }


@app.route("/auto_tune_tags/<device_id>", methods=["POST"])
def auto_tune_tags(device_id):
    if device_id in configured_camera_ids():
        return jsonify({"status": "error", "message": "Automatic camera-setting sweep is for Android ADB cameras."}), 400

    payload = request.get_json(silent=True) or {}
    operation_id = payload.get("operation_id")
    original = dict(settings_for_device(device_id))
    try:
        expected_markers = int(payload.get("expected_markers") or original.get("expected_marker_count") or 4)
    except (TypeError, ValueError):
        expected_markers = 4
    expected_markers = int(max(1, min(50, expected_markers)))
    update_operation(operation_id, f"{device_id}: starting measurement-profile calibration for {expected_markers} expected tags.", 2)
    profiles = [
        {"delay_ms": 5000, "antibanding": "60hz", "focus_mode": "continuous-picture", "exposure_compensation": 0, "iso": "auto", "white_balance": "daylight", "zoom_percent": original.get("zoom_percent", 0)},
        {"delay_ms": 7000, "antibanding": "60hz", "focus_mode": "auto", "exposure_compensation": 0, "iso": "200", "white_balance": "daylight", "zoom_percent": original.get("zoom_percent", 0)},
        {"delay_ms": 8000, "antibanding": "60hz", "focus_mode": "macro", "exposure_compensation": 0, "iso": "200", "white_balance": "daylight", "zoom_percent": original.get("zoom_percent", 0)},
        {"delay_ms": 9000, "antibanding": "50hz", "focus_mode": "auto", "exposure_compensation": 1, "iso": "400", "white_balance": "daylight", "zoom_percent": original.get("zoom_percent", 0)},
        {"delay_ms": 9000, "antibanding": "50hz", "focus_mode": "auto", "exposure_compensation": 1, "iso": "400", "white_balance": "daylight", "zoom_percent": original.get("zoom_percent", 0)},
        {"delay_ms": 10000, "antibanding": "60hz", "focus_mode": "continuous-picture", "exposure_compensation": -1, "iso": "auto", "white_balance": "daylight", "zoom_percent": original.get("zoom_percent", 0)},
        {"delay_ms": 10000, "antibanding": "auto", "focus_mode": "auto", "exposure_compensation": 0, "iso": "400", "white_balance": "cloudy-daylight", "zoom_percent": original.get("zoom_percent", 0)},
        {"delay_ms": 11000, "antibanding": "60hz", "focus_mode": "infinity", "exposure_compensation": 0, "iso": "400", "white_balance": "daylight", "zoom_percent": original.get("zoom_percent", 0)},
        {"delay_ms": 12000, "antibanding": "off", "focus_mode": "continuous-picture", "exposure_compensation": 0, "iso": "800", "white_balance": "daylight", "zoom_percent": original.get("zoom_percent", 0)},
    ]
    results = []
    best = {"score": -1.0, "profile": original, "score_detail": {}}
    previous_score = None

    for idx, profile in enumerate(profiles, start=1):
        update_operation(
            operation_id,
            f"{device_id}: trial {idx}/{len(profiles)} - focus {profile['focus_mode']}, exposure {profile['exposure_compensation']}, ISO {profile['iso']}, {profile['antibanding']}.",
            5 + int((idx - 1) / len(profiles) * 85),
        )
        current = dict(original)
        current.update(profile)
        current["light_mode"] = "day"
        current["profile_name"] = "day"
        current["expected_marker_count"] = expected_markers
        current["measurement_locked"] = False
        current["measurement_locked_at"] = ""
        device_settings[device_id] = current
        save_device_settings()
        ok = capture_and_sync(device_id)
        analysis = latest_analysis_record(device_id) if ok else {}
        quality = image_quality_metrics(latest_capture_path(device_id)) if ok else {}
        score_detail = score_measurement_trial(analysis, quality, expected_markers, previous_score)
        trial = {"trial": idx, "ok": bool(ok), "profile": profile, **score_detail}
        results.append(trial)
        update_operation(
            operation_id,
            f"{device_id}: trial {idx} {'captured' if ok else 'failed'} - {score_detail['markers']}/{expected_markers} tags, score {round(score_detail['score'] * 100)}%, sharpness {score_detail['sharpness']}.",
            5 + int(idx / len(profiles) * 85),
        )
        previous_score = score_detail
        if score_detail["score"] > best["score"]:
            best = {"score": score_detail["score"], "profile": current, "score_detail": score_detail}

    winning = best["profile"] if best["score"] > 0 else original
    winning["expected_marker_count"] = expected_markers
    winning["measurement_profile_score"] = best["score"]
    winning["measurement_profile_summary"] = (
        f"{best['score_detail'].get('markers', 0)}/{expected_markers} tags, "
        f"sharpness {best['score_detail'].get('sharpness', 0)}, "
        f"scale {best['score_detail'].get('scale') or '--'}, "
        f"green {best['score_detail'].get('green_index') if best['score_detail'].get('green_index') is not None else '--'}"
    )
    winning["measurement_locked"] = bool(payload.get("lock", True)) and best["score"] > 0
    winning["measurement_locked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if winning["measurement_locked"] else ""
    device_settings[device_id] = winning
    save_device_settings()
    accepted_scale = best["score_detail"].get("scale")
    if winning["measurement_locked"] and accepted_scale:
        stats_file = os.path.join(DATA_ROOT, "plant_stats.json")
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                saved_stats = json.load(f)
        except (OSError, json.JSONDecodeError):
            saved_stats = {}
        saved_stats.setdefault(device_id, {}).update({
            "stable_scale_px_per_mm": float(accepted_scale),
            "stable_scale_source": "measurement_profile_sweep",
            "stable_scale_calibrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(saved_stats, f, indent=4)
        update_operation(operation_id, f"{device_id}: accepted stable scale {float(accepted_scale):.4f} px/mm from the winning trial.", 96)
    response = {
        "status": "ok",
        "device": device_id,
        "expected_markers": expected_markers,
        "best_score": best["score"],
        "best_markers": best["score_detail"].get("markers", 0),
        "best_profile": settings_response(device_id),
        "best_detail": best["score_detail"],
        "trials": results,
    }
    update_operation(
        operation_id,
        f"{device_id}: calibration complete. Best score {round(best['score'] * 100)}% with {best['score_detail'].get('markers', 0)}/{expected_markers} tags.",
        100,
        "complete",
        response,
    )
    return jsonify(response)

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


@app.route("/live_stream/<camera_id>")
def live_stream(camera_id):
    global active_live_stream_device
    if camera_id not in configured_camera_ids() or not camera_has_live_stream(camera_id):
        return "No live stream configured", 404

    if not live_stream_lock.acquire(blocking=False):
        return "Another live stream is already active", 409
    active_live_stream_device = camera_id

    def generate():
        global active_live_stream_device
        try:
            yield from mjpeg_live_frames(camera_id)
        finally:
            active_live_stream_device = None
            live_stream_lock.release()

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


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
            <video id="vid-{{d}}" controls loop muted preload="none" data-src="/video/{{d}}"></video>

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
            if (document.hidden) return;
            document.querySelectorAll('img').forEach(img => {
                const base = img.src.split('?')[0];
                img.src = base + '?t=' + Date.now();
            });
            updateStats();
        }, 30000);
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) document.querySelectorAll('video').forEach(v => v.pause());
        });
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
        .mission-image-wrap { aspect-ratio: 1 / 1; border-radius: 50%; overflow: visible; background: radial-gradient(circle, #0b100f 0 64%, transparent 65%); border: 1px solid #26302c; display: grid; place-items: center; margin: 18px auto; width: min(100%, 520px); }
        .mission-image-wrap img { width: auto; height: auto; max-width: 86%; max-height: 86%; object-fit: contain; border: 1px solid #26302c; border-radius: 6px; transform: rotate(var(--rotation, 0deg)); transform-origin: 50% 50%; box-shadow: 0 10px 30px rgba(0,0,0,.35); }
        .mission-image-wrap .calib-overlay { transform: rotate(var(--rotation, 0deg)); transform-origin: 50% 50%; }
        .rotation-row { display: grid; grid-template-columns: 64px 1fr 74px; align-items: center; gap: 8px; margin-top: 10px; }
        .rotation-row input[type="number"] { min-height: 32px; border-radius: 6px; border: 1px solid #3a4541; background: #101514; color: var(--text); padding: 0 8px; }
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
        .setup-control-card { border-top: 1px solid #26302c; margin-top: 12px; padding-top: 12px; }
        .setup-workbench { display: grid; grid-template-columns: minmax(220px, .55fr) minmax(360px, 1.05fr) minmax(300px, .75fr); gap: 14px; align-items: start; margin-top: 14px; }
        .setup-stage { display: grid; gap: 12px; }
        .setup-mode-tabs { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; margin-bottom: 10px; }
        .setup-mode-tabs button { min-height: 40px; padding: 6px 8px; }
        .setup-mode-tabs button.active { border-color: var(--green); background: #18231e; color: var(--text); }
        .setup-panel { display: none; border: 1px solid #26302c; border-radius: 8px; padding: 12px; background: var(--panel-2); }
        .setup-panel.active { display: block; }
        .setup-panel h3 { margin: 0 0 8px; }
        .setup-summary { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
        .setup-summary div { border: 1px solid #26302c; border-radius: 6px; padding: 9px; background: #0d1311; }
        .setup-summary span { display: block; color: var(--muted); font-size: .72rem; text-transform: uppercase; }
        .setup-summary strong { display: block; margin-top: 4px; overflow-wrap: anywhere; }
        .lock-notice { display: none; border: 1px solid var(--amber); background: rgba(255, 191, 87, .12); border-radius: 8px; padding: 10px; margin: 10px 0; color: var(--text); }
        .lock-notice.active { display: block; }
        .locked-control { opacity: .58; }
        .setup-control-card.locked input:disabled, .setup-control-card.locked select:disabled, .locked-control:disabled { cursor: not-allowed; }
        .setup-checklist { display: grid; gap: 8px; }
        .setup-checklist div { display: flex; justify-content: space-between; gap: 10px; border-top: 1px solid #26302c; padding-top: 8px; }
        .setup-checklist .ok { color: var(--green); font-weight: 800; }
        .setup-checklist .warn { color: var(--amber); font-weight: 800; }
        select, input[type="range"] { width: 100%; accent-color: var(--green); }
        select { min-height: 34px; border-radius: 6px; border: 1px solid #3a4541; background: #101514; color: var(--text); padding: 0 8px; }
        .telemetry { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
        .telemetry.mission-canopy-only { grid-template-columns: 1fr; }
        .telemetry div { background: var(--panel-2); border: 1px solid #222c28; border-radius: 6px; padding: 10px; min-height: 58px; }
        .telemetry span { display: block; color: var(--muted); font-size: .72rem; text-transform: uppercase; }
        .telemetry strong { display: block; margin-top: 5px; overflow-wrap: anywhere; }
        .chart-wrap { min-height: 320px; height: 320px; padding-bottom: 8px; }
        .chart-wrap.tall { min-height: 420px; height: 420px; }
        .chart-field { display: grid; gap: 6px; align-content: end; }
        .chart-field span { color: var(--muted); font-size: .74rem; }
        .chart-field.inline { grid-template-columns: 82px 1fr 74px; align-items: center; }
        .chart-field input[type="number"] { min-width: 0; }
        .chart-field .range-value { text-align: right; color: var(--text); font-weight: 800; overflow-wrap: anywhere; }
        .custom-window { display: grid; grid-template-columns: minmax(70px, 1fr) minmax(86px, 1fr); gap: 6px; }
        .chart-slider-row, .chart-input-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; grid-column: 1 / -1; }
        .chart-input-row { border-top: 1px solid #26302c; padding-top: 8px; margin-top: 2px; }
        .point-popover { position: fixed; z-index: 30; min-width: 210px; background: #121816; border: 1px solid #495650; border-radius: 8px; padding: 12px; box-shadow: 0 18px 50px rgba(0,0,0,.45); }
        .point-popover .value { font-weight: 800; margin: 6px 0 10px; }
        canvas { width: 100% !important; height: 100% !important; }
        .segment-box {
            position: absolute;
            border: 2px solid var(--cyan);
            background: rgba(119, 183, 197, .12);
            color: #fff;
            pointer-events: none;
            font-size: .75rem;
            font-weight: 800;
            padding: 2px 5px;
            text-shadow: 0 1px 2px #000;
        }
        .draw-segment-active { outline: 2px solid var(--cyan); cursor: crosshair; }
        .live-active { outline: 2px solid var(--green); }
        .split-grid { display: grid; grid-template-columns: minmax(300px, .9fr) minmax(320px, 1.1fr); gap: 16px; align-items: start; }
        .device-list { display: grid; gap: 8px; }
        .device-list button { text-align: left; }
        .device-list button.active { border-color: var(--green); background: #18231e; }
        .seg-camera-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-top: 12px; }
        .seg-camera-card { border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: var(--panel-2); }
        .manual-crop { display: none; width: 100%; min-height: 260px; border: 1px solid #73562b; border-radius: 6px; background: #050706; margin-top: 10px; cursor: crosshair; }
        .modal-backdrop { position: fixed; inset: 0; z-index: 50; display: none; background: rgba(0,0,0,.76); padding: 22px; }
        .modal-backdrop.active { display: grid; place-items: center; }
        .manual-tag-modal { width: min(96vw, 1320px); height: min(94vh, 920px); background: #0f1413; border: 1px solid #3a4541; border-radius: 8px; display: grid; grid-template-rows: auto 1fr auto; gap: 12px; padding: 14px; box-shadow: 0 24px 70px rgba(0,0,0,.6); }
        .manual-tag-modal canvas { width: 100% !important; height: 100% !important; min-height: 0; border: 1px solid #73562b; border-radius: 6px; background: #050706; cursor: crosshair; }
        .modal-head, .modal-actions { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
        .chart-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 8px 0 12px; }
        .chart-controls input { min-height: 34px; border-radius: 6px; border: 1px solid #3a4541; background: #101514; color: var(--text); padding: 0 8px; }
        .chart-controls input[type="range"] { padding: 0; min-height: 28px; }
        .timeline { display: grid; gap: 10px; margin-top: 12px; }
        .event { display: grid; grid-template-columns: 82px 1fr; gap: 10px; border-top: 1px solid #26302c; padding-top: 10px; color: #c9d3ce; }
        .event time { color: var(--muted); font-size: .78rem; }
        .log { font-family: Consolas, monospace; font-size: .76rem; white-space: pre-wrap; background: #070908; border: 1px solid #202824; border-radius: 6px; padding: 10px; max-height: 130px; overflow: auto; color: #aab7b0; }
        .operation-console { font-family: Consolas, monospace; white-space: pre-wrap; min-height: 118px; max-height: 240px; overflow: auto; background: #050807; color: #9dd7ac; border: 1px solid #26342d; border-radius: 6px; padding: 10px; line-height: 1.45; }
        .operation-progress { width: 100%; height: 8px; margin: 8px 0; accent-color: var(--green); }
        .empty { border: 1px dashed #3a4541; border-radius: 8px; padding: 28px; color: var(--muted); text-align: center; }
        @media (max-width: 980px) {
            .shell { grid-template-columns: 1fr; }
            .sidebar { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
            .workspace, .summary, .setup-workbench { grid-template-columns: 1fr; }
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
            <div class="nav-item" data-view="segmentation" onclick="showSection('segmentation', this)">Setup</div>
            <div class="nav-item" data-view="growth" onclick="showSection('growth', this)">Growth Analytics</div>
            <div class="nav-item" data-view="movement" onclick="showSection('movement', this)">Movement</div>
            <div class="nav-item" data-view="experiments" onclick="showSection('experiments', this)">Active Experiments</div>
            <div class="nav-section">Operations</div>
            <div class="nav-item" data-view="volume" onclick="showSection('volume', this)">Canopy Volume</div>
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
                    <div class="panel-head"><div><h2>Camera Feeds</h2><div class="muted">Latest frame with capture controls</div></div></div>
                    {% if devices %}
                    <div class="camera-grid">
                        {% for d in devices %}
                        <article class="device-card" data-device="{{ d }}" data-live-stream="{{ '1' if d in network_stream_ids else '0' }}">
                            <div class="device-title"><div class="device-id" id="title-{{ d }}">{{ device_aliases.get(d, d) }}</div><span class="badge" id="badge-{{ d }}" style="display:none">Fastest</span></div>
                            <div class="muted">Last sync: <span id="last-{{ d }}">{{ last.get(d, 'Initializing') }}</span></div>
                            <div class="image-wrap mission-image-wrap" id="image-wrap-{{d}}" style="--rotation:0deg">
                                <img id="analysis-{{d}}" src="/analysis_debug/{{d}}" onload="renderCalibrationOverlays('{{d}}', (latestStats['{{d}}'] || {}).data || {}); renderAllSegmentOverlays()" onpointerdown="startIgnoreDrag(event, '{{d}}')" onpointermove="moveIgnoreDrag(event, '{{d}}')" onpointerup="finishIgnoreDrag(event, '{{d}}')" onpointercancel="cancelIgnoreDrag('{{d}}')" onerror="this.onerror=null; this.src='/last_frame/{{d}}'">
                                <div class="calib-overlay" id="segment-overlay-{{d}}"></div>
                                <div class="calib-overlay" id="calib-overlay-{{d}}"></div>
                                <div id="selection-{{d}}" style="position:absolute; border: 2px dashed var(--amber); pointer-events:none; display:none;"></div>
                            </div>
                            <div class="controls">
                                <button onclick="captureDevice('{{d}}')">Capture</button>
                                <button onclick="toggleMissionGreenMask('{{d}}')" id="mission-greenmask-toggle-{{d}}">Greenmask On</button>
                                <button onclick="refreshDevice('{{d}}')">Refresh</button>
                            </div>
                            <div class="rotation-row">
                                <label class="muted" for="rotation-{{d}}">Rotate</label>
                                <input id="rotation-{{d}}" type="range" min="0" max="360" step="1" value="0" oninput="previewRotation('{{d}}', this.value)" onchange="saveDeviceSetting('{{d}}', 'display_rotation_deg', this.value)">
                                <input id="rotation-number-{{d}}" type="number" min="0" max="360" step="1" value="0" oninput="previewRotation('{{d}}', this.value)" onchange="saveDeviceSetting('{{d}}', 'display_rotation_deg', this.value)">
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
                            <div class="controls"><button onclick="loadMissionTimelapse('{{d}}')">Load Timelapse</button></div>
                            <video id="vid-{{d}}" controls loop muted preload="none" data-src="/video/{{d}}"></video>
                            <div class="telemetry mission-canopy-only" id="stats-{{d}}">
                                <div><span>Canopy Area</span><strong>--</strong></div>
                            </div>
                        </article>
                        {% endfor %}
                    </div>
                    {% else %}
                    <div class="empty">No ADB cameras are online. Connect the Galaxy S5 Neo and keep USB debugging enabled.</div>
                    {% endif %}
                </section>
                <aside class="panel">
                    <div class="panel-head">
                        <div><h2>Growth Analytics</h2><div class="muted">Selected metric over time</div></div>
                        <select id="fleet-metric" onchange="setFleetControl('metric', this.value, true)">
                            <option value="area">Canopy area</option>
                            <option value="growth_speed">Growth speed</option>
                            <option value="green_index">Green index</option>
                            <option value="canopy_coverage">Canopy coverage</option>
                            <option value="volume_cm3">Canopy volume</option>
                        </select>
                    </div>
                    <div class="chart-controls" id="fleet-chart-controls"></div>
                    <div class="chart-wrap"><canvas id="fleet-chart"></canvas></div>
                    <div class="timeline" id="event-timeline"><div class="event"><time>Now</time><div>Waiting for capture telemetry.</div></div></div>
                </aside>
                <section class="panel" style="grid-column:1 / -1">
                    <h2>Running Activity</h2>
                    <div class="muted" style="margin-top:6px">Live play-by-play of captures, syncs, analysis, color checks, movement calculations, and recovery actions.</div>
                    <div class="operation-console" id="activity-console" style="margin-top:12px">Waiting for live activity.</div>
                </section>
            </div>
            <section class="panel view-section" id="view-segmentation">
                <h2>Connected Devices</h2>
                <div class="setup-workbench">
                    <aside>
                        <div class="device-list">
                            {% for d in devices %}
                            <button onclick="selectSegmentationDevice('{{d}}')" title="Select this connected camera for setup, segmentation, and calibration."><span id="device-button-label-{{d}}">{{ device_aliases.get(d, d) }}</span><br><small class="muted">{{ d }}</small></button>
                            {% endfor %}
                        </div>
                        <div class="timeline" id="setup-checklist"></div>
                    </aside>
                    <div class="setup-stage">
                        <div class="setup-summary" id="setup-summary"></div>
                        <div class="controls" style="margin-top:10px">
                            <button id="setup-greenmask-toggle" onclick="toggleGreenMask()" title="Switch between green overlay and raw frame.">Greenmask On</button>
                        </div>
                        <div class="image-wrap">
                            <img id="segment-image" src="" onpointerdown="startSegmentDrag(event)" onpointermove="moveSegmentDrag(event)" onpointerup="finishSegmentDrag(event)" ondblclick="finishPointMode()" onerror="handleSetupImageError(this)">
                            <div id="segment-overlays"></div>
                            <div id="segment-selection" style="position:absolute; border:2px dashed var(--cyan); pointer-events:none; display:none;"></div>
                        </div>
                        <h3 style="margin-top:14px">Device Log</h3>
                        <div class="log" id="selected-device-log">Select a device to see its capture and sync log.</div>
                        <div class="timeline" id="segment-list"></div>
                    </div>
                    <aside>
                        <div class="setup-mode-tabs">
                            <button class="active" data-setup-mode="setup" onclick="showSetupMode('setup')">Setup</button>
                            <button data-setup-mode="segmentation" onclick="showSetupMode('segmentation')">Segment</button>
                            <button data-setup-mode="color" onclick="showSetupMode('color')">Color</button>
                            <button data-setup-mode="qa" onclick="showSetupMode('qa')">QA</button>
                        </div>
                        <div class="setup-panel active" data-setup-panel="setup">
                            <h3>Device Setup</h3>
                            <div class="muted">Name the device, assign its chamber, optimize capture settings, confirm tag/scale detection, then lock the fixed measurement profile. Rotation is display-only and remains editable after lock.</div>
                            <div class="lock-notice" id="setup-lock-notice"><strong>Measurement setup is locked.</strong><br>Unlock Setup before changing camera, calibration, segmentation, or color settings. Rotation stays editable because it only changes display orientation.</div>
                            <label class="settings-row" title="How many ArUco/ChArUco markers should be visible in this fixed view. The sweep scores profiles against this target."><span class="muted">Expected tags</span><input id="expected-tags-input" type="number" min="1" max="50" value="4" onchange="saveExpectedTags()"></label>
                            <div id="setup-controls-panel"></div>
                            <div class="timeline" id="setup-steps"></div>
                            <div class="controls">
                                <button onclick="captureSelectedSetupFrame()" title="Capture one fresh frame using the current settings.">Test Frame</button>
                                <button class="primary" data-locked-edit onclick="autoTuneSelectedTags()" title="Ask for expected visible tags, then sweep camera settings and lock the best measurement profile.">Calibrate Measurement Profile</button>
                                <button data-locked-edit onclick="enableManualMarkerMode()" title="Crop and click a visible marker by hand.">Manual Tag</button>
                                <button data-locked-edit onclick="clearManualTagsForSelected()" title="Remove all manual tags for this device.">Clear Tags</button>
                                <button class="primary" data-locked-edit onclick="lockSelectedMeasurementSetup()" title="Mark the current camera, calibration, greenmask, and motion setup as fixed.">Lock Setup</button>
                            </div>
                            <progress class="operation-progress" id="setup-operation-progress" max="100" value="0"></progress>
                            <div class="operation-console" id="setup-operation-console">Ready. Long-running setup actions will report their work here.</div>
                        </div>
                        <div class="setup-panel" data-setup-panel="segmentation">
                            <h3>Segmentation</h3>
                            <div class="muted">Define trays/plants and exclude false green areas.</div>
                            <div class="controls">
                                <button data-locked-edit onclick="enableSegmentMode()" title="Drag a rectangular tray or plant region.">Segment Box</button>
                                <button data-locked-edit onclick="enablePolygonSegmentMode()" title="Click around a skewed tray, then double-click.">Segment Polygon</button>
                                <button data-locked-edit onclick="enableIgnoreEditorMode('ignore-box')" title="Drag over false green/artifact space.">Ignore Box</button>
                                <button data-locked-edit onclick="enableIgnoreEditorMode('ignore-polygon')" title="Click around false green/artifact space, then double-click.">Ignore Polygon</button>
                                <button data-locked-edit onclick="clearSegmentsForSelected()" title="Remove every saved segment.">Clear Segments</button>
                                <button data-locked-edit onclick="clearEditorIgnores()" title="Remove ignored regions.">Clear Ignores</button>
                                <button onclick="refreshSegmentationFrame()" title="Reload the selected frame.">Refresh Frame</button>
                            </div>
                        </div>
                        <div class="setup-panel" data-setup-panel="color">
                            <h3>Color Reference</h3>
                            <div class="muted">Colored ChArUco cells are detected and sampled automatically from their known board positions. Manual polygons remain available for handmade or partially hidden references.</div>
                            <label class="settings-row" title="Enable drift scoring and optional analysis-only green-index correction. Raw photos are never changed."><span class="muted">Color QA</span><input id="color-reference-enabled" type="checkbox" checked onchange="saveColorReferenceSettings()"><strong id="color-reference-enabled-label">On</strong></label>
                            <label class="settings-row" title="Off records patch values only. Simple uses neutral patches. Advanced uses all matching patches to solve a color matrix for analysis metrics."><span class="muted">Correction</span><select id="color-reference-mode" onchange="saveColorReferenceSettings()"><option value="off">Off</option><option value="simple">Simple</option><option value="advanced">Advanced</option></select><strong></strong></label>
                            <div class="controls">
                                <button data-locked-edit onclick="enableColorPatchMode()" title="Fallback: click around a color swatch, then double-click to save it.">Add Manual Patch</button>
                                <button class="primary" data-locked-edit onclick="finishColorPatchMode()" title="Finish the current color patch polygon and enter its name/code.">Finish Color Patch</button>
                                <button data-locked-edit onclick="setColorReferenceBaseline()" title="Use the current sampled patch colors as the baseline for future drift/correction.">Set Baseline</button>
                                <button data-locked-edit onclick="clearColorPatchesForSelected()" title="Remove every saved color patch for this camera.">Clear Color Board</button>
                                <button onclick="refreshSegmentationFrame()" title="Reload the selected frame.">Refresh Frame</button>
                            </div>
                            <div class="muted" id="color-patch-status">No active color patch.</div>
                            <div class="timeline" id="color-reference-status"></div>
                            <div class="timeline" id="color-patch-list"></div>
                        </div>
                        <div class="setup-panel" data-setup-panel="qa">
                            <h3>Measurement QA</h3>
                            <div class="muted">Check whether the current measurement is believable.</div>
                            <div id="setup-qa-panel" class="timeline"></div>
                        </div>
                        <div id="setup-device-controls-source" style="display:none">
                        {% for d in devices %}
                        <div class="setup-control-card" id="setup-controls-{{d}}" data-setup-device="{{d}}">
                            <div class="settings-row"><label class="muted" for="alias-{{d}}">Name</label><input id="alias-{{d}}" type="text" value="{{ device_aliases.get(d, '') }}" placeholder="{{ d }}" title="Local display name only. This does not rename capture folders or device IDs." onchange="saveDeviceAlias('{{d}}', this.value)"><strong></strong></div>
                            <div class="settings-row"><label class="muted" for="chamber-{{d}}">Chamber</label><input id="chamber-{{d}}" type="text" value="{{ (device_metadata.get(d, {}) or {}).get('chamber', '') }}" placeholder="Chamber A" onchange="saveDeviceMetadata('{{d}}', 'chamber', this.value)"><strong></strong></div>
                            <div class="settings-row"><label class="muted" for="role-{{d}}">Role</label><select id="role-{{d}}" onchange="saveDeviceMetadata('{{d}}', 'role', this.value)">{% set role = (device_metadata.get(d, {}) or {}).get('role', 'tray') %}<option value="tray" {{ 'selected' if role == 'tray' else '' }}>Tray</option><option value="wall" {{ 'selected' if role == 'wall' else '' }}>Wall</option><option value="overview" {{ 'selected' if role == 'overview' else '' }}>Overview</option><option value="night_vision" {{ 'selected' if role == 'night_vision' else '' }}>Night vision</option><option value="calibration" {{ 'selected' if role == 'calibration' else '' }}>Calibration</option><option value="other" {{ 'selected' if role == 'other' else '' }}>Other</option></select><strong></strong></div>
                            <div class="controls" style="margin-top:12px"><button id="live-button-{{d}}" onclick="toggleLiveView('{{d}}')">Live View</button><button onclick="autoTuneTags('{{d}}')">Calibrate Measurement Profile</button><button id="auto-light-{{d}}" onclick="setAutoLight('{{d}}')">Auto Light</button></div>
                            <label class="settings-row" title="When off, this device contributes a one-second labeled night placeholder instead of black frames. Measurements ignore placeholders, but timelapse timing stays aligned."><span class="muted">Night frames</span><input id="collect-night-{{d}}" type="checkbox" onchange="saveDeviceSetting('{{d}}', 'collect_night_frames', this.checked)"><strong id="collect-night-label-{{d}}">Collect</strong></label>
                            <div class="settings-row"><label class="muted" for="zoom-{{d}}">Zoom</label><input id="zoom-{{d}}" type="range" min="0" max="100" value="0" oninput="previewZoom('{{d}}', this.value)" onchange="saveDeviceSetting('{{d}}', 'zoom_percent', this.value)"><strong id="zoom-value-{{d}}">0%</strong></div>
                            <div class="settings-row"><label class="muted" for="delay-{{d}}">Settle</label><input id="delay-{{d}}" type="range" min="500" max="15000" step="500" value="5000" oninput="previewDelay('{{d}}', this.value)" onchange="saveDeviceSetting('{{d}}', 'delay_ms', this.value)"><strong id="delay-value-{{d}}">5.0s</strong></div>
                            <div class="settings-row"><label class="muted" for="antibanding-{{d}}">Banding</label><select id="antibanding-{{d}}" onchange="saveDeviceSetting('{{d}}', 'antibanding', this.value)"><option value="60hz">60 Hz</option><option value="50hz">50 Hz</option><option value="auto">Auto</option><option value="off">Off</option></select><strong></strong></div>
                            <div class="settings-row"><label class="muted" for="white-balance-{{d}}">WB</label><select id="white-balance-{{d}}" onchange="saveDeviceSetting('{{d}}', 'white_balance', this.value)"><option value="daylight">Daylight</option><option value="cloudy-daylight">Cloudy</option><option value="fluorescent">Fluorescent</option><option value="incandescent">Incandescent</option><option value="shade">Shade</option><option value="twilight">Twilight</option><option value="warm-fluorescent">Warm Fluor.</option><option value="auto">Auto</option></select><strong></strong></div>
                            <div class="settings-row"><label class="muted" for="focus-{{d}}">Focus</label><select id="focus-{{d}}" onchange="saveDeviceSetting('{{d}}', 'focus_mode', this.value)"><option value="continuous-picture">Continuous</option><option value="auto">Auto</option><option value="macro">Macro</option><option value="infinity">Infinity</option><option value="fixed">Fixed</option></select><strong></strong></div>
                            <div class="settings-row"><label class="muted" for="exposure-{{d}}">Exposure</label><input id="exposure-{{d}}" type="range" min="-12" max="12" step="1" value="0" oninput="previewExposure('{{d}}', this.value)" onchange="saveDeviceSetting('{{d}}', 'exposure_compensation', this.value)"><strong id="exposure-value-{{d}}">0</strong></div>
                            <div class="settings-row"><label class="muted" for="iso-{{d}}">ISO</label><select id="iso-{{d}}" onchange="saveDeviceSetting('{{d}}', 'iso', this.value)"><option value="auto">Auto</option><option value="100">100</option><option value="200">200</option><option value="400">400</option><option value="800">800</option><option value="1600">1600</option></select><strong></strong></div>
                            <div class="rotation-row">
                                <label class="muted" for="setup-rotation-{{d}}">Rotate</label>
                                <input id="setup-rotation-{{d}}" type="range" min="0" max="360" step="1" value="0" oninput="previewRotation('{{d}}', this.value)" onchange="saveDeviceSetting('{{d}}', 'display_rotation_deg', this.value)">
                                <input id="setup-rotation-number-{{d}}" type="number" min="0" max="360" step="1" value="0" oninput="previewRotation('{{d}}', this.value)" onchange="saveDeviceSetting('{{d}}', 'display_rotation_deg', this.value)">
                            </div>
                        </div>
                        {% endfor %}
                        </div>
                    </aside>
                </div>
            </section>
            <section class="panel view-section" id="view-growth">
                <h2>Growth Analytics</h2>
                <div class="muted" style="margin-top:6px">Each camera gets its own chart. Named segmentation regions become individual lines.</div>
                <div id="growth-charts" class="timeline"></div>
            </section>
            <section class="panel view-section" id="view-movement">
                <h2>Movement</h2>
                <div class="muted" style="margin-top:6px">Movement uses canopy centroid drift between calibrated captures. Circadian rhythm, droop warning, brightness flattening/stretching, and plant identity by movement axis are staged here as the next analysis layer.</div>
                <div class="timeline">
                    <div class="event">
                        <time>Video</time>
                        <div>
                            <strong>Retroactive movement analysis</strong><br>
                            Import frames from an existing MP4 in <code>C:\\dev\\pt\\videos</code>, then analyze them like normal captures.
                            <div class="controls" style="margin-top:10px">
                                <select id="video-import-select"></select>
                                <input id="video-import-device" type="text" placeholder="device_id from filename">
                                <input id="video-import-sample" type="number" min="1" max="3600" value="60" title="Seconds between sampled frames">
                                <input id="video-import-max" type="number" min="1" max="2000" value="240" title="Maximum frames to extract">
                                <button class="primary" onclick="importSelectedVideo()">Import + Analyze</button>
                                <button onclick="loadVideoImports()">Refresh Videos</button>
                            </div>
                            <div class="muted" id="video-import-status" style="margin-top:8px">Video importer ready.</div>
                        </div>
                    </div>
                </div>
                <h3 style="margin-top:18px">Circadian Fingerprints</h3>
                <div class="timeline" id="movement-rhythm-list"></div>
                <div class="timeline" id="movement-list"></div>
            </section>
            <section class="panel view-section" id="view-experiments">
                <h2>Active Experiments</h2>
                <div class="muted" style="margin-top:6px">Assign cameras/plants, record interventions, and compare growth, color, and canopy movement against those events.</div>
                <div class="controls" style="margin-top:12px">
                    <input id="experiment-id" type="text" placeholder="experiment_id">
                    <input id="experiment-name" type="text" placeholder="Experiment name">
                    <input id="experiment-hypothesis" type="text" placeholder="Hypothesis">
                    <button class="primary" onclick="saveExperiment()">Save Experiment</button>
                </div>
                <div class="controls" style="margin-top:8px">
                    <select id="experiment-event-id"></select>
                    <select id="experiment-event-type"><option value="observation">Observation</option><option value="planted">Planted</option><option value="germination">Germination / first plant</option><option value="irrigation">Irrigation</option><option value="fertilizer">Fertilizer</option><option value="light_change">Light change</option><option value="harvest">Harvest</option><option value="stress">Stress</option><option value="treatment">Treatment</option><option value="breeding">Breeding / parentage</option></select>
                    <select id="experiment-event-device"><option value="">All devices</option>{% for d in devices %}<option value="{{d}}">{{ device_aliases.get(d, d) }}</option>{% endfor %}</select>
                    <input id="experiment-event-plant" type="text" placeholder="plant/tray id">
                    <input id="experiment-event-notes" type="text" placeholder="Event notes">
                    <button onclick="logExperimentEvent()">Log Event</button>
                    <button onclick="enablePlantNotifications()">Enable Notifications</button>
                </div>
                <h3 style="margin-top:18px">Tray / Plant Records</h3>
                <div class="controls" style="margin-top:8px">
                    <input id="plant-record-id" type="text" placeholder="plant_or_tray_id">
                    <input id="plant-record-name" type="text" placeholder="plant / tray name">
                    <input id="plant-record-variety" type="text" placeholder="variety">
                    <input id="plant-record-planted" type="date" title="planted date">
                    <input id="plant-record-germination" type="datetime-local" title="first germination / first plant">
                    <input id="plant-record-device" type="text" placeholder="phone/device id">
                    <input id="plant-record-segment" type="text" placeholder="segment id">
                    <input id="plant-record-irrigation" type="text" placeholder="irrigation">
                    <input id="plant-record-fertilizer" type="text" placeholder="fertilizer">
                    <input id="plant-record-harvest" type="text" placeholder="harvest">
                    <input id="plant-record-mother" type="text" placeholder="mother parent">
                    <input id="plant-record-father" type="text" placeholder="father parent">
                    <input id="plant-record-notes" type="text" placeholder="other notes">
                    <button class="primary" onclick="savePlantRecord()">Save Plant/Tray</button>
                </div>
                <div class="timeline" id="biology-alerts"></div>
                <div class="timeline" id="plant-records-list"></div>
                <h3 style="margin-top:18px">Calendar</h3>
                <div class="timeline" id="experiment-calendar"></div>
                <h3 style="margin-top:18px">Circadian Fingerprints</h3>
                <div class="timeline" id="rhythm-list"></div>
                <div class="timeline" id="experiments-list"></div>
                <h3 style="margin-top:18px">Trait And Movement Ranking</h3>
                <div class="timeline" id="trait-ranking"></div>
            </section>
            <section class="panel view-section" id="view-volume">
                <h2>Canopy Volume</h2>
                <div class="muted" style="margin-top:6px">Select cameras seeing the same plants and calibration target, confirm target dimensions, then calibrate and reconstruct.</div>
                <div class="timeline" id="volume-camera-list"></div>
                <div class="controls" style="max-width:520px">
                    <button onclick="calibrateMulti()">Calibrate 3D</button>
                    <button onclick="reconstruct3D()">Run Volumetric</button>
                </div>
                <canvas id="volume-preview" style="margin-top:12px; height:320px !important; background:#070908; border:1px solid #26302c; border-radius:6px"></canvas>
                <div class="timeline" id="calibration-list"><div class="event"><time>Ready</time><div>Use the red marker editors on Mission Control to confirm ArUco/ChArUco dimensions first.</div></div></div>
            </section>
            <section class="panel view-section" id="view-settings">
                <h2>Settings</h2>
                <div class="timeline"><div class="event"><time>Camera</time><div>Use Setup for capture tuning, tag detection, segmentation, and ignore regions. Mission Control is for observing, capture, refresh, and ESCAM movement.</div></div></div>
                <h3 style="margin-top:18px">Wi-Fi ADB</h3>
                <div class="timeline">
                    <div class="event">
                        <time>Optional</time>
                        <div>
                            <strong>Connect Android phones through the PC hotspot</strong><br>
                            USB remains the setup and recovery path. Enter the phone's hotspot address for direct connection without relying on network discovery.
                            <div class="controls" style="margin-top:8px">
                                <input id="adb-wifi-name" type="text" placeholder="Camera name">
                                <input id="adb-wifi-endpoint" type="text" placeholder="192.168.137.42:5555">
                                <button class="primary" onclick="connectWifiAdb()">Connect and Save</button>
                                <button class="warn" onclick="disconnectWifiAdb()">Disconnect and Forget</button>
                                <button onclick="reconnectWifiAdb()">Reconnect Saved</button>
                            </div>
                            <div class="controls" style="margin-top:8px">
                                <select id="adb-usb-serial"><option value="">Select USB phone</option></select>
                                <button onclick="prepareLegacyWifiAdb()">Prepare USB Phone for Wi-Fi</button>
                            </div>
                            <div class="controls" style="margin-top:8px">
                                <input id="adb-pair-endpoint" type="text" placeholder="Android 11+ pairing IP:port">
                                <input id="adb-pair-code" type="text" inputmode="numeric" placeholder="Pairing code">
                                <button onclick="pairWifiAdb()">Pair Android 11+</button>
                            </div>
                            <div class="muted" id="adb-transport-status" style="margin-top:8px">Loading ADB transports...</div>
                        </div>
                    </div>
                </div>
                <h3 style="margin-top:18px">Long-Term Metrics</h3>
                <div class="timeline">
                    <div class="event"><time>SQLite</time><div><strong>Durable metric history</strong><br>New captures are stored in an append-only SQLite database with hourly/daily rollups for long-term charts.<br><div class="controls" style="margin-top:8px"><button onclick="startMetricBackfill()">Rebuild From Captures</button><button class="warn" onclick="clearMetricHistory()">Clear Derived Metrics</button></div><div class="muted" id="metric-store-status" style="margin-top:8px">Idle</div></div></div>
                </div>
                <h3 style="margin-top:18px">Tag Detection Sweep</h3>
                <div class="timeline">
                    {% for d in devices %}
                    <div class="event"><time>ADB</time><div><strong>{{ d }}</strong><br>Ask how many tags should be visible, then sweep settle, banding, focus, exposure, ISO, and WB to lock the best measurement profile.<br><button onclick="autoTuneTags('{{d}}')">Calibrate Measurement Profile</button></div></div>
                    {% endfor %}
                </div>
                <h3 style="margin-top:18px">Timelapses</h3>
                <div class="timeline" id="timelapse-list"></div>
                <h3 style="margin-top:18px">Custom Timelapse Export</h3>
                <div class="timeline">
                    <div class="event">
                        <time>Range</time>
                        <div>
                            <strong>Render a clip from saved frames</strong><br>
                            Pick a device and date range, such as last weekend, without changing the live timelapse.
                            <div class="controls" style="margin-top:8px">
                                <select id="custom-timelapse-device">{% for d in devices %}<option value="{{d}}">{{ device_aliases.get(d, d) }}</option>{% endfor %}</select>
                                <input id="custom-timelapse-start" type="datetime-local">
                                <input id="custom-timelapse-end" type="datetime-local">
                                <button onclick="makeCustomTimelapse()">Render Clip</button>
                            </div>
                            <div class="muted" id="custom-timelapse-status" style="margin-top:8px">Idle</div>
                            <div id="custom-timelapse-player"></div>
                        </div>
                    </div>
                </div>
            </section>
        </main>
    </div>
    <div class="modal-backdrop" id="manual-tag-modal">
        <div class="manual-tag-modal">
            <div class="modal-head">
                <div><h2>Manual Tag</h2><div class="muted" id="manual-tag-help">Click the four outer marker corners around the perimeter.</div></div>
                <button onclick="closeManualTagModal()">Close</button>
            </div>
            <canvas id="manual-marker-crop" onclick="manualCropClick(event)"></canvas>
            <div class="modal-actions">
                <div class="muted" id="manual-tag-count">0 / 4 corners</div>
                <div class="actions">
                    <button onclick="resetManualTagCorners()">Reset Corners</button>
                    <button class="primary" onclick="finishManualTagFromModal()">Finish Tag</button>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const charts = {};
        const ignoreState = {};
        const segmentState = { deviceId: null, enabled: false, dragging: false, mode: 'box', points: [], roi: null, manualMarkers: {}, colorBoards: {}, segments: {} };
        const growthControls = {};
        const fleetControls = { metric: 'area', windowMode: 'all', customValue: 1, customUnit: 'days', windowHours: 0, minX: '', maxX: '', minY: '', maxY: '', trim: true, outlierSensitivity: 2.5 };
        const MAX_CHART_POINTS = 700;
        const connectedDevices = new Set({{ devices|tojson }});
        const deviceAliases = {{ device_aliases|tojson }};
        const deviceMetadata = {{ device_metadata|tojson }};
        const deviceLogs = {{ logs|tojson }};
        const settingsCache = {};
        const colorReferenceCache = {};
        const networkCameraStatus = {};
        let showSetupGreenMask = true;
        let showMissionGreenMask = true;
        const missionGreenMaskState = {};
        let activeSetupMode = 'setup';
        let pointPopover = null;
        let setupOperationTimer = null;
        let liveDevice = null;
        let liveTimer = null;
        let liveBusy = false;
        let latestStats = {};
        let activeView = 'mission';
        const seenAlertKeys = new Set();
        function fmt(value, digits = 1) { return Number.isFinite(value) ? value.toFixed(digits) : '--'; }
        function aliasOf(deviceId) { return deviceAliases[deviceId] || deviceId; }
        function showOperation(title, detail, tone = 'normal') {
            const el = document.getElementById('operation-result');
            el.style.display = 'block';
            el.style.borderColor = tone === 'bad' ? 'var(--red)' : tone === 'warn' ? 'var(--amber)' : 'var(--line)';
            el.innerHTML = `<h2>${title}</h2><div class="muted" style="margin-top:6px">${detail}</div>`;
        }
        function newOperationId(prefix) {
            return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        }
        function renderSetupOperation(job) {
            const consoleEl = document.getElementById('setup-operation-console');
            const progressEl = document.getElementById('setup-operation-progress');
            if (consoleEl) {
                consoleEl.textContent = (job.lines || []).join('\\\\n') || 'Waiting for operation output...';
                consoleEl.scrollTop = consoleEl.scrollHeight;
            }
            if (progressEl) progressEl.value = Number(job.progress || 0);
        }
        function watchSetupOperation(operationId) {
            clearInterval(setupOperationTimer);
            const poll = () => fetch('/operations/' + encodeURIComponent(operationId))
                .then(r => r.ok ? r.json() : null)
                .then(job => {
                    if (!job) return;
                    renderSetupOperation(job);
                    if (job.status === 'complete' || job.status === 'error') {
                        clearInterval(setupOperationTimer);
                        setupOperationTimer = null;
                    }
                }).catch(() => {});
            poll();
            setupOperationTimer = setInterval(poll, 700);
        }
        function adbRequest(path, payload) {
            return fetch(path, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload || {})
            }).then(async response => {
                const body = await response.json();
                if (!response.ok) throw new Error(body.message || 'ADB command failed');
                return body;
            });
        }
        function loadAdbTransportStatus() {
            fetch('/adb_transport').then(r => r.json()).then(status => {
                const el = document.getElementById('adb-transport-status');
                const configured = (status.configured || []).map(device => {
                    const label = device.name || device.endpoint || 'Invalid endpoint';
                    return `${label}: ${device.online ? 'online' : 'offline'}`;
                });
                if (el) el.textContent = status.adb_available
                    ? `USB: ${(status.usb || []).join(', ') || 'none'} | Wi-Fi: ${(status.wifi || []).join(', ') || 'none'} | Saved: ${configured.join(', ') || 'none'}`
                    : 'ADB was not found. Install Android platform-tools or set PT_ADB.';
                const select = document.getElementById('adb-usb-serial');
                if (select) {
                    const selected = select.value;
                    select.innerHTML = '<option value="">Select USB phone</option>' + (status.usb || []).map(serial => `<option value="${serial}">${serial}</option>`).join('');
                    if ((status.usb || []).includes(selected)) select.value = selected;
                }
            }).catch(error => {
                const el = document.getElementById('adb-transport-status');
                if (el) el.textContent = `Could not read ADB status: ${error.message}`;
            });
        }
        function connectWifiAdb() {
            const endpoint = document.getElementById('adb-wifi-endpoint').value.trim();
            const name = document.getElementById('adb-wifi-name').value.trim();
            adbRequest('/adb_transport/connect', {endpoint, name, remember: true})
                .then(body => {
                    showOperation('Wi-Fi ADB connected', body.message || body.endpoint);
                    loadAdbTransportStatus();
                })
                .catch(error => showOperation('Wi-Fi ADB connection failed', error.message, 'bad'));
        }
        function reconnectWifiAdb() {
            adbRequest('/adb_transport/reconnect', {}).then(body => {
                const messages = (body.results || []).map(result => `${result.endpoint}: ${result.message}`).join(' | ');
                showOperation('Saved Wi-Fi ADB reconnect complete', messages || 'No saved endpoints are enabled.', body.ok ? 'normal' : 'warn');
                loadAdbTransportStatus();
            }).catch(error => showOperation('Wi-Fi ADB reconnect failed', error.message, 'bad'));
        }
        function disconnectWifiAdb() {
            const endpoint = document.getElementById('adb-wifi-endpoint').value.trim();
            adbRequest('/adb_transport/disconnect', {endpoint, forget: true}).then(body => {
                showOperation('Wi-Fi ADB disconnected', body.message || body.endpoint);
                loadAdbTransportStatus();
            }).catch(error => showOperation('Wi-Fi ADB disconnect failed', error.message, 'bad'));
        }
        function prepareLegacyWifiAdb() {
            const serial = document.getElementById('adb-usb-serial').value;
            const name = document.getElementById('adb-wifi-name').value.trim();
            adbRequest('/adb_transport/prepare_legacy', {serial, name, remember: true}).then(body => {
                document.getElementById('adb-wifi-endpoint').value = body.endpoint || '';
                showOperation('Phone prepared for Wi-Fi ADB', `${serial} is available at ${body.endpoint}.`);
                loadAdbTransportStatus();
            }).catch(error => showOperation('Legacy Wi-Fi setup failed', error.message, 'bad'));
        }
        function pairWifiAdb() {
            const endpoint = document.getElementById('adb-pair-endpoint').value.trim();
            const pairingCode = document.getElementById('adb-pair-code').value.trim();
            adbRequest('/adb_transport/pair', {endpoint, pairing_code: pairingCode}).then(body => {
                showOperation('Wireless debugging paired', `${body.endpoint}: ${body.message}. Enter Android's connection address above, then Connect and Save.`);
            }).catch(error => showOperation('Wireless debugging pairing failed', error.message, 'bad'));
        }
        function showSection(name, item) {
            activeView = name;
            document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            const section = document.getElementById('view-' + name);
            if (section) section.classList.add('active');
            if (item) item.classList.add('active');
            pauseHiddenVideos();
            if (name === 'settings') renderTimelapses(latestStats);
            if (name === 'growth') {
                renderGrowthCharts(latestStats);
                loadActivity();
            }
            if (name === 'movement') {
                renderMovementWorkspace(latestStats);
                loadVideoImports();
            }
            if (name === 'segmentation') loadSegments();
            if (name === 'segmentation' && !segmentState.deviceId) {
                const first = Object.keys(latestStats || {})[0];
                if (first) selectSegmentationDevice(first);
            }
        }
        function calibrateMulti() {
            if (confirm("Calibrate the multi-camera coordinate system using current frames?")) {
                showOperation('Calibration running', 'Looking for ChArUco corners in the current frames...');
                fetch('/calibrate_multicam' + selectedVolumeQuery()).then(async r => {
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
            fetch('/reconstruct' + selectedVolumeQuery()).then(async r => {
                const body = r.headers.get('content-type')?.includes('application/json') ? await r.json() : { error: await r.text() };
                if (!r.ok) throw new Error(body.error || body.message || 'Reconstruction failed');
                return body;
            }).then(d => {
                const cm3 = d.volume_mm3 ? (d.volume_mm3 / 1000).toFixed(2) : '0.00';
                showOperation('Volumetric reconstruction complete', `Volume: ${cm3} cm3. Occupied voxels: ${d.occupied_voxels || 0}. Grid: ${(d.grid_shape || []).join(' x ')}`);
                renderVolumePreview(d);
            }).catch(e => showOperation('Reconstruction failed', e.message, 'bad'));
        }
        function renderVolumePreview(data) {
            const canvas = document.getElementById('volume-preview');
            if (!canvas || !data.preview_points) return;
            const rect = canvas.getBoundingClientRect();
            canvas.width = Math.max(320, Math.floor(rect.width));
            canvas.height = 320;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#070908';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            const bounds = data.world_bounds || [-200, -200, 0, 200, 200, 400];
            ctx.fillStyle = '#7ac77f';
            for (const p of data.preview_points) {
                const x = (p[0] - bounds[0]) / Math.max(1, bounds[3] - bounds[0]) * canvas.width;
                const y = canvas.height - ((p[2] - bounds[2]) / Math.max(1, bounds[5] - bounds[2]) * canvas.height);
                ctx.globalAlpha = 0.45;
                ctx.fillRect(x, y, 2, 2);
            }
            ctx.globalAlpha = 1;
            ctx.fillStyle = '#8c9a94';
            ctx.fillText('Side preview: X by height. Sparse/missing areas indicate reconstruction gaps.', 12, 20);
        }
        function selectedVolumeQuery() {
            if (activeView !== 'volume') return '';
            const devices = Array.from(document.querySelectorAll('.volume-camera-check:checked')).map(el => el.value);
            return devices.length ? '?devices=' + encodeURIComponent(devices.join(',')) : '';
        }
        function frameUrl(deviceId, greenmask) {
            return (greenmask ? '/analysis_debug/' : '/last_frame/') + deviceId + '?t=' + Date.now();
        }
        function refreshFrame(deviceId) {
            const img = document.getElementById('analysis-' + deviceId);
            const greenmask = missionGreenMaskState[deviceId] !== false;
            if (img) {
                img.style.display = 'block';
                img.src = frameUrl(deviceId, greenmask);
            }
        }
        function toggleMissionGreenMask(deviceId = null) {
            if (deviceId) {
                const next = missionGreenMaskState[deviceId] === false;
                missionGreenMaskState[deviceId] = next;
                const button = document.getElementById('mission-greenmask-toggle-' + deviceId);
                if (button) button.textContent = next ? 'Greenmask On' : 'Greenmask Off';
                refreshFrame(deviceId);
                return;
            }
            showMissionGreenMask = !showMissionGreenMask;
            document.querySelectorAll('article[data-device]').forEach(card => {
                missionGreenMaskState[card.dataset.device] = showMissionGreenMask;
                const button = document.getElementById('mission-greenmask-toggle-' + card.dataset.device);
                if (button) button.textContent = showMissionGreenMask ? 'Greenmask On' : 'Greenmask Off';
                refreshFrame(card.dataset.device);
            });
        }
        function captureDevice(deviceId) {
            if (liveDevice === deviceId) toggleLiveView(deviceId);
            showOperation('Capture running', `${deviceId}: requesting a fresh frame...`);
            return fetch('/capture/' + deviceId).then(async response => {
                const detail = await response.text();
                if (!response.ok) throw new Error(detail || `Capture failed (${response.status})`);
                refreshFrame(deviceId);
                refreshSegmentationFrame();
                updateStats();
                loadNetworkCameraStatus(false);
                refreshSelectedDeviceLog(deviceId);
                showOperation('Capture complete', `${deviceId}: fresh frame captured and analyzed.`);
                return detail;
            }).catch(error => {
                showOperation('Capture failed', `${deviceId}: ${error.message}`, 'bad');
                loadNetworkCameraStatus(true);
                throw error;
            });
        }
        function refreshDevice(deviceId) {
            if (deviceUsesNetworkStream(deviceId)) {
                captureDevice(deviceId).catch(() => {});
            } else {
                refreshFrame(deviceId);
                updateStats();
            }
        }
        function deviceUsesNetworkStream(deviceId) {
            return document.querySelector(`[data-device="${deviceId}"]`)?.dataset.liveStream === '1';
        }
        function stopLiveStreamImage(deviceId) {
            const img = document.getElementById('analysis-' + deviceId);
            if (!img) return;
            img.src = '/analysis_debug/' + deviceId + '?t=' + Date.now();
        }
        function liveCaptureOnce(deviceId) {
            if (liveBusy || liveDevice !== deviceId || document.hidden) return;
            liveBusy = true;
            fetch('/capture/' + deviceId)
                .then(() => {
                    refreshFrame(deviceId);
                    updateStats();
                })
                .catch(e => showOperation('Live view capture failed', `${deviceId}: ${e.message}`, 'warn'))
                .finally(() => { liveBusy = false; });
        }
        function toggleLiveView(deviceId) {
            if (liveDevice === deviceId) {
                clearInterval(liveTimer);
                liveTimer = null;
                liveBusy = false;
                document.getElementById('analysis-' + deviceId)?.classList.remove('live-active');
                const currentButton = document.getElementById('live-button-' + deviceId);
                if (currentButton) currentButton.textContent = 'Live View';
                if (deviceUsesNetworkStream(deviceId)) stopLiveStreamImage(deviceId);
                liveDevice = null;
                showOperation('Live view stopped', `${deviceId} is back to normal refresh cadence.`);
                return;
            }
            if (liveDevice) {
                document.getElementById('analysis-' + liveDevice)?.classList.remove('live-active');
                const previousButton = document.getElementById('live-button-' + liveDevice);
                if (previousButton) previousButton.textContent = 'Live View';
                if (deviceUsesNetworkStream(liveDevice)) stopLiveStreamImage(liveDevice);
            }
            clearInterval(liveTimer);
            liveBusy = false;
            liveDevice = deviceId;
            document.getElementById('analysis-' + deviceId)?.classList.add('live-active');
            const liveButton = document.getElementById('live-button-' + deviceId);
            if (liveButton) liveButton.textContent = 'Stop Live';
            if (deviceUsesNetworkStream(deviceId)) {
                const img = document.getElementById('analysis-' + deviceId);
                img.src = '/live_stream/' + deviceId + '?t=' + Date.now();
                showOperation('Live stream active', `${deviceId} is showing the configured network stream. Only one live stream runs at a time.`);
                return;
            }
            const settingsDelay = Number(document.getElementById('delay-' + deviceId)?.value || 5000);
            const interval = deviceId.startsWith('escam_') ? 3500 : Math.max(8000, settingsDelay + 3000);
            liveCaptureOnce(deviceId);
            liveTimer = setInterval(() => liveCaptureOnce(deviceId), interval);
            showOperation('Live view active', `${deviceId} is taking fresh low-rate captures every ${(interval / 1000).toFixed(1)}s. Only one live view runs at a time.`);
        }
        function isElementVisible(el) {
            return !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
        }
        function refreshVisibleFrames() {
            if (document.hidden) return;
            document.querySelectorAll('.view-section.active img[id^="analysis-"]').forEach(img => {
                if (isElementVisible(img)) img.src = img.src.split('?')[0] + '?t=' + Date.now();
            });
        }
        function pauseHiddenVideos() {
            document.querySelectorAll('video').forEach(video => {
                const activeSection = video.closest('.view-section.active');
                if (!activeSection || document.hidden) video.pause();
            });
        }
        function clearIgnore(deviceId) {
            if (isSetupLocked(deviceId)) {
                showLockedSetupMessage(deviceId);
                return;
            }
            fetch('/clear_ignore/' + deviceId).then(async r => {
                if (!r.ok) throw new Error((await r.json()).message || 'Clear ignore failed');
                return r;
            }).then(() => {
                showOperation('Ignore regions cleared', `${deviceId} will use the full frame on the next analysis pass.`);
                refreshFrame(deviceId);
                updateStats();
            }).catch(e => showOperation('Ignore clear blocked', e.message, 'bad'));
        }
        function enableIgnoreMode(deviceId) {
            if (isSetupLocked(deviceId)) {
                showLockedSetupMessage(deviceId);
                return;
            }
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
            if (isSetupLocked(deviceId)) {
                showLockedSetupMessage(deviceId);
                return;
            }
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
            if (isSetupLocked(deviceId)) {
                showLockedSetupMessage(deviceId);
                cancelIgnoreDrag(deviceId);
                return;
            }
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
                .then(async r => {
                    if (!r.ok) throw new Error((await r.json()).message || 'Ignore region failed');
                    return r;
                })
                .then(() => {
                    showOperation('Ignore region saved', `${deviceId}: [${region.join(', ')}]. The overlay has been regenerated without that region.`);
                    refreshFrame(deviceId);
                    updateStats();
                }).catch(e => showOperation('Ignore region blocked', e.message, 'bad'));
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
        function previewRotation(deviceId, value) {
            const rotation = ((Number(value) % 360) + 360) % 360;
            const wrap = document.getElementById('image-wrap-' + deviceId);
            if (wrap) wrap.style.setProperty('--rotation', `${rotation}deg`);
            const slider = document.getElementById('rotation-' + deviceId);
            const number = document.getElementById('rotation-number-' + deviceId);
            const setupSlider = document.getElementById('setup-rotation-' + deviceId);
            const setupNumber = document.getElementById('setup-rotation-number-' + deviceId);
            if (slider && document.activeElement !== slider) slider.value = rotation;
            if (number && document.activeElement !== number) number.value = Math.round(rotation);
            if (setupSlider && document.activeElement !== setupSlider) setupSlider.value = rotation;
            if (setupNumber && document.activeElement !== setupNumber) setupNumber.value = Math.round(rotation);
        }
        function isSetupLocked(deviceId) {
            return Boolean((settingsCache[deviceId] || {}).measurement_locked);
        }
        function showLockedSetupMessage(deviceId) {
            showOperation('Setup is locked', `${aliasOf(deviceId)} is measurement-locked. Hit Unlock Setup before changing camera, calibration, segmentation, or color settings. Rotation is display-only and remains editable.`, 'bad');
        }
        function selectedSetupIsLocked() {
            if (!segmentState.deviceId) return false;
            if (!isSetupLocked(segmentState.deviceId)) return false;
            showLockedSetupMessage(segmentState.deviceId);
            stopPointMode();
            return true;
        }
        function setControlLocked(control, locked) {
            if (!control) return;
            control.disabled = locked;
            control.classList.toggle('locked-control', locked);
            control.title = locked ? 'Unlock Setup before changing measurement settings.' : '';
        }
        function applySetupLockState(deviceId) {
            const locked = isSetupLocked(deviceId);
            const notice = document.getElementById('setup-lock-notice');
            if (notice) notice.classList.toggle('active', locked && segmentState.deviceId === deviceId);
            const card = document.getElementById('setup-controls-' + deviceId);
            if (card) {
                card.classList.toggle('locked', locked);
                ['zoom', 'delay', 'antibanding', 'white-balance', 'focus', 'exposure', 'iso', 'collect-night'].forEach(prefix => setControlLocked(document.getElementById(prefix + '-' + deviceId), locked));
                const auto = document.getElementById('auto-light-' + deviceId);
                setControlLocked(auto, locked);
            }
            if (segmentState.deviceId === deviceId) {
                setControlLocked(document.getElementById('expected-tags-input'), locked);
                setControlLocked(document.getElementById('color-reference-enabled'), locked);
                setControlLocked(document.getElementById('color-reference-mode'), locked);
                document.querySelectorAll('[data-locked-edit]').forEach(button => setControlLocked(button, locked));
            }
        }
        function saveDeviceSetting(deviceId, key, value) {
            if (isSetupLocked(deviceId) && key !== 'measurement_locked' && key !== 'display_rotation_deg') {
                showLockedSetupMessage(deviceId);
                applyDeviceSettings(deviceId, settingsCache[deviceId] || {});
                return Promise.resolve(settingsCache[deviceId] || {});
            }
            const payload = {};
            payload[key] = key === 'collect_night_frames' ? Boolean(value) : ((key === 'zoom_percent' || key === 'delay_ms' || key === 'exposure_compensation' || key === 'display_rotation_deg' || key === 'expected_marker_count') ? Number(value) : value);
            return fetch('/device_settings/' + deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Setting save failed');
                return body;
            }).then(settings => {
                applyDeviceSettings(deviceId, settings);
                showOperation('Camera setting saved', `${deviceId}: zoom ${settings.zoom_percent}%, settle ${(settings.delay_ms / 1000).toFixed(1)}s, exposure ${settings.exposure_compensation}, ISO ${settings.iso}, focus ${settings.focus_mode}, WB ${settings.white_balance}, night frames ${settings.collect_night_frames ? 'on' : 'off'}. It applies on the next capture.`);
                return settings;
            }).catch(error => {
                showOperation('Camera setting blocked', error.message, 'bad');
                applyDeviceSettings(deviceId, settingsCache[deviceId] || {});
                return settingsCache[deviceId] || {};
            });
        }
        function saveExpectedTags() {
            if (!segmentState.deviceId) return;
            const input = document.getElementById('expected-tags-input');
            saveDeviceSetting(segmentState.deviceId, 'expected_marker_count', input ? input.value : 4);
        }
        function saveDeviceSettings(deviceId, payload) {
            if (isSetupLocked(deviceId) && !((Object.keys(payload).length === 1) && payload.measurement_locked === false)) {
                showLockedSetupMessage(deviceId);
                return Promise.resolve(settingsCache[deviceId] || {});
            }
            return fetch('/device_settings/' + deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Setting save failed');
                return body;
            }).then(settings => {
                applyDeviceSettings(deviceId, settings);
                return settings;
            }).catch(error => {
                showOperation('Camera setting blocked', error.message, 'bad');
                applyDeviceSettings(deviceId, settingsCache[deviceId] || {});
                return settingsCache[deviceId] || {};
            });
        }
        function setAutoLight(deviceId) {
            if (isSetupLocked(deviceId)) {
                showLockedSetupMessage(deviceId);
                return;
            }
            saveDeviceSettings(deviceId, { light_mode: 'auto', profile_name: 'auto' }).then(settings => {
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
            settingsCache[deviceId] = settings || {};
            const zoom = document.getElementById('zoom-' + deviceId);
            const zoomValue = document.getElementById('zoom-value-' + deviceId);
            const delay = document.getElementById('delay-' + deviceId);
            const delayValue = document.getElementById('delay-value-' + deviceId);
            const exposure = document.getElementById('exposure-' + deviceId);
            const exposureValue = document.getElementById('exposure-value-' + deviceId);
            const iso = document.getElementById('iso-' + deviceId);
            const whiteBalance = document.getElementById('white-balance-' + deviceId);
            const focus = document.getElementById('focus-' + deviceId);
            const antibanding = document.getElementById('antibanding-' + deviceId);
            const rotation = document.getElementById('rotation-' + deviceId);
            const rotationNumber = document.getElementById('rotation-number-' + deviceId);
            const setupRotation = document.getElementById('setup-rotation-' + deviceId);
            const setupRotationNumber = document.getElementById('setup-rotation-number-' + deviceId);
            const collectNight = document.getElementById('collect-night-' + deviceId);
            const collectNightLabel = document.getElementById('collect-night-label-' + deviceId);
            const expectedTags = document.getElementById('expected-tags-input');
            const auto = document.getElementById('auto-light-' + deviceId);
            if (zoom) zoom.value = settings.zoom_percent || 0;
            if (zoomValue) zoomValue.textContent = `${settings.zoom_percent || 0}%`;
            if (delay) delay.value = settings.delay_ms || 5000;
            if (delayValue) delayValue.textContent = `${((settings.delay_ms || 5000) / 1000).toFixed(1)}s`;
            if (exposure) exposure.value = settings.exposure_compensation || 0;
            if (exposureValue) exposureValue.textContent = settings.exposure_compensation || 0;
            if (iso) iso.value = settings.iso || 'auto';
            if (whiteBalance) whiteBalance.value = settings.white_balance || 'daylight';
            if (focus) focus.value = settings.focus_mode || 'continuous-picture';
            if (antibanding) antibanding.value = settings.antibanding || '60hz';
            if (rotation) rotation.value = settings.display_rotation_deg || 0;
            if (rotationNumber) rotationNumber.value = Math.round(settings.display_rotation_deg || 0);
            if (setupRotation) setupRotation.value = settings.display_rotation_deg || 0;
            if (setupRotationNumber) setupRotationNumber.value = Math.round(settings.display_rotation_deg || 0);
            previewRotation(deviceId, settings.display_rotation_deg || 0);
            if (collectNight) collectNight.checked = settings.collect_night_frames !== false;
            if (collectNightLabel) collectNightLabel.textContent = settings.collect_night_frames === false ? 'Skip' : 'Collect';
            if (expectedTags && segmentState.deviceId === deviceId) expectedTags.value = settings.expected_marker_count || 4;
            if (auto) {
                const luma = settings.latest_luminance === null || settings.latest_luminance === undefined ? '--' : Number(settings.latest_luminance).toFixed(0);
                auto.textContent = settings.light_mode === 'auto' ? `Auto light: ${(settings.active_light_mode || 'day').toUpperCase()} (${luma})` : 'Auto Light';
            }
            applySetupLockState(deviceId);
        }
        function loadDeviceSettings() {
            fetch('/device_settings').then(r => r.json()).then(all => {
                Object.entries(all).forEach(([deviceId, settings]) => applyDeviceSettings(deviceId, settings));
            });
        }
        function loadSegments() {
            Promise.all([
                fetch('/segments').then(r => r.json()),
                fetch('/manual_markers').then(r => r.json()),
                fetch('/color_boards').then(r => r.json())
            ]).then(([segments, manualMarkers, colorBoards]) => {
                segmentState.segments = segments || {};
                segmentState.manualMarkers = manualMarkers || {};
                segmentState.colorBoards = colorBoards || {};
                if (!segmentState.deviceId) {
                    const first = Object.keys(segmentState.segments || {})[0] || Object.keys(latestStats || {})[0] || Array.from(connectedDevices)[0];
                    if (first) selectSegmentationDevice(first);
                }
                renderAllSegmentOverlays();
                renderSegmentList();
                renderColorPatchList();
                renderGrowthCharts(latestStats);
            });
        }
        function showSetupMode(mode) {
            activeSetupMode = mode;
            document.querySelectorAll('[data-setup-mode]').forEach(button => button.classList.toggle('active', button.dataset.setupMode === mode));
            document.querySelectorAll('[data-setup-panel]').forEach(panel => panel.classList.toggle('active', panel.dataset.setupPanel === mode));
            renderSetupWorkbench();
        }
        function selectedSetupCard() {
            if (!segmentState.deviceId) return null;
            return document.getElementById('setup-controls-' + segmentState.deviceId);
        }
        function renderSetupWorkbench() {
            const card = selectedSetupCard();
            const setup = document.getElementById('setup-controls-panel');
            if (setup) setup.innerHTML = '';
            if (card) {
                setup?.appendChild(card);
                card.style.display = 'block';
            }
            renderSetupSummary();
            renderSetupChecklist();
            renderSetupQA();
            renderSetupTuneSteps();
            if (segmentState.deviceId) applySetupLockState(segmentState.deviceId);
        }
        function renderSetupSummary() {
            const root = document.getElementById('setup-summary');
            if (!root || !segmentState.deviceId) return;
            const id = segmentState.deviceId;
            const meta = deviceMetadata[id] || {};
            const info = latestStats[id] || {};
            const data = info.data || {};
            root.innerHTML = `
                <div><span>Device</span><strong>${aliasOf(id)}</strong></div>
                <div><span>Chamber / role</span><strong>${meta.chamber || '--'} / ${meta.role || '--'}</strong></div>
                <div><span>Markers</span><strong>${data.markers_found || 0} detected</strong></div>
                <div><span>Stable scale</span><strong>${info.stable_scale_px_per_mm ? fmt(Number(info.stable_scale_px_per_mm), 3) + ' px/mm' : '--'}</strong></div>
            `;
            const log = document.getElementById('selected-device-log');
            if (log) {
                log.textContent = `Device ID ${id}:\\\\n` + (deviceLogs[id] || 'No log yet.');
                log.scrollTop = log.scrollHeight;
            }
            refreshSelectedDeviceLog(id);
        }
        function refreshSelectedDeviceLog(deviceId = segmentState.deviceId) {
            if (!deviceId) return;
            const log = document.getElementById('selected-device-log');
            if (!log) return;
            fetch('/device_log/' + encodeURIComponent(deviceId)).then(r => r.json()).then(body => {
                deviceLogs[deviceId] = body.log || 'No log yet.';
                if (segmentState.deviceId === deviceId) {
                    log.textContent = `Device ID ${deviceId}:\\\\n${deviceLogs[deviceId]}`;
                    log.scrollTop = log.scrollHeight;
                }
            }).catch(() => {});
        }
        function renderSetupChecklist() {
            const root = document.getElementById('setup-checklist');
            if (!root || !segmentState.deviceId) return;
            const id = segmentState.deviceId;
            const meta = deviceMetadata[id] || {};
            const info = latestStats[id] || {};
            const data = info.data || {};
            const segments = (segmentState.segments || {})[id] || [];
            const checks = [
                ['Named', Boolean(deviceAliases[id]), aliasOf(id)],
                ['Assigned', Boolean(meta.chamber && meta.role), `${meta.chamber || '--'} / ${meta.role || '--'}`],
                ['Frame', Boolean(info.filename), info.filename || '--'],
                ['Tags', Number(data.markers_found || 0) > 0, `${data.markers_found || 0} visible`],
                ['Scale', Boolean(info.stable_scale_px_per_mm || data.scale_px_per_mm), `${fmt(Number(info.stable_scale_px_per_mm || data.scale_px_per_mm), 3)} px/mm`],
                ['Segments', segments.length > 0, `${segments.length} saved`],
                ['Locked', Boolean((settingsCache[id] || {}).measurement_locked), (settingsCache[id] || {}).measurement_locked_at || '--'],
            ];
            root.innerHTML = `<div class="event"><time>Ready</time><div class="setup-checklist">${checks.map(([name, ok, detail]) => `<div><span>${name}<br><small class="muted">${detail}</small></span><strong class="${ok ? 'ok' : 'warn'}">${ok ? 'OK' : 'Needs work'}</strong></div>`).join('')}</div></div>`;
        }
        function renderSetupQA() {
            const root = document.getElementById('setup-qa-panel');
            if (!root || !segmentState.deviceId) return;
            const id = segmentState.deviceId;
            const info = latestStats[id] || {};
            const data = info.data || {};
            const color = data.color_metrics || {};
            const scaleRejected = data.scale_rejected || ((info.history || []).slice(-1)[0] || {}).scale_rejected;
            const confidence = Number(data.markers_found || 0) > 0 && !scaleRejected && Number(data.canopy_area_mm2 || data.plant_area_mm2 || 0) > 0;
            root.innerHTML = `
                <div class="event"><time>${confidence ? 'Good' : 'Check'}</time><div><strong>Measurement confidence: ${confidence ? 'usable' : 'needs attention'}</strong><br>Markers ${data.markers_found || 0}, scale ${data.scale_px_per_mm ? fmt(Number(data.scale_px_per_mm), 3) + ' px/mm' : '--'}, canopy ${fmt(Number(data.canopy_area_mm2 || data.plant_area_mm2 || 0))} mm2, green index ${color.green_index !== undefined ? fmt(Number(color.green_index), 3) : '--'}${scaleRejected ? '<br><span class="warn">Scale was rejected on this frame.</span>' : ''}</div></div>
                <div class="event"><time>Actions</time><div class="controls"><button onclick="captureDevice('${id}')">Test Capture</button><button onclick="refreshSegmentationFrame()">Refresh Frame</button><button onclick="showSetupMode('segmentation')">Edit Segments</button></div></div>
            `;
        }
        function renderSetupTuneSteps() {
            const root = document.getElementById('setup-steps');
            if (!root || !segmentState.deviceId) return;
            const id = segmentState.deviceId;
            const settings = settingsCache[id] || {};
            const info = latestStats[id] || {};
            const data = info.data || {};
            const color = data.color_metrics || {};
            const locked = settings.measurement_locked;
            const expected = settings.expected_marker_count || 4;
            const score = settings.measurement_profile_score ? `${Math.round(Number(settings.measurement_profile_score) * 100)}%` : '--';
            root.innerHTML = `
                <div class="event"><time>1</time><div><strong>Set expected visible tags</strong><br>${expected} expected in this fixed view. The sweep keeps the profile with the best combined marker, sharpness, scale, green, and exposure score.</div></div>
                <div class="event"><time>2</time><div><strong>Run measurement-profile sweep</strong><br>${data.markers_found || 0}/${expected} markers visible. Last locked score ${score}. ${settings.measurement_profile_summary || ''}</div></div>
                <div class="event"><time>3</time><div><strong>Confirm green measurement</strong><br>Green index ${color.green_index !== undefined ? fmt(Number(color.green_index), 3) : '--'}, canopy ${fmt(Number(data.canopy_area_mm2 || data.plant_area_mm2 || 0))} mm2. Use Segment mode to remove false green.</div></div>
                <div class="event"><time>4</time><div><strong>Lock fixed geometry</strong><br>Lock Setup automatically accepts the visible scale (${fmt(Number(info.stable_scale_px_per_mm || data.scale_px_per_mm), 3)} px/mm) and native color reference. Phones and boards should then stay fixed except harvests or small tray nudges.</div></div>
                <div class="event"><time>${locked ? 'Locked' : 'Open'}</time><div><strong>${locked ? 'Setup locked for measurement' : 'Setup not locked yet'}</strong><br>${locked ? (settings.measurement_locked_at || '') : 'Lock after tags, greenmask, and motion view are acceptable.'}<br><button onclick="lockSelectedMeasurementSetup(${locked ? 'false' : 'true'})">${locked ? 'Unlock Setup' : 'Lock Setup'}</button></div></div>
            `;
        }
        function selectSegmentationDevice(deviceId) {
            segmentState.deviceId = deviceId;
            segmentState.roi = null;
            segmentState.points = [];
            document.querySelectorAll('.device-list button').forEach(button => button.classList.remove('active'));
            document.querySelectorAll('.device-list button').forEach(button => {
                if (button.textContent.includes(deviceId)) button.classList.add('active');
            });
            document.querySelectorAll('.setup-control-card').forEach(card => { card.style.display = 'none'; });
            const img = document.getElementById('segment-image');
            img.style.display = 'block';
            delete img.dataset.fallback;
            img.src = (showSetupGreenMask ? '/analysis_debug/' : '/last_frame/') + deviceId + '?t=' + Date.now();
            img.onload = () => renderSegmentationEditorOverlays();
            const crop = document.getElementById('manual-marker-crop');
            if (crop) crop.style.display = 'none';
            renderSegmentList();
            renderColorPatchList();
            renderSetupWorkbench();
            applyDeviceSettings(deviceId, settingsCache[deviceId] || {});
            loadColorReferenceSettings(deviceId);
        }
        function refreshSegmentationFrame() {
            if (segmentState.deviceId) selectSegmentationDevice(segmentState.deviceId);
        }
        function handleSetupImageError(img) {
            if (!segmentState.deviceId) {
                img.style.display = 'none';
                return;
            }
            if (img.dataset.fallback === '1') {
                img.style.display = 'none';
                return;
            }
            img.dataset.fallback = '1';
            img.style.display = 'block';
            img.src = '/last_frame/' + segmentState.deviceId + '?t=' + Date.now();
        }
        function toggleGreenMask() {
            showSetupGreenMask = !showSetupGreenMask;
            const button = document.getElementById('setup-greenmask-toggle');
            if (button) button.textContent = showSetupGreenMask ? 'Greenmask On' : 'Greenmask Off';
            refreshSegmentationFrame();
        }
        function saveDeviceAlias(deviceId, alias) {
            fetch('/device_aliases/' + encodeURIComponent(deviceId), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ alias })
            }).then(r => r.json()).then(body => {
                deviceAliases[deviceId] = body.alias || '';
                const label = document.getElementById('device-button-label-' + deviceId);
                if (label) label.textContent = aliasOf(deviceId);
                const title = document.getElementById('title-' + deviceId);
                if (title) title.textContent = aliasOf(deviceId);
                renderFleetChart(latestStats, true);
                renderGrowthCharts(latestStats, true);
                showOperation('Device name saved', `${deviceId}: ${aliasOf(deviceId)}`);
            });
        }
        function saveDeviceMetadata(deviceId, key, value) {
            const payload = {};
            payload[key] = value;
            fetch('/device_metadata/' + encodeURIComponent(deviceId), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(r => r.json()).then(body => {
                deviceMetadata[deviceId] = body.metadata || {};
                renderSetupWorkbench();
                showOperation('Device context saved', `${aliasOf(deviceId)}: ${key} updated.`);
            });
        }
        function calibrateSelectedAruco() {
            if (!segmentState.deviceId) return;
            const operationId = newOperationId('scale');
            watchSetupOperation(operationId);
            fetch('/calibrate_aruco/' + encodeURIComponent(segmentState.deviceId), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({operation_id: operationId})
            }).then(async r => {
                const body = r.headers.get('content-type')?.includes('application/json') ? await r.json() : { message: await r.text() };
                if (!r.ok) throw new Error(body.message || 'Calibration failed');
                return body;
            }).then(body => {
                showOperation('ArUco scale calibrated', `${aliasOf(segmentState.deviceId)}: stable scale set to ${fmt(Number(body.scale_px_per_mm), 3)} px/mm from ${body.source || 'current marker detection'}.`);
                updateStats();
            }).catch(e => showOperation('ArUco calibration failed', e.message, 'bad'));
        }
        function autoTuneSelectedTags() {
            if (selectedSetupIsLocked()) return;
            if (segmentState.deviceId) autoTuneTags(segmentState.deviceId);
        }
        function captureSelectedSetupFrame() {
            if (!segmentState.deviceId) return;
            captureDevice(segmentState.deviceId);
            setTimeout(() => refreshSegmentationFrame(), 1200);
        }
        function lockSelectedMeasurementSetup(lock = true) {
            if (!segmentState.deviceId) return;
            const deviceId = segmentState.deviceId;
            const operationId = newOperationId(lock ? 'lock' : 'unlock');
            watchSetupOperation(operationId);
            fetch('/lock_setup/' + encodeURIComponent(deviceId), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({lock, operation_id: operationId})
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Setup lock failed');
                return body;
            }).then(body => {
                applyDeviceSettings(deviceId, body.settings || {});
                showOperation(lock ? 'Measurement setup locked' : 'Measurement setup unlocked', `${aliasOf(deviceId)}: ${lock ? `scale ${fmt(Number(body.scale), 3)} px/mm accepted and setup marked stable.` : 'setup can be edited again.'}`);
                renderSetupWorkbench();
                updateStats();
            }).catch(e => showOperation(lock ? 'Setup lock failed' : 'Setup unlock failed', e.message, 'bad'));
        }
        function enableSegmentMode() {
            if (!segmentState.deviceId) {
                const first = Object.keys(latestStats || {})[0];
                if (first) selectSegmentationDevice(first);
            }
            if (selectedSetupIsLocked()) return;
            segmentState.enabled = true;
            segmentState.mode = 'box';
            segmentState.points = [];
            document.getElementById('segment-image')?.classList.add('draw-segment-active');
            showOperation('Segmentation draw mode', 'Drag across the selected image to define a tray or individual plant region.');
        }
        function enablePolygonSegmentMode() {
            if (!segmentState.deviceId) {
                const first = Object.keys(latestStats || {})[0];
                if (first) selectSegmentationDevice(first);
            }
            if (selectedSetupIsLocked()) return;
            segmentState.enabled = true;
            segmentState.mode = 'polygon';
            segmentState.points = [];
            document.getElementById('segment-image')?.classList.add('draw-segment-active');
            showOperation('Polygon segmentation mode', 'Click each tray corner in order, then double-click the image to finish.');
        }
        function enableManualMarkerMode() {
            if (!segmentState.deviceId) {
                const first = Object.keys(latestStats || {})[0];
                if (first) selectSegmentationDevice(first);
            }
            if (selectedSetupIsLocked()) return;
            segmentState.enabled = true;
            segmentState.mode = 'manual-roi';
            segmentState.points = [];
            segmentState.roi = null;
            document.getElementById('segment-image')?.classList.add('draw-segment-active');
            showOperation('Manual tag mode', 'First drag a box around only the visible tag. A magnified crop will open, then click the four outer corners clockwise or counter-clockwise.');
        }
        function enableIgnoreEditorMode(mode) {
            if (!segmentState.deviceId) {
                const first = Object.keys(latestStats || {})[0];
                if (first) selectSegmentationDevice(first);
            }
            if (selectedSetupIsLocked()) return;
            segmentState.enabled = true;
            segmentState.mode = mode;
            segmentState.points = [];
            document.getElementById('segment-image')?.classList.add('mask-active');
            showOperation(mode === 'ignore-polygon' ? 'Ignore polygon mode' : 'Ignore box mode', mode === 'ignore-polygon' ? 'Click around false green/artifact space, then double-click the image to finish.' : 'Drag over false green/artifact space.');
        }
        function enableColorPatchMode() {
            if (!segmentState.deviceId) {
                const first = Object.keys(latestStats || {})[0];
                if (first) selectSegmentationDevice(first);
            }
            if (selectedSetupIsLocked()) return;
            segmentState.enabled = true;
            segmentState.mode = 'color-patch';
            segmentState.points = [];
            document.getElementById('segment-image')?.classList.add('draw-segment-active');
            updateColorPatchStatus();
            showOperation('Color patch mode', 'Click around one paint swatch color area, avoiding text/tape/edges, then press Finish Color Patch.');
        }
        function updateColorPatchStatus() {
            const el = document.getElementById('color-patch-status');
            if (!el) return;
            if (segmentState.mode === 'color-patch' && segmentState.enabled) {
                el.textContent = `${segmentState.points.length} point(s) plotted. Use at least 3, then press Finish Color Patch.`;
            } else {
                el.textContent = 'No active color patch.';
            }
        }
        function finishColorPatchMode() {
            if (selectedSetupIsLocked()) return;
            if (segmentState.mode !== 'color-patch') {
                showOperation('Color patch inactive', 'Press Add Color Patch first, then click around the swatch area.', 'warn');
                return;
            }
            finishPointMode();
        }
        function segmentImagePoint(event, img) {
            const rect = img.getBoundingClientRect();
            return {
                x: (event.clientX - rect.left) * (img.naturalWidth / rect.width),
                y: (event.clientY - rect.top) * (img.naturalHeight / rect.height),
                sx: event.clientX - rect.left,
                sy: event.clientY - rect.top
            };
        }
        function startSegmentDrag(event) {
            if (!segmentState.enabled || !segmentState.deviceId) return;
            if (selectedSetupIsLocked()) return;
            event.preventDefault();
            const point = segmentImagePoint(event, event.target);
            if (segmentState.mode === 'polygon' || segmentState.mode === 'ignore-polygon' || segmentState.mode === 'color-patch') {
                segmentState.points.push(point);
                renderPointModeOverlay();
                updateColorPatchStatus();
                return;
            }
            segmentState.dragging = true;
            segmentState.start = point;
            const sel = document.getElementById('segment-selection');
            sel.style.left = point.sx + 'px';
            sel.style.top = point.sy + 'px';
            sel.style.width = '0px';
            sel.style.height = '0px';
            sel.style.display = 'block';
        }
        function moveSegmentDrag(event) {
            if (!segmentState.dragging || !['box', 'ignore-box', 'manual-roi'].includes(segmentState.mode)) return;
            event.preventDefault();
            const point = segmentImagePoint(event, event.target);
            segmentState.current = point;
            const sel = document.getElementById('segment-selection');
            sel.style.left = Math.min(segmentState.start.sx, point.sx) + 'px';
            sel.style.top = Math.min(segmentState.start.sy, point.sy) + 'px';
            sel.style.width = Math.abs(point.sx - segmentState.start.sx) + 'px';
            sel.style.height = Math.abs(point.sy - segmentState.start.sy) + 'px';
        }
        function finishSegmentDrag(event) {
            if (!segmentState.dragging || !['box', 'ignore-box', 'manual-roi'].includes(segmentState.mode)) return;
            moveSegmentDrag(event);
            const region = [
                Math.min(segmentState.start.x, segmentState.current.x),
                Math.min(segmentState.start.y, segmentState.current.y),
                Math.max(segmentState.start.x, segmentState.current.x),
                Math.max(segmentState.start.y, segmentState.current.y)
            ].map(Math.round);
            segmentState.dragging = false;
            segmentState.enabled = false;
            document.getElementById('segment-selection').style.display = 'none';
            document.getElementById('segment-image')?.classList.remove('draw-segment-active');
            if (Math.abs(region[2] - region[0]) < 10 || Math.abs(region[3] - region[1]) < 10) return;
            if (segmentState.mode === 'ignore-box') {
                saveIgnoreFromEditor({ region });
                return;
            }
            if (segmentState.mode === 'manual-roi') {
                segmentState.roi = region;
                segmentState.enabled = true;
                segmentState.mode = 'manual-corners';
                segmentState.points = [];
                drawManualCrop();
                showOperation('Manual tag crop ready', 'Click the four outer marker corners on the magnified crop. Order should go around the perimeter, clockwise or counter-clockwise.');
                return;
            }
            const name = prompt('Segment name', `Tray ${(segmentState.segments[segmentState.deviceId] || []).length + 1}`);
            if (!name) return;
            fetch('/segments/' + segmentState.deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name, region })
            }).then(r => r.json()).then(() => {
                showOperation('Segment saved', `${segmentState.deviceId}: ${name}`);
                loadSegments();
                updateStats();
            });
        }
        function finishPointMode() {
            if (!segmentState.enabled || !segmentState.deviceId) return;
            if (selectedSetupIsLocked()) return;
            if (segmentState.mode === 'polygon') {
                if (segmentState.points.length < 3) {
                    showOperation('Polygon skipped', 'A polygon needs at least 3 points.', 'warn');
                    return;
                }
                const name = prompt('Segment name', `Tray ${(segmentState.segments[segmentState.deviceId] || []).length + 1}`);
                if (!name) return;
                const polygon = segmentState.points.map(p => [Math.round(p.x), Math.round(p.y)]);
                fetch('/segments/' + segmentState.deviceId, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ name, polygon })
                }).then(r => r.json()).then(() => {
                    stopPointMode();
                    showOperation('Polygon segment saved', `${segmentState.deviceId}: ${name}`);
                    loadSegments();
                    updateStats();
                });
                return;
            }
            if (segmentState.mode === 'ignore-polygon') {
                if (segmentState.points.length < 3) {
                    showOperation('Ignore polygon skipped', 'A polygon needs at least 3 points.', 'warn');
                    return;
                }
                const polygon = segmentState.points.map(p => [Math.round(p.x), Math.round(p.y)]);
                saveIgnoreFromEditor({ polygon });
                return;
            }
            if (segmentState.mode === 'color-patch') {
                saveColorPatchFromPoints();
                return;
            }
            if (segmentState.mode === 'manual-corners') {
                if (segmentState.points.length !== 4) {
                    showOperation('Manual tag skipped', 'Manual tag registration needs exactly 4 corner clicks.', 'warn');
                    return;
                }
                const size = Number(prompt('Marker size in mm', '60'));
                if (!Number.isFinite(size) || size <= 0) return;
                const markerId = prompt('Marker label', 'manual') || 'manual';
                const corners = segmentState.points.map(p => [Math.round(p.x), Math.round(p.y)]);
                fetch('/manual_marker/' + segmentState.deviceId, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ corners, size_mm: size, marker_id: markerId })
                }).then(r => r.json()).then(() => {
                    stopPointMode();
                    showOperation('Manual tag saved', `${segmentState.deviceId}: ${size} mm reference registered.`);
                    refreshFrame(segmentState.deviceId);
                    updateStats();
                });
            }
        }
        function stopPointMode() {
            segmentState.enabled = false;
            segmentState.dragging = false;
            segmentState.points = [];
            segmentState.roi = null;
            document.getElementById('segment-image')?.classList.remove('draw-segment-active');
            document.getElementById('segment-image')?.classList.remove('mask-active');
            closeManualTagModal();
            renderSegmentationEditorOverlays();
            updateColorPatchStatus();
        }
        function saveColorPatchFromPoints() {
            if (selectedSetupIsLocked()) return;
            if (segmentState.points.length < 3) {
                showOperation('Color patch skipped', 'A color patch needs at least 3 points.', 'warn');
                return;
            }
            const boardName = prompt('Board name', 'Wall Color Board') || 'Wall Color Board';
            const patchName = prompt('Patch name', `Patch ${(((segmentState.colorBoards[segmentState.deviceId] || [])[0] || {}).patches || []).length + 1}`) || 'Patch';
            const code = prompt('Paint code / catalog ID', '') || '';
            const role = prompt('Role', 'green_reference') || '';
            const polygon = segmentState.points.map(p => [Math.round(p.x), Math.round(p.y)]);
            fetch('/color_board/' + segmentState.deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ board_name: boardName, patch_name: patchName, code, role, polygon })
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Color patch save failed');
                return body;
            }).then(() => {
                stopPointMode();
                showOperation('Color patch saved', `${segmentState.deviceId}: ${patchName} ${code}`);
                loadSegments();
                updateStats();
            }).catch(e => showOperation('Color patch save failed', e.message, 'bad'));
        }
        function saveIgnoreFromEditor(payload) {
            if (selectedSetupIsLocked()) return;
            fetch('/ignore_region/' + segmentState.deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(() => {
                stopPointMode();
                showOperation('Ignore region saved', `${segmentState.deviceId}: ignored area removed from the green mask.`);
                refreshFrame(segmentState.deviceId);
                selectSegmentationDevice(segmentState.deviceId);
                updateStats();
            });
        }
        function clearEditorIgnores() {
            if (!segmentState.deviceId) return;
            if (selectedSetupIsLocked()) return;
            clearIgnore(segmentState.deviceId);
            selectSegmentationDevice(segmentState.deviceId);
        }
        function drawManualCrop() {
            const img = document.getElementById('segment-image');
            const canvas = document.getElementById('manual-marker-crop');
            if (!img || !canvas || !segmentState.roi) return;
            const [x1, y1, x2, y2] = segmentState.roi;
            const cropW = Math.max(1, x2 - x1);
            const cropH = Math.max(1, y2 - y1);
            document.getElementById('manual-tag-modal')?.classList.add('active');
            const modalRect = canvas.parentElement.getBoundingClientRect();
            const availableW = Math.max(640, modalRect.width - 28);
            const availableH = Math.max(420, modalRect.height - 120);
            const scale = Math.min(8, availableW / cropW, availableH / cropH);
            canvas.width = Math.max(1, Math.round(cropW * scale));
            canvas.height = Math.max(1, Math.round(cropH * scale));
            const ctx = canvas.getContext('2d');
            ctx.imageSmoothingEnabled = false;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, x1, y1, cropW, cropH, 0, 0, canvas.width, canvas.height);
            updateManualTagCount();
        }
        function manualCropClick(event) {
            if (segmentState.mode !== 'manual-corners' || !segmentState.roi) return;
            if (selectedSetupIsLocked()) return;
            const canvas = event.target;
            const rect = canvas.getBoundingClientRect();
            const [x1, y1, x2, y2] = segmentState.roi;
            const x = x1 + ((event.clientX - rect.left) / rect.width) * (x2 - x1);
            const y = y1 + ((event.clientY - rect.top) / rect.height) * (y2 - y1);
            segmentState.points.push({ x, y, sx: event.clientX - rect.left, sy: event.clientY - rect.top });
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#df7d7d';
            ctx.fillRect(event.clientX - rect.left - 4, event.clientY - rect.top - 4, 8, 8);
            ctx.fillStyle = '#fff';
            ctx.fillText(String(segmentState.points.length), event.clientX - rect.left + 6, event.clientY - rect.top + 6);
            updateManualTagCount();
        }
        function updateManualTagCount() {
            const count = document.getElementById('manual-tag-count');
            if (count) count.textContent = `${segmentState.points.length} / 4 corners`;
        }
        function resetManualTagCorners() {
            if (selectedSetupIsLocked()) return;
            segmentState.points = [];
            drawManualCrop();
        }
        function finishManualTagFromModal() {
            finishPointMode();
        }
        function closeManualTagModal() {
            document.getElementById('manual-tag-modal')?.classList.remove('active');
        }
        function clearManualTagsForSelected() {
            if (!segmentState.deviceId) return;
            if (selectedSetupIsLocked()) return;
            if (!confirm(`Remove all manual tags for ${segmentState.deviceId}?`)) return;
            fetch(`/manual_marker/${segmentState.deviceId}/clear`, { method: 'POST' }).then(() => {
                showOperation('Manual tags removed', `${segmentState.deviceId}: all manual tags removed.`);
                loadSegments();
                refreshFrame(segmentState.deviceId);
                updateStats();
            });
        }
        function clearSegmentsForSelected() {
            if (!segmentState.deviceId) return;
            if (selectedSetupIsLocked()) return;
            if (!confirm(`Remove all saved segments for ${aliasOf(segmentState.deviceId)}?`)) return;
            fetch(`/segments/${encodeURIComponent(segmentState.deviceId)}`, { method: 'DELETE' }).then(() => {
                showOperation('Segments cleared', `${aliasOf(segmentState.deviceId)}: all segments removed.`);
                loadSegments();
                refreshFrame(segmentState.deviceId);
                updateStats();
            });
        }
        function renderPointModeOverlay() {
            const overlay = document.getElementById('segment-overlays');
            const img = document.getElementById('segment-image');
            if (!overlay || !img || !img.naturalWidth) return;
            renderSegmentationEditorOverlays();
            const rect = img.getBoundingClientRect();
            const sx = rect.width / Math.max(1, img.naturalWidth);
            const sy = rect.height / Math.max(1, img.naturalHeight);
            const points = segmentState.points.map(p => [p.x * sx, p.y * sy]);
            const stroke = segmentState.mode === 'color-patch' ? '#d8ad5f' : '#77b7c5';
            const poly = points.length > 1 ? `<svg class="segment-box" style="left:0;top:0;width:${rect.width}px;height:${rect.height}px;padding:0;border:0;background:transparent" viewBox="0 0 ${rect.width} ${rect.height}" preserveAspectRatio="none"><polyline points="${points.map(p => p.join(',')).join(' ')}" fill="none" stroke="${stroke}" stroke-width="2"></polyline></svg>` : '';
            overlay.innerHTML += poly + segmentState.points.map((p, idx) => `<div class="segment-box" style="left:${p.x * sx - 5}px;top:${p.y * sy - 5}px;width:10px;height:10px;border-color:${stroke}">${idx + 1}</div>`).join('');
        }
        function renderSegmentBoxes(container, img, segments, colorBoards = []) {
            if (!container || !img || !img.naturalWidth) return;
            const rect = img.getBoundingClientRect();
            const sx = rect.width / Math.max(1, img.naturalWidth);
            const sy = rect.height / Math.max(1, img.naturalHeight);
            const segmentHtml = (segments || []).map(seg => {
                const [x1, y1, x2, y2] = seg.region || [0, 0, 0, 0];
                if (seg.polygon && seg.polygon.length >= 3) {
                    const xs = seg.polygon.map(p => p[0]);
                    const ys = seg.polygon.map(p => p[1]);
                    const minX = Math.min(...xs), minY = Math.min(...ys);
                    const points = seg.polygon.map(p => `${(p[0] - minX) * sx},${(p[1] - minY) * sy}`).join(' ');
                    return `<svg class="segment-box" style="left:${minX * sx}px;top:${minY * sy}px;width:${(Math.max(...xs) - minX) * sx}px;height:${(Math.max(...ys) - minY) * sy}px;padding:0" viewBox="0 0 ${(Math.max(...xs) - minX) * sx} ${(Math.max(...ys) - minY) * sy}" preserveAspectRatio="none"><polygon points="${points}" fill="rgba(119,183,197,.12)" stroke="#77b7c5" stroke-width="2"></polygon><text x="4" y="14" fill="#fff">${seg.name || seg.id}</text></svg>`;
                }
                return `<div class="segment-box" title="${seg.name || seg.id}" style="left:${x1 * sx}px;top:${y1 * sy}px;width:${(x2 - x1) * sx}px;height:${(y2 - y1) * sy}px"></div>`;
            }).join('');
            const colorHtml = (colorBoards || []).flatMap(board => (board.patches || []).map(patch => {
                const pts = patch.polygon || [];
                if (pts.length < 3) return '';
                const xs = pts.map(p => p[0]);
                const ys = pts.map(p => p[1]);
                const minX = Math.min(...xs), minY = Math.min(...ys);
                const points = pts.map(p => `${(p[0] - minX) * sx},${(p[1] - minY) * sy}`).join(' ');
                return `<svg class="segment-box" style="left:${minX * sx}px;top:${minY * sy}px;width:${(Math.max(...xs) - minX) * sx}px;height:${(Math.max(...ys) - minY) * sy}px;padding:0;border-color:#d8ad5f" viewBox="0 0 ${(Math.max(...xs) - minX) * sx} ${(Math.max(...ys) - minY) * sy}" preserveAspectRatio="none"><polygon points="${points}" fill="rgba(216,173,95,.12)" stroke="#d8ad5f" stroke-width="2"></polygon><text x="4" y="14" fill="#fff">${patch.name || patch.code || 'color'}</text></svg>`;
            })).join('');
            container.innerHTML = segmentHtml + colorHtml;
        }
        function renderSegmentationEditorOverlays() {
            renderSegmentBoxes(document.getElementById('segment-overlays'), document.getElementById('segment-image'), segmentState.segments[segmentState.deviceId] || [], segmentState.colorBoards[segmentState.deviceId] || []);
        }
        function renderAllSegmentOverlays() {
            Object.entries(segmentState.segments || {}).forEach(([deviceId, segments]) => {
                const img = document.getElementById('analysis-' + deviceId);
                const overlay = document.getElementById('segment-overlay-' + deviceId);
                renderSegmentBoxes(overlay, img, segments);
            });
            renderSegmentationEditorOverlays();
        }
        function renderSegmentList() {
            const list = document.getElementById('segment-list');
            if (!list) return;
            const deviceId = segmentState.deviceId || Object.keys(segmentState.segments || {})[0];
            const segments = (segmentState.segments || {})[deviceId] || [];
            const markers = (segmentState.manualMarkers || {})[deviceId] || [];
            const segmentHtml = segments.length ? segments.map(seg => `<div class="event"><time>${deviceId}</time><div><strong>${seg.name}</strong><br>${(seg.polygon ? 'polygon' : (seg.region || []).join(', '))}<br><button type="button" onclick="deleteSegment(${JSON.stringify(deviceId)},${JSON.stringify(seg.id)})">Delete</button></div></div>`).join('') : '<div class="event"><time>None</time><div>No segments saved for the selected camera.</div></div>';
            const markerHtml = markers.length ? markers.map((marker, idx) => `<div class="event"><time>Tag</time><div><strong>${marker.id || 'manual'}</strong><br>${marker.size_mm} mm manual marker<br><button onclick="deleteManualMarker('${deviceId}','${marker.uid || idx}')">Remove Tag</button></div></div>`).join('') : '';
            list.innerHTML = segmentHtml + markerHtml;
        }
        function renderColorPatchList() {
            const list = document.getElementById('color-patch-list');
            if (!list) return;
            const deviceId = segmentState.deviceId || Object.keys(segmentState.colorBoards || {})[0];
            const boards = (segmentState.colorBoards || {})[deviceId] || [];
            const sampled = (((latestStats[deviceId] || {}).data || {}).color_boards || []);
            const rows = boards.flatMap(board => (board.patches || []).map((patch, idx) => {
                const sampledPatch = ((sampled.find(b => b.name === board.name) || {}).patches || []).find(p => p.uid === patch.uid) || {};
                const rgb = sampledPatch.mean_rgb ? sampledPatch.mean_rgb.map(v => Math.round(v)).join(', ') : '--';
                return `<div class="event"><time>${board.name || 'Board'}</time><div><strong>${patch.name || 'Patch'}</strong><br>${patch.code || '--'} ${patch.role ? '(' + patch.role + ')' : ''}<br>RGB ${rgb}, samples ${sampledPatch.samples || '--'}<br><button onclick="deleteColorPatch(${JSON.stringify(deviceId)},${JSON.stringify(board.name || 'Color Board')},${JSON.stringify(patch.uid || idx)})">Remove Patch</button></div></div>`;
            }));
            list.innerHTML = rows.length ? rows.join('') : '<div class="event"><time>None</time><div>No color patches saved for this camera.</div></div>';
            renderColorReferenceStatus();
        }
        function applyColorReferenceSettings(deviceId, reference) {
            colorReferenceCache[deviceId] = reference || {};
            if (segmentState.deviceId !== deviceId) return;
            const enabled = document.getElementById('color-reference-enabled');
            const enabledLabel = document.getElementById('color-reference-enabled-label');
            const mode = document.getElementById('color-reference-mode');
            if (enabled) enabled.checked = reference.enabled !== false;
            if (enabledLabel) enabledLabel.textContent = reference.enabled === false ? 'Off' : 'On';
            if (mode) mode.value = reference.mode || 'off';
            renderColorReferenceStatus();
            applySetupLockState(deviceId);
        }
        function loadColorReferenceSettings(deviceId) {
            if (!deviceId) return;
            fetch('/color_reference/' + encodeURIComponent(deviceId))
                .then(r => r.json())
                .then(ref => applyColorReferenceSettings(deviceId, ref))
                .catch(() => {});
        }
        function saveColorReferenceSettings() {
            if (!segmentState.deviceId) return;
            if (selectedSetupIsLocked()) {
                applyColorReferenceSettings(segmentState.deviceId, colorReferenceCache[segmentState.deviceId] || {});
                return;
            }
            const enabled = document.getElementById('color-reference-enabled')?.checked;
            const mode = document.getElementById('color-reference-mode')?.value || 'off';
            fetch('/color_reference/' + encodeURIComponent(segmentState.deviceId), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ enabled, mode })
            }).then(r => r.json()).then(body => {
                applyColorReferenceSettings(segmentState.deviceId, body.reference || {});
                updateStats();
                showOperation('Color reference saved', `${segmentState.deviceId}: ${enabled ? 'QA on' : 'QA off'}, correction ${mode}. Raw captures are unchanged.`);
            });
        }
        function setColorReferenceBaseline() {
            if (!segmentState.deviceId) return;
            if (selectedSetupIsLocked()) return;
            if (!confirm(`Use the current sampled color patches as the baseline for ${segmentState.deviceId}?`)) return;
            const enabled = document.getElementById('color-reference-enabled')?.checked;
            const mode = document.getElementById('color-reference-mode')?.value || 'off';
            fetch('/color_reference/' + encodeURIComponent(segmentState.deviceId), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ enabled, mode, set_baseline: true })
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Baseline failed');
                return body;
            }).then(body => {
                applyColorReferenceSettings(segmentState.deviceId, body.reference || {});
                updateStats();
                showOperation('Color baseline set', `${segmentState.deviceId}: current patch colors are now the drift/correction baseline.`);
            }).catch(e => showOperation('Color baseline failed', e.message, 'bad'));
        }
        function renderColorReferenceStatus() {
            const root = document.getElementById('color-reference-status');
            if (!root || !segmentState.deviceId) return;
            const ref = colorReferenceCache[segmentState.deviceId] || {};
            const correction = (((latestStats[segmentState.deviceId] || {}).data || {}).color_correction || {});
            const drift = correction.drift || {};
            const baseline = ref.baseline || {};
            const corrected = correction.active_corrected_color_metrics || {};
            const raw = (((latestStats[segmentState.deviceId] || {}).data || {}).color_metrics || {});
            const confidence = Number.isFinite(Number(drift.confidence)) ? `${Math.round(Number(drift.confidence) * 100)}%` : '--';
            root.innerHTML = `
                <div class="event"><time>${ref.mode || 'off'}</time><div><strong>Color correction</strong><br>Baseline: ${baseline.timestamp || 'not set'}<br>Drift: ${drift.status || 'no baseline'} (${confidence}), patches ${drift.patches || 0}<br>Green index raw ${raw.green_index !== undefined ? fmt(Number(raw.green_index), 3) : '--'} / corrected ${corrected.green_index !== undefined ? fmt(Number(corrected.green_index), 3) : '--'}</div></div>
            `;
        }
        function clearColorPatchesForSelected() {
            if (!segmentState.deviceId) return;
            if (selectedSetupIsLocked()) return;
            if (!confirm(`Clear all color patches for ${segmentState.deviceId}?`)) return;
            fetch(`/color_board/${segmentState.deviceId}/clear`, { method: 'POST' }).then(() => {
                showOperation('Color board cleared', `${segmentState.deviceId}: color patches removed.`);
                loadSegments();
                updateStats();
            });
        }
        function deleteColorPatch(deviceId, boardName, uid) {
            if (isSetupLocked(deviceId)) {
                showLockedSetupMessage(deviceId);
                return;
            }
            fetch(`/color_board/${encodeURIComponent(deviceId)}/${encodeURIComponent(boardName)}/${encodeURIComponent(uid)}`, { method: 'DELETE' }).then(() => {
                showOperation('Color patch removed', `${deviceId}: ${uid}`);
                loadSegments();
                updateStats();
            });
        }
        function deleteSegment(deviceId, segmentId) {
            if (isSetupLocked(deviceId)) {
                showLockedSetupMessage(deviceId);
                return;
            }
            fetch(`/segments/${encodeURIComponent(deviceId)}/${encodeURIComponent(segmentId)}`, { method: 'DELETE' })
                .then(r => r.json())
                .then(body => {
                    segmentState.segments[deviceId] = (segmentState.segments[deviceId] || []).filter(seg => seg.id !== segmentId);
                    renderAllSegmentOverlays();
                    renderSegmentList();
                    refreshFrame(deviceId);
                    if (segmentState.deviceId === deviceId) selectSegmentationDevice(deviceId);
                    showOperation(
                        body.deleted ? 'Segment deleted' : 'Segment already gone',
                        `${deviceId}: ${segmentId}`
                    );
                    loadSegments();
                    updateStats();
                })
                .catch(e => showOperation('Segment delete failed', `${deviceId}: ${e.message}`, 'bad'));
        }
        function deleteManualMarker(deviceId, uid) {
            if (isSetupLocked(deviceId)) {
                showLockedSetupMessage(deviceId);
                return;
            }
            fetch(`/manual_marker/${deviceId}/${uid}`, { method: 'DELETE' }).then(() => {
                showOperation('Manual tag removed', `${deviceId}: manual tag was removed and analysis regenerated.`);
                loadSegments();
                refreshFrame(deviceId);
                updateStats();
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
                return `<button class="marker-hotspot" title="ArUco ${marker.id}" style="left:${left}px;top:${top}px;width:${width}px;height:${height}px" onclick='openMarkerEditor(event, "${deviceId}", ${JSON.stringify(marker)})'></button>`;
            }).join('');
            let charucoHtml = '';
            if (data.charuco_bbox) {
                const b = data.charuco_bbox;
                charucoHtml = `<button class="marker-hotspot charuco-hotspot" title="ChArUco board" style="left:${b.x * scale.sx}px;top:${b.y * scale.sy}px;width:${Math.max(36, b.width * scale.sx)}px;height:${Math.max(36, b.height * scale.sy)}px" onclick='openCharucoEditor(event, "${deviceId}", ${JSON.stringify(data.charuco_target || data.calibration_target || {})})'></button>`;
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
        function metricCatalog() {
            return {
                area: { label: 'Canopy area', unit: 'mm2', value: h => Number(h.area || 0) },
                growth_speed: { label: 'Growth speed', unit: 'mm2/hr', series: buildGrowthSpeedSeries },
                green_index: { label: 'Green index', unit: 'ExG', value: h => colorMetricValue(h, 'green_index') },
                canopy_coverage: { label: 'Canopy coverage', unit: '%', value: h => Number.isFinite(Number(h.canopy_coverage)) ? Number(h.canopy_coverage) * 100 : null },
                volume_cm3: { label: 'Canopy volume', unit: 'cm3', value: h => Number.isFinite(Number(h.volume_cm3)) ? Number(h.volume_cm3) : null }
            };
        }
        function defaultGrowthControl() {
            return { windowMode: 'all', customValue: 1, customUnit: 'days', windowHours: 0, minX: '', maxX: '', minY: '', maxY: '', trim: true, outlierSensitivity: 2.5 };
        }
        function parseWindowMode(controls) {
            const mode = controls.windowMode || 'all';
            if (mode === 'all') return 0;
            if (mode === 'custom') {
                const value = Math.max(0, Number(controls.customValue || 0));
                return (controls.customUnit || 'hours') === 'days' ? value * 24 : value;
            }
            if (mode === 'slider') return Math.max(0, Number(controls.windowHours || 0));
            if (mode.endsWith('d')) return Number(mode.slice(0, -1)) * 24;
            if (mode.endsWith('h')) return Number(mode.slice(0, -1));
            return 0;
        }
        function historySpanHours(history) {
            const times = (history || []).map(h => Date.parse(h.timestamp || '')).filter(Number.isFinite).sort((a, b) => a - b);
            if (times.length < 2) return 1;
            return Math.max(1, Math.ceil((times[times.length - 1] - times[0]) / 3600000));
        }
        function historyRangeLabel(history) {
            const times = (history || []).map(h => h.timestamp || '').filter(Boolean).sort();
            if (!times.length) return 'No stored data yet';
            return `${times[0]} to ${times[times.length - 1]} (${times.length} points)`;
        }
        function filterHistoryByControls(history, controls, valueFn = h => Number(h.area || 0)) {
            const raw = (history || []).filter(h => !h.ignored);
            const newest = raw.length ? Math.max(...raw.map(h => Date.parse(h.timestamp || '')).filter(Number.isFinite)) : NaN;
            const hours = parseWindowMode(controls);
            let filtered = raw;
            if (Number.isFinite(newest) && hours > 0) {
                const cutoff = newest - hours * 3600000;
                filtered = raw.filter(h => Date.parse(h.timestamp || '') >= cutoff);
            }
            const minX = Date.parse(controls.minX || '');
            const maxX = Date.parse(controls.maxX || '');
            if (Number.isFinite(minX)) filtered = filtered.filter(h => Date.parse(h.timestamp || '') >= minX);
            if (Number.isFinite(maxX)) filtered = filtered.filter(h => Date.parse(h.timestamp || '') <= maxX);
            return trimHistoryOutliers(filtered, valueFn, controls.trim !== false, Number(controls.outlierSensitivity || 2.5));
        }
        function dataPointDensity(points) {
            return points.length > 500 ? 0 : points.length > 250 ? 1 : 2;
        }
        function downsamplePoints(points, maxPoints = MAX_CHART_POINTS) {
            const valid = (points || []).filter(p => p && (p.y === null || Number.isFinite(Number(p.y))));
            if (valid.length <= maxPoints) return valid;
            const result = [valid[0]];
            const bucketSize = (valid.length - 2) / Math.max(1, maxPoints - 2);
            for (let i = 0; i < maxPoints - 2; i++) {
                const start = Math.floor(1 + i * bucketSize);
                const end = Math.min(valid.length - 1, Math.floor(1 + (i + 1) * bucketSize));
                const bucket = valid.slice(start, end).filter(p => Number.isFinite(Number(p.y)));
                if (!bucket.length) continue;
                const pick = bucket.reduce((best, p) => Math.abs(Number(p.y)) > Math.abs(Number(best.y)) ? p : best, bucket[0]);
                result.push(pick);
            }
            result.push(valid[valid.length - 1]);
            return result;
        }
        function metricSeries(history, metricKey) {
            const metric = metricCatalog()[metricKey] || metricCatalog().area;
            if (metric.series) return metric.series(history);
            return history.map(h => ({ x: h.timestamp || '', y: metric.value(h), timestamp: h.timestamp, filename: h.filename, segmentId: null }));
        }
        function renderWindowControls(prefix, controls, maxHours, updateFnName) {
            maxHours = Math.max(24, Number(maxHours || 1));
            const customVisible = (controls.windowMode || 'all') === 'custom';
            const sliderHours = Math.max(1, Math.min(maxHours, Math.round(parseWindowMode(controls) || maxHours)));
            return `
                <div class="chart-slider-row">
                    <label class="chart-field"><span>Time scroll</span><input id="${prefix}-time-slider" type="range" min="1" max="${Math.max(1, maxHours)}" step="1" value="${sliderHours}" oninput="${updateFnName}('windowHours', this.value, false); ${updateFnName}('windowMode', 'slider', false)"><span class="range-value" id="${prefix}-time-value">${sliderHours}h</span></label>
                    <label class="chart-field"><span>Outlier sensitivity</span><input id="${prefix}-outlier-slider" type="range" min="0.5" max="5" step="0.1" value="${controls.outlierSensitivity || 2.5}" oninput="${updateFnName}('outlierSensitivity', this.value, false)"><span class="range-value" id="${prefix}-outlier-value">${fmt(Number(controls.outlierSensitivity || 2.5), 1)}</span></label>
                </div>
                <label class="chart-field"><span>Time window</span><select onchange="${updateFnName}('windowMode', this.value, true)">
                    <option value="all" ${(controls.windowMode || 'all') === 'all' ? 'selected' : ''}>All data</option>
                    <option value="1h" ${controls.windowMode === '1h' ? 'selected' : ''}>Last hour</option>
                    <option value="6h" ${controls.windowMode === '6h' ? 'selected' : ''}>Last 6 hours</option>
                    <option value="12h" ${controls.windowMode === '12h' ? 'selected' : ''}>Last 12 hours</option>
                    <option value="24h" ${controls.windowMode === '24h' ? 'selected' : ''}>Last 24 hours</option>
                    <option value="3d" ${controls.windowMode === '3d' ? 'selected' : ''}>Last 3 days</option>
                    <option value="7d" ${controls.windowMode === '7d' ? 'selected' : ''}>Last 7 days</option>
                    <option value="custom" ${customVisible ? 'selected' : ''}>Custom</option>
                </select></label>
                <label class="chart-field"><span>Outliers</span><select onchange="${updateFnName}('trim', this.value, false)"><option value="true" ${controls.trim !== false ? 'selected' : ''}>Trim outliers</option><option value="false" ${controls.trim === false ? 'selected' : ''}>Show all</option></select></label>
                <div class="chart-input-row">
                    <label class="chart-field"><span>X min</span><input type="datetime-local" value="${controls.minX || ''}" oninput="${updateFnName}('minX', this.value, false)"></label>
                    <label class="chart-field"><span>X max</span><input type="datetime-local" value="${controls.maxX || ''}" oninput="${updateFnName}('maxX', this.value, false)"></label>
                    <label class="chart-field"><span>Custom range</span><div class="custom-window"><input type="number" min="0" step="1" value="${controls.customValue || 1}" ${customVisible ? '' : 'disabled'} oninput="${updateFnName}('customValue', this.value, false)"><select ${customVisible ? '' : 'disabled'} onchange="${updateFnName}('customUnit', this.value, false)"><option value="hours" ${(controls.customUnit || 'days') === 'hours' ? 'selected' : ''}>Hours</option><option value="days" ${(controls.customUnit || 'days') === 'days' ? 'selected' : ''}>Days</option></select></div></label>
                </div>
            `;
        }
        function renderFleetControls(stats) {
            const root = document.getElementById('fleet-chart-controls');
            if (!root) return;
            const metricKey = fleetControls.metric || 'area';
            const metric = metricCatalog()[metricKey] || metricCatalog().area;
            const entries = Object.entries(stats || {}).filter(([id]) => connectedDevices.has(id));
            const histories = entries.flatMap(([, info]) => info.history || []);
            const maxHours = historySpanHours(histories);
            const values = entries.flatMap(([, info]) => metricSeries(info.history || [], metricKey).map(p => Number(p.y))).filter(v => Number.isFinite(v) && v > 0);
            const highest = Math.max(1, ...values);
            const lowest = Math.min(...values, 0);
            const yMinValue = Math.max(0, Math.min(Math.floor(highest), Number(fleetControls.minY || lowest)));
            const yMaxValue = Math.min(Math.ceil(highest), Math.max(1, Number(fleetControls.maxY || highest)));
            root.innerHTML = renderWindowControls('fleet', fleetControls, maxHours, 'setFleetControl') + `
                <div class="chart-slider-row">
                    <label class="chart-field"><span>Y min</span><input id="fleet-y-min-slider" type="range" min="0" max="${Math.ceil(highest)}" step="${Math.max(1, Math.ceil(highest / 250))}" value="${yMinValue}" oninput="setFleetControl('minY', this.value, false)"><span class="range-value" id="fleet-y-min-value">${fmt(yMinValue, 0)}</span></label>
                    <label class="chart-field"><span>Y max</span><input id="fleet-y-max-slider" type="range" min="1" max="${Math.ceil(highest)}" step="${Math.max(1, Math.ceil(highest / 250))}" value="${yMaxValue}" oninput="setFleetControl('maxY', this.value, false)"><span class="range-value" id="fleet-y-max-value">${fmt(yMaxValue, 0)}</span></label>
                </div>
                <div class="chart-input-row">
                    <label class="chart-field"><span>Y min text</span><input type="number" step="any" value="${fleetControls.minY || ''}" placeholder="${fmt(yMinValue, 0)}" oninput="setFleetControl('minY', this.value, false)"></label>
                    <label class="chart-field"><span>Y max text</span><input type="number" step="any" value="${fleetControls.maxY || ''}" placeholder="${fmt(yMaxValue, 0)}" oninput="setFleetControl('maxY', this.value, false)"></label>
                </div>
                <div class="chart-field"><span>Available data</span><strong class="muted">${historyRangeLabel(histories)}</strong></div>
            `;
        }
        function renderFleetChart(stats, rebuildControls = true) {
            const ctx = document.getElementById('fleet-chart');
            if (!ctx) return;
            if (rebuildControls) renderFleetControls(stats);
            const metricKey = fleetControls.metric || 'area';
            const metric = metricCatalog()[metricKey] || metricCatalog().area;
            const colors = ['#7ac77f', '#77b7c5', '#d8ad5f', '#df7d7d', '#b997d6', '#91c46c'];
            const datasets = Object.entries(stats || {}).filter(([id]) => connectedDevices.has(id)).map(([id, info], idx) => {
                const history = filterHistoryByControls(info.history || [], fleetControls, metric.value || (h => Number(h.area || 0)));
                const points = downsamplePoints(metricSeries(history, metricKey));
                return { label: aliasOf(id), deviceId: id, data: points, borderColor: colors[idx % colors.length], backgroundColor: 'transparent', tension: 0.25, pointRadius: dataPointDensity(points) };
            }).filter(ds => ds.data.some(p => p.y !== null && p.y !== undefined));
            const minY = Number(fleetControls.minY || 0);
            const maxY = Number(fleetControls.maxY || 0);
            const options = { yTitle: metric.unit, minY: Number.isFinite(minY) && fleetControls.minY !== '' ? minY : undefined, maxY: maxY > 0 ? maxY : undefined, onPointClick: showFleetPointPopover };
            if (!charts.fleet) {
                charts.fleet = new Chart(ctx, { type: 'line', data: { datasets }, options: chartOptions(options) });
            } else {
                charts.fleet.data.datasets = datasets;
                charts.fleet.options = chartOptions(options);
                charts.fleet.update('none');
            }
        }
        function renderGrowthCharts(stats, rebuildControls = true) {
            const root = document.getElementById('growth-charts');
            if (!root || activeView !== 'growth') return;
            const entries = Object.entries(stats || {}).filter(([id]) => connectedDevices.has(id));
            if (rebuildControls) {
                root.innerHTML = entries.map(([id, info]) => {
                    const safeId = cssEscape(id);
                    const controls = growthControls[id] || defaultGrowthControl();
                    growthControls[id] = controls;
                    const rawValues = ((info || {}).history || []).filter(h => !h.ignored).map(h => Number(h.area || 0)).filter(v => Number.isFinite(v) && v > 0);
                    const highest = Math.max(1, ...rawValues);
                    const yMinValue = Math.max(0, Math.min(Math.floor(highest), Number(controls.minY || 0)));
                    const yMaxValue = Math.min(Math.ceil(highest), Math.max(1, Number(controls.maxY || highest)));
                    const maxHours = historySpanHours((info || {}).history || []);
                    return `<div class="event" style="grid-template-columns:1fr"><div><strong>${aliasOf(id)}</strong><div class="muted">${id}</div><div class="chart-controls">${renderWindowControls('growth-' + safeId, controls, maxHours, `setGrowthControl.bind(null,'${id}')`)}<div class="chart-slider-row"><label class="chart-field"><span>Area Y min</span><input id="growth-y-min-${safeId}" type="range" min="0" max="${Math.ceil(highest)}" step="${Math.max(1, Math.ceil(highest / 250))}" value="${yMinValue}" oninput="setGrowthControl('${id}','minY',this.value,false)"><span class="range-value" id="growth-y-min-value-${safeId}">${fmt(yMinValue, 0)}</span></label><label class="chart-field"><span>Area Y max</span><input id="growth-y-max-${safeId}" type="range" min="1" max="${Math.ceil(highest)}" step="${Math.max(1, Math.ceil(highest / 250))}" value="${yMaxValue}" oninput="setGrowthControl('${id}','maxY',this.value,false)"><span class="range-value" id="growth-y-max-value-${safeId}">${fmt(yMaxValue, 0)}</span></label></div><div class="chart-input-row"><label class="chart-field"><span>Y min text</span><input type="number" step="any" value="${controls.minY || ''}" placeholder="${fmt(yMinValue, 0)}" oninput="setGrowthControl('${id}','minY',this.value,false)"></label><label class="chart-field"><span>Y max text</span><input type="number" step="any" value="${controls.maxY || ''}" placeholder="${fmt(yMaxValue, 0)}" oninput="setGrowthControl('${id}','maxY',this.value,false)"></label></div><div class="chart-field"><span>Available data</span><strong class="muted">${historyRangeLabel((info || {}).history || [])}</strong></div></div><div class="chart-wrap tall"><canvas id="area-chart-${safeId}"></canvas></div><div class="chart-wrap"><canvas id="speed-chart-${safeId}"></canvas></div><div class="chart-wrap"><canvas id="green-chart-${safeId}"></canvas></div></div></div>`;
                }).join('');
            }
            drawGrowthCharts(stats);
        }
        function drawGrowthCharts(stats) {
            const entries = Object.entries(stats || {}).filter(([id]) => connectedDevices.has(id));
            const colors = ['#7ac77f', '#77b7c5', '#d8ad5f', '#df7d7d', '#b997d6', '#91c46c'];
            entries.forEach(([id, info]) => {
                const safeId = cssEscape(id);
                const areaCtx = document.getElementById('area-chart-' + safeId);
                const speedCtx = document.getElementById('speed-chart-' + safeId);
                const greenCtx = document.getElementById('green-chart-' + safeId);
                if (!areaCtx || !speedCtx || !greenCtx) return;
                const controls = growthControls[id] || defaultGrowthControl();
                const history = filterHistoryByControls(info.history || [], controls, h => Number(h.area || 0));
                const segmentNames = new Map();
                history.forEach(h => (h.segments || []).forEach(s => segmentNames.set(s.id, s.name || s.id)));
                const yValues = history.map(h => Number(h.area || 0)).filter(v => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
                const trimmedMax = yValues.length && controls.trim !== false ? yValues[Math.max(0, Math.floor(yValues.length * 0.95) - 1)] * 1.15 : undefined;
                const manualMin = controls.minY === '' ? undefined : Number(controls.minY);
                const manualMax = Number(controls.maxY || 0);
                const areaData = downsamplePoints(history.map(h => ({ x: h.timestamp || '', y: Number(h.area || 0), timestamp: h.timestamp, filename: h.filename, segmentId: null })));
                const datasets = [{
                    label: 'Canopy area',
                    data: areaData,
                    borderColor: colors[0],
                    backgroundColor: 'transparent',
                    tension: 0.25,
                    pointRadius: dataPointDensity(areaData)
                }];
                const volumeData = downsamplePoints(history.map(h => ({ x: h.timestamp || '', y: Number.isFinite(Number(h.volume_cm3)) ? Number(h.volume_cm3) : null, timestamp: h.timestamp, filename: h.filename })));
                if (volumeData.some(p => p.y !== null)) {
                    datasets.push({ label: 'Canopy volume', data: volumeData, borderColor: colors[3], backgroundColor: 'transparent', tension: 0.25, pointRadius: dataPointDensity(volumeData) });
                }
                Array.from(segmentNames.entries()).forEach(([segId, name], idx) => {
                    const points = downsamplePoints(history.map(h => {
                        if ((h.ignored_segments || []).includes(segId)) return { x: h.timestamp || '', y: null, timestamp: h.timestamp, filename: h.filename, segmentId: segId };
                        const seg = (h.segments || []).find(s => s.id === segId);
                        return { x: h.timestamp || '', y: seg ? Number(seg.canopy_area_mm2 || 0) : null, timestamp: h.timestamp, filename: h.filename, segmentId: segId };
                    }));
                    datasets.push({ label: name, data: points, borderColor: colors[(idx + 1) % colors.length], backgroundColor: 'transparent', tension: 0.25, pointRadius: dataPointDensity(points) });
                });
                renderMetricChart('area-' + id, areaCtx, datasets, { yTitle: 'mm2', minY: Number.isFinite(manualMin) ? manualMin : undefined, maxY: manualMax > 0 ? manualMax : trimmedMax, onPointClick: (event, chart) => showGrowthPointPopover(event, chart, id) });
                const speedData = downsamplePoints(buildGrowthSpeedSeries(history));
                renderMetricChart('speed-' + id, speedCtx, [{ label: 'Growth speed', data: speedData, borderColor: colors[1], backgroundColor: 'transparent', tension: 0.25, pointRadius: dataPointDensity(speedData) }], { yTitle: 'mm2/hr', onPointClick: (event, chart) => showGrowthPointPopover(event, chart, id) });
                const greenData = downsamplePoints(history.map(h => ({ x: h.timestamp || '', y: colorMetricValue(h, 'green_index'), timestamp: h.timestamp, filename: h.filename })));
                renderMetricChart('green-' + id, greenCtx, [{ label: 'Green index', data: greenData, borderColor: colors[2], backgroundColor: 'transparent', tension: 0.25, pointRadius: dataPointDensity(greenData) }], { yTitle: 'ExG', onPointClick: (event, chart) => showGrowthPointPopover(event, chart, id) });
            });
        }
        function renderMetricChart(key, ctx, datasets, options = {}) {
            if (charts[key]) charts[key].destroy();
            charts[key] = new Chart(ctx, {
                    type: 'line',
                    data: { datasets },
                    options: chartOptions(options)
                });
        }
        function trimHistoryOutliers(history, valueFn, enabled, sensitivity = 2.5) {
            if (!enabled || history.length < 8) return history;
            const values = history.map(valueFn).filter(v => Number.isFinite(v)).sort((a, b) => a - b);
            if (values.length < 8) return history;
            const q1 = values[Math.floor(values.length * 0.25)];
            const q3 = values[Math.floor(values.length * 0.75)];
            const iqr = Math.max(1, q3 - q1);
            const factor = Math.max(0.5, Math.min(5, Number(sensitivity || 2.5)));
            const lower = q1 - factor * iqr;
            const upper = q3 + factor * iqr;
            return history.filter(h => {
                const value = valueFn(h);
                return !Number.isFinite(value) || (value >= lower && value <= upper);
            });
        }
        function colorMetricValue(entry, key) {
            const correction = entry.color_correction || {};
            const corrected = correction.active_corrected_color_metrics || {};
            if (corrected[key] !== undefined && correction.enabled !== false && correction.mode && correction.mode !== 'off') {
                const correctedValue = Number(corrected[key]);
                if (Number.isFinite(correctedValue)) return correctedValue;
            }
            const value = ((entry.color_metrics || {})[key]);
            return Number.isFinite(Number(value)) ? Number(value) : null;
        }
        function buildGrowthSpeedSeries(history) {
            return history.map((h, idx) => {
                let speed = Number(h.growth_rate_mm2_hr);
                if (!Number.isFinite(speed) && idx > 0) {
                    const prev = history[idx - 1];
                    const dt = (Date.parse(h.timestamp || '') - Date.parse(prev.timestamp || '')) / 3600000;
                    if (dt > 0.01) speed = (Number(h.area || 0) - Number(prev.area || 0)) / dt;
                }
                return { x: h.timestamp || '', y: Number.isFinite(speed) ? speed : null, timestamp: h.timestamp, filename: h.filename };
            });
        }
        function setGrowthControl(deviceId, key, value, rebuild = false) {
            growthControls[deviceId] = growthControls[deviceId] || defaultGrowthControl();
            growthControls[deviceId][key] = key === 'trim' ? value === 'true' : value;
            if (key === 'minY' || key === 'maxY') {
                const label = document.getElementById('growth-y-' + (key === 'minY' ? 'min' : 'max') + '-value-' + cssEscape(deviceId));
                if (label) label.textContent = fmt(Number(value), 0);
            }
            if (key === 'outlierSensitivity') {
                const label = document.getElementById('growth-' + cssEscape(deviceId) + '-outlier-value');
                if (label) label.textContent = fmt(Number(value), 1);
            }
            if (key === 'windowHours') {
                const label = document.getElementById('growth-' + cssEscape(deviceId) + '-time-value');
                if (label) label.textContent = `${Math.round(Number(value || 0))}h`;
            }
            renderGrowthCharts(latestStats, rebuild);
        }
        function setFleetControl(key, value, rebuild = false) {
            fleetControls[key] = key === 'trim' ? value === 'true' : value;
            if (key === 'metric') {
                fleetControls.minY = '';
                fleetControls.maxY = '';
            }
            if (key === 'minY' || key === 'maxY') {
                const label = document.getElementById('fleet-y-' + (key === 'minY' ? 'min' : 'max') + '-value');
                if (label) label.textContent = fmt(Number(value), 0);
            }
            if (key === 'outlierSensitivity') {
                const label = document.getElementById('fleet-outlier-value');
                if (label) label.textContent = fmt(Number(value), 1);
            }
            if (key === 'windowHours') {
                const label = document.getElementById('fleet-time-value');
                if (label) label.textContent = `${Math.round(Number(value || 0))}h`;
            }
            renderFleetChart(latestStats, rebuild);
        }
        function closePointPopover() {
            if (pointPopover) pointPopover.remove();
            pointPopover = null;
        }
        function showGrowthPointPopover(event, chart, deviceId) {
            closePointPopover();
            const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: false }, false);
            if (!points.length) return;
            const point = points[0];
            const dataset = chart.data.datasets[point.datasetIndex];
            const item = dataset.data[point.index];
            if (!item) return;
            const value = Number(item.y);
            const pop = document.createElement('div');
            pop.className = 'point-popover';
            pop.innerHTML = `<div class="muted">${deviceId}</div><div>${dataset.label || 'Metric'}</div><div class="value">${Number.isFinite(value) ? value.toFixed(2) : '--'}</div><div class="muted">${item.timestamp || item.filename || ''}</div><div class="popover-actions"><button onclick='ignoreGrowthPoint(${JSON.stringify(deviceId)}, ${JSON.stringify(item)})'>Ignore</button><button class="warn" onclick='deleteGrowthPoint(${JSON.stringify(deviceId)}, ${JSON.stringify(item)})'>Delete</button><button onclick="closePointPopover()">Close</button></div>`;
            document.body.appendChild(pop);
            const native = event.native || event;
            pop.style.left = Math.min(window.innerWidth - pop.offsetWidth - 12, Math.max(12, native.clientX + 10)) + 'px';
            pop.style.top = Math.min(window.innerHeight - pop.offsetHeight - 12, Math.max(12, native.clientY + 10)) + 'px';
            pointPopover = pop;
        }
        function showFleetPointPopover(event, chart) {
            const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: false }, false);
            if (!points.length) return;
            const dataset = chart.data.datasets[points[0].datasetIndex];
            showGrowthPointPopover(event, chart, dataset ? (dataset.deviceId || dataset.label) : '');
        }
        function ignoreGrowthPoint(deviceId, item) {
            fetch('/ignore_growth_point/' + deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ timestamp: item.timestamp, filename: item.filename, segment_id: item.segmentId })
            }).then(() => { closePointPopover(); updateStats(); });
        }
        function deleteGrowthPoint(deviceId, item) {
            if (!confirm(`Delete this data point for ${deviceId}?`)) return;
            fetch('/delete_growth_point/' + deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ timestamp: item.timestamp, filename: item.filename, segment_id: item.segmentId })
            }).then(() => { closePointPopover(); updateStats(); });
        }
        function chartOptions(extra = {}) {
            const yMin = Number.isFinite(Number(extra.minY)) ? Number(extra.minY) : undefined;
            const yMax = Number.isFinite(Number(extra.maxY)) ? Number(extra.maxY) : undefined;
            const usableMin = yMin !== undefined && (yMax === undefined || yMin < yMax) ? yMin : undefined;
            const usableMax = yMax !== undefined && (yMin === undefined || yMin < yMax) ? yMax : undefined;
            return {
                responsive: true,
                maintainAspectRatio: false,
                parsing: { xAxisKey: 'x', yAxisKey: 'y' },
                plugins: { legend: { display: true, position: 'bottom', labels: { color: '#cbd6d0', boxWidth: 12 } } },
                onClick: extra.onPointClick,
                scales: {
                    y: { suggestedMin: usableMin, suggestedMax: usableMax, min: usableMin, max: usableMax, title: { display: true, text: extra.yTitle || 'mm2', color: '#8c9a94' }, grid: { color: '#27312d' }, ticks: { color: '#8c9a94' } },
                    x: { grid: { display: false }, ticks: { color: '#8c9a94', maxRotation: 0, autoSkip: true, maxTicksLimit: 5 } }
                }
            };
        }
        function cssEscape(value) {
            return String(value).replace(/[^a-zA-Z0-9_-]/g, '_');
        }
        function renderTimelapses(stats) {
            if (activeView !== 'settings') return;
            const timelapses = document.getElementById('timelapse-list');
            if (!timelapses) return;
            const entries = Object.entries(stats || {});
            timelapses.innerHTML = entries.map(([id]) => `<div class="event"><time>Video</time><div><strong>${id}</strong><div class="controls"><button onclick="openTimelapse('${id}')">Open</button><button onclick="showFileLocation('video','${id}')">Show File</button><button onclick="resetTimelapse('${id}')">Archive + Reset</button></div><div id="timelapse-player-${cssEscape(id)}"></div></div></div>`).join('');
        }
        function openTimelapse(deviceId) {
            const target = document.getElementById('timelapse-player-' + cssEscape(deviceId));
            if (target) target.innerHTML = `<video controls preload="metadata" src="/video/${deviceId}?t=${Date.now()}" style="margin-top:8px"></video>`;
        }
        function loadMissionTimelapse(deviceId) {
            const video = document.getElementById('vid-' + deviceId);
            if (!video) return;
            video.src = `/video/${deviceId}?t=${Date.now()}`;
            video.load();
            video.play().catch(() => {});
        }
        function makeCustomTimelapse() {
            const device = document.getElementById('custom-timelapse-device')?.value;
            const start = document.getElementById('custom-timelapse-start')?.value;
            const end = document.getElementById('custom-timelapse-end')?.value;
            const status = document.getElementById('custom-timelapse-status');
            const player = document.getElementById('custom-timelapse-player');
            if (!device || !start || !end) {
                if (status) status.textContent = 'Choose a device, start, and end.';
                return;
            }
            if (status) status.textContent = 'Rendering custom clip...';
            if (player) player.innerHTML = '';
            fetch('/custom_timelapse/' + encodeURIComponent(device), {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ start, end })
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Custom render failed');
                return body;
            }).then(d => {
                if (status) status.textContent = `Rendered ${d.frames} frames: ${d.filename}`;
                if (player) player.innerHTML = `<video controls preload="metadata" src="${d.url}?t=${Date.now()}" style="margin-top:8px"></video>`;
            }).catch(e => {
                if (status) status.textContent = e.message;
            });
        }
        function showFileLocation(kind, deviceId) {
            fetch(`/show_location/${kind}/${deviceId}`, { method: 'POST' }).then(r => r.json()).then(d => showOperation('File location opened', d.path || deviceId));
        }
        function resetTimelapse(deviceId) {
            if (!confirm(`Archive existing captures/video and start a fresh timelapse for ${deviceId}?`)) return;
            fetch(`/reset_timelapse/${deviceId}`, { method: 'POST' }).then(r => r.json()).then(d => {
                showOperation('Timelapse reset', `${deviceId}: previous data archived at ${d.archive}.`);
                updateStats();
            });
        }
        function clearMetricHistory() {
            if (!confirm('Clear derived metric history? Raw capture images will be kept.')) return;
            fetch('/metrics/clear', { method: 'POST' }).then(r => r.json()).then(d => {
                showOperation('Metric history cleared', d.message || 'Derived metrics cleared.');
                latestStats = {};
                updateMetricStoreStatus();
                updateStats();
            });
        }
        function startMetricBackfill() {
            if (!confirm('Rebuild metric history from existing capture images? This can take a long time.')) return;
            fetch('/metrics/backfill', { method: 'POST' }).then(r => r.json()).then(d => {
                showOperation('Metric backfill', d.message || 'Backfill started.');
                updateMetricStoreStatus();
            });
        }
        function updateMetricStoreStatus() {
            const el = document.getElementById('metric-store-status');
            const videoStatus = document.getElementById('video-import-status');
            if (!el && !videoStatus) return;
            fetch('/metrics/backfill').then(r => r.json()).then(s => {
                const running = Number(s.running || 0) === 1;
                const text = running
                    ? `Backfill running: ${s.processed || 0}/${s.total || 0} ${s.current_device || ''} ${s.message || ''}`
                    : `Backfill ${s.message || 'idle'}: ${s.processed || 0}/${s.total || 0}`;
                if (el) el.textContent = text;
                if (videoStatus) videoStatus.textContent = text;
            }).catch(() => {});
        }
        function autoTuneTags(deviceId) {
            if (isSetupLocked(deviceId)) {
                showLockedSetupMessage(deviceId);
                return;
            }
            const currentExpected = Number((deviceId === segmentState.deviceId ? document.getElementById('expected-tags-input')?.value : settingsCache[deviceId]?.expected_marker_count) || 4);
            const entered = prompt(`How many individual visible markers should this camera see? Count each single ArUco tag once, and count each visible marker inside a ChArUco board once.`, String(currentExpected));
            if (entered === null) return;
            const expected = Math.max(1, Math.min(50, Number(entered) || currentExpected || 4));
            if (deviceId === segmentState.deviceId) {
                const input = document.getElementById('expected-tags-input');
                if (input) input.value = expected;
            }
            if (!confirm(`Run measurement-profile calibration for ${deviceId} expecting ${expected} visible marker(s)? This will take several captures and lock the best profile.`)) return;
            const operationId = newOperationId('profile');
            watchSetupOperation(operationId);
            showOperation('Lighting calibration running', `${deviceId}: scoring tag completeness, sharpness, scale consistency, green signal, and exposure stability...`);
            fetch('/auto_tune_tags/' + deviceId, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ expected_markers: expected, lock: true, operation_id: operationId })
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Sweep failed');
                return body;
            }).then(d => {
                const trials = (d.trials || []).map(t => `#${t.trial}: ${t.markers}/${d.expected_markers} tags, score ${Math.round((t.score || 0) * 100)}%, sharp ${t.sharpness}`).join(', ');
                applyDeviceSettings(deviceId, d.best_profile || {});
                renderSetupWorkbench();
                showOperation('Lighting calibration complete', `${deviceId}: best score ${Math.round((d.best_score || 0) * 100)}%, ${d.best_markers}/${d.expected_markers} tags. ${trials}`);
                updateStats();
            }).catch(e => showOperation('Lighting calibration failed', e.message, 'bad'));
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
                            <div><span>Canopy Area</span><strong>${fmt(area)} mm2</strong></div>
                        `;
                    }
                    renderCalibrationOverlays(id, data);
                    const history = info.history || [];
                    if (history.length) {
                        const h = history[history.length - 1];
                        events.push({ time: (h.timestamp || '').split(' ')[1] || 'Recent', text: `${aliasOf(id)}: canopy ${fmt(Number(h.area || 0))} mm2` });
                    }
                }
                document.getElementById('last-capture-summary').textContent = latest ? latest.split(' ')[1] : '--';
                document.getElementById('fastest-summary').textContent = fastest || '--';
                document.getElementById('calibration-summary').textContent = calibrated + '/' + entries.length;
                renderFleetChart(stats);
                const timeline = document.getElementById('event-timeline');
                if (timeline) timeline.innerHTML = events.slice(-6).reverse().map(e => `<div class="event"><time>${e.time}</time><div>${e.text}</div></div>`).join('');
                const plants = document.getElementById('plants-list');
                if (plants) plants.innerHTML = entries.map(([id, info]) => `<div class="event"><time>${(info.timestamp || '').split(' ')[1] || '--'}</time><div><strong>${aliasOf(id)}</strong><br><span class="muted">${id}</span><br>Area ${fmt(Number((info.data || {}).plant_area_mm2 || 0))} mm2, growth ${fmt(Number(info.growth_rate_mm2_hr || 0), 2)} mm2/hr</div></div>`).join('');
                const ranking = document.getElementById('trait-ranking');
                if (ranking) ranking.innerHTML = entries.slice().sort((a, b) => Number(b[1].growth_rate_mm2_hr || 0) - Number(a[1].growth_rate_mm2_hr || 0)).map(([id, info], idx) => {
                    const movement = (info.data || {}).movement || (((info.history || []).slice(-1)[0] || {}).movement) || {};
                    return `<div class="event"><time>#${idx + 1}</time><div><strong>${id}</strong><br>Grow speed ${fmt(Number(info.growth_rate_mm2_hr || 0), 2)} mm2/hr<br>Canopy movement ${fmt(Number(movement.displacement_mm || 0), 2)} mm (${fmt(Number(movement.speed_mm_hr || 0), 2)} mm/hr)<br>Recovery can now be compared against logged harvest/treatment events.</div></div>`;
                }).join('');
                renderTimelapses(stats);
                renderGrowthCharts(stats);
                renderMovementWorkspace(stats);
                renderAllSegmentOverlays();
                renderColorPatchList();
                renderSetupWorkbench();
                updateMetricStoreStatus();
                const calibration = document.getElementById('calibration-list');
                if (calibration) calibration.innerHTML = entries.map(([id, info]) => `<div class="event"><time>${((info.data || {}).markers_found || 0) ? 'Seen' : 'Missing'}</time><div>${id}: ${((info.data || {}).markers_found || 0)} markers, scale ${((info.data || {}).scale_px_per_mm || '--')} px/mm</div></div>`).join('');
                const volumeCameras = document.getElementById('volume-camera-list');
                if (volumeCameras) volumeCameras.innerHTML = entries.map(([id, info]) => `<div class="event"><time><input class="volume-camera-check" type="checkbox" value="${id}" checked></time><div><strong>${id}</strong><br>${((info.data || {}).markers_found || 0)} markers, ChArUco corners ${((info.data || {}).charuco_corners_found || 0)}</div></div>`).join('');
            }).catch(e => console.error("Update Stats Error:", e));
        }
        function loadNetworkCameraStatus(probe = false) {
            fetch('/network_camera_status' + (probe ? '?probe=1' : '')).then(r => r.json()).then(statuses => {
                Object.assign(networkCameraStatus, statuses || {});
                renderMovementWorkspace(latestStats);
            }).catch(() => {});
        }
        function movementAxis(history) {
            const points = (history || []).map(h => (h.movement || {}).centroid_px).filter(p => Array.isArray(p) && p.length === 2);
            if (points.length < 2) return null;
            const first = points[0];
            const last = points[points.length - 1];
            const dx = Number(last[0]) - Number(first[0]);
            const dy = Number(last[1]) - Number(first[1]);
            return {dx, dy, angle: Math.atan2(dy, dx) * 180 / Math.PI, samples: points.length};
        }
        function renderMovementWorkspace(stats) {
            const root = document.getElementById('movement-list');
            if (!root) return;
            const entries = Object.entries(stats || {});
            fetch('/biology_state').then(r => r.json()).then(body => renderRhythmList(body.rhythms || {})).catch(() => {});
            root.innerHTML = entries.length ? entries.map(([id, info]) => {
                const history = info.history || [];
                const latest = ((info.data || {}).movement) || ((history.slice(-1)[0] || {}).movement) || {};
                const axis = movementAxis(history);
                const network = networkCameraStatus[id] || {};
                const speed = Number(latest.speed_mm_hr || 0);
                const droop = speed > 0.8 ? 'Watch movement spike/droop against watering events' : 'Collecting baseline';
                const brightness = axis && Math.abs(axis.dx) > Math.abs(axis.dy) * 2 ? 'Horizontal flattening/stretch signal candidate' : 'No flattening signal yet';
                const identity = axis ? `Axis ${fmt(axis.angle, 0)} deg from ${axis.samples} samples; useful for identity once per-plant segments are stable.` : 'Need more centroid samples.';
                const cameraState = network.configured ? `<br>Camera: ${network.reachable ? 'online' : 'offline'} ${network.host || ''}` : '';
                return `<div class="event"><time>${(info.timestamp || '').split(' ')[1] || '--'}</time><div><strong>${aliasOf(id)}</strong>${cameraState}<br>Movement ${fmt(Number(latest.displacement_mm || 0), 2)} mm, ${fmt(speed, 2)} mm/hr<br>Circadian: ${identity}<br>Drought/droop: ${droop}<br>Brightness/stretch: ${brightness}</div></div>`;
            }).join('') : '<div class="event"><time>Ready</time><div>Movement analysis will populate after calibrated captures produce canopy centroids.</div></div>';
        }
        function loadVideoImports() {
            const select = document.getElementById('video-import-select');
            if (!select) return;
            fetch('/video_imports').then(r => r.json()).then(body => {
                const videos = body.videos || [];
                select.innerHTML = videos.length
                    ? videos.map(v => `<option value="${v.name}" data-device="${v.device_id}">${v.name} (${v.size_mb} MB, ${v.device_id})</option>`).join('')
                    : '<option value="">No videos found</option>';
                const selected = select.options[select.selectedIndex];
                const device = document.getElementById('video-import-device');
                if (device && selected && !device.value) device.value = selected.dataset.device || '';
                select.onchange = () => {
                    const option = select.options[select.selectedIndex];
                    if (device && option) device.value = option.dataset.device || '';
                };
            }).catch(e => {
                const status = document.getElementById('video-import-status');
                if (status) status.textContent = `Video list failed: ${e.message}`;
            });
        }
        function importSelectedVideo() {
            const select = document.getElementById('video-import-select');
            const status = document.getElementById('video-import-status');
            const video = select?.value;
            if (!video) {
                if (status) status.textContent = 'Choose a video first.';
                return;
            }
            const payload = {
                video,
                device_id: document.getElementById('video-import-device')?.value || '',
                sample_seconds: document.getElementById('video-import-sample')?.value || 60,
                max_frames: document.getElementById('video-import-max')?.value || 240,
            };
            fetch('/video_imports/import', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Video import failed');
                return body;
            }).then(body => {
                if (status) status.textContent = `${body.message}. Metric backfill is running.`;
                showOperation('Video import started', `${video}: extracting frames and analyzing movement.`);
                updateMetricStoreStatus();
            }).catch(e => {
                if (status) status.textContent = e.message;
                showOperation('Video import failed', e.message, 'bad');
            });
        }
        function loadActivity() {
            fetch('/activity').then(r => r.json()).then(body => {
                const el = document.getElementById('activity-console');
                if (!el) return;
                const lines = body.lines || [];
                el.textContent = lines.length ? lines.join('\\\\n') : 'Waiting for live activity.';
                el.scrollTop = el.scrollHeight;
            }).catch(() => {});
        }
        function loadExperiments() {
            fetch('/experiments').then(r => r.json()).then(body => {
                const experiments = body.experiments || [];
                const events = body.events || [];
                const plants = body.plants || [];
                const biology = body.biology || {};
                const select = document.getElementById('experiment-event-id');
                if (select) select.innerHTML = experiments.map(item => `<option value="${item.id}">${item.name || item.id}</option>`).join('');
                const root = document.getElementById('experiments-list');
                if (root) root.innerHTML = experiments.length ? experiments.map(item => {
                    const related = events.filter(event => event.experiment_id === item.id).slice(-5).reverse();
                    const eventText = related.length ? related.map(event => `${event.timestamp} ${event.type}${event.device_id ? ` (${event.device_id})` : ''}: ${event.notes || ''}`).join('<br>') : 'No events logged.';
                    return `<div class="event"><time>${item.status || 'active'}</time><div><strong>${item.name || item.id}</strong><br>${item.hypothesis || 'No hypothesis recorded.'}<br><span class="muted">Devices: ${(item.devices || []).join(', ') || 'not assigned'}</span><br>${eventText}</div></div>`;
                }).join('') : '<div class="event"><time>Ready</time><div>Create the first experiment above, then log interventions and observations against it.</div></div>';
                renderPlantRecords(plants, biology.rhythms || {});
                renderExperimentCalendar(events, plants);
                renderBiologyAlerts(biology.alerts || []);
                renderRhythmList(biology.rhythms || {});
            }).catch(() => {});
        }
        function renderBiologyAlerts(alerts) {
            const root = document.getElementById('biology-alerts');
            if (!root) return;
            root.innerHTML = alerts.length ? alerts.slice(-12).reverse().map(alert => `<div class="event"><time>${alert.severity || 'info'}</time><div><strong>${alert.type || 'alert'}</strong><br>${alert.device_id || ''}: ${alert.message || ''}</div></div>`).join('') : '<div class="event"><time>OK</time><div>No rhythm, green-index, or droop alerts yet.</div></div>';
            alerts.slice(-6).forEach(alert => {
                const key = `${alert.type}:${alert.device_id}:${alert.message}`;
                if (seenAlertKeys.has(key)) return;
                seenAlertKeys.add(key);
                if ('Notification' in window && Notification.permission === 'granted') {
                    new Notification(`Plant alert: ${alert.type || 'change'}`, { body: `${alert.device_id || ''}: ${alert.message || ''}` });
                }
            });
        }
        function enablePlantNotifications() {
            if (!('Notification' in window)) {
                showOperation('Notifications unavailable', 'This browser window does not expose desktop notifications.', 'warn');
                return;
            }
            Notification.requestPermission().then(permission => {
                showOperation('Notifications', permission === 'granted' ? 'Plant alerts can now appear as desktop notifications.' : 'Notification permission was not granted.', permission === 'granted' ? 'normal' : 'warn');
            });
        }
        function renderRhythmList(rhythms) {
            const entries = Object.entries(rhythms || {});
            const html = entries.length ? entries.map(([id, rhythm]) => `<div class="event"><time>${fmt(Number(rhythm.period_hours || 0), 1)}h</time><div><strong>${aliasOf(id)}</strong><br>Frequency ${Number(rhythm.frequency_hz || 0).toExponential(3)} Hz, amplitude ${fmt(Number(rhythm.amplitude_px || 0), 2)} px, strength ${fmt(Number(rhythm.strength || 0), 2)}<br><span class="muted">Fingerprint: ${rhythm.fingerprint || '--'}</span></div></div>`).join('') : '<div class="event"><time>Collect</time><div>Need at least several hours of movement samples to estimate rhythm.</div></div>';
            ['rhythm-list', 'movement-rhythm-list'].forEach(id => {
                const root = document.getElementById(id);
                if (root) root.innerHTML = html;
            });
        }
        function renderPlantRecords(plants, rhythms) {
            const root = document.getElementById('plant-records-list');
            if (!root) return;
            root.innerHTML = plants.length ? plants.map(plant => {
                const planted = plant.planted_date || '';
                const germ = plant.first_germination_at || '';
                let germText = '--';
                if (planted && germ) {
                    const days = (Date.parse(germ) - Date.parse(planted)) / 86400000;
                    if (Number.isFinite(days)) germText = `${fmt(days, 1)} days`;
                }
                const rhythm = rhythms[plant.device_id] || {};
                return `<div class="event"><time>${plant.tray_id || plant.plant_id}</time><div><strong>${plant.plant_name || plant.plant_id}</strong> ${plant.variety ? `(${plant.variety})` : ''}<br>Device ${plant.device_id || '--'}, segment ${plant.segment_id || '--'}<br>Planted ${planted || '--'}, germination ${germ || '--'} (${germText})<br>Irrigation ${plant.irrigation || '--'} | Fertilizer ${plant.fertilizer || '--'} | Harvest ${plant.harvest || '--'}<br>Parents ${plant.parent_mother || '--'} x ${plant.parent_father || '--'}<br>Rhythm ${rhythm.frequency_hz ? Number(rhythm.frequency_hz).toExponential(3) + ' Hz' : 'collecting'}<br><span class="muted">${plant.other || ''}</span></div></div>`;
            }).join('') : '<div class="event"><time>Ready</time><div>Add a tray or plant record above. Use segment IDs when you want per-plant tracking later.</div></div>';
        }
        function renderExperimentCalendar(events, plants) {
            const root = document.getElementById('experiment-calendar');
            if (!root) return;
            const plantEvents = plants.flatMap(plant => {
                const rows = [];
                if (plant.planted_date) rows.push({timestamp: plant.planted_date, type: 'planted', notes: plant.plant_name || plant.plant_id, plant_id: plant.plant_id});
                if (plant.first_germination_at) rows.push({timestamp: plant.first_germination_at, type: 'germination', notes: plant.plant_name || plant.plant_id, plant_id: plant.plant_id});
                return rows;
            });
            const rows = [...events, ...plantEvents].sort((a, b) => Date.parse(a.timestamp || '') - Date.parse(b.timestamp || '')).slice(-40).reverse();
            root.innerHTML = rows.length ? rows.map(event => `<div class="event"><time>${(event.timestamp || '').split(' ')[0] || '--'}</time><div><strong>${event.type || 'event'}</strong> ${event.plant_id ? `(${event.plant_id})` : ''}<br>${event.device_id || event.tray_id || ''} ${event.segment_id || ''}<br>${event.notes || ''}</div></div>`).join('') : '<div class="event"><time>Ready</time><div>Calendar fills from planted, germination, fertilizer, irrigation, harvest, and observation events.</div></div>';
        }
        function saveExperiment() {
            const id = document.getElementById('experiment-id').value.trim();
            const name = document.getElementById('experiment-name').value.trim();
            const hypothesis = document.getElementById('experiment-hypothesis').value.trim();
            const devices = Array.from(connectedDevices);
            fetch('/experiments', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id, name, hypothesis, devices, status: 'active'})
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Experiment save failed');
                return body;
            }).then(() => {
                showOperation('Experiment saved', `${name || id}: tracking ${devices.length} connected/configured device(s).`);
                loadExperiments();
            }).catch(error => showOperation('Experiment save failed', error.message, 'bad'));
        }
        function logExperimentEvent() {
            const experimentId = document.getElementById('experiment-event-id').value;
            const type = document.getElementById('experiment-event-type').value;
            const deviceId = document.getElementById('experiment-event-device').value;
            const plantId = document.getElementById('experiment-event-plant').value.trim();
            const notes = document.getElementById('experiment-event-notes').value.trim();
            fetch('/experiment_events', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({experiment_id: experimentId, type, device_id: deviceId, plant_id: plantId, tray_id: plantId, notes})
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Event save failed');
                return body;
            }).then(() => {
                document.getElementById('experiment-event-notes').value = '';
                showOperation('Experiment event logged', `${type}${deviceId ? ` for ${deviceId}` : ''}.`);
                loadExperiments();
            }).catch(error => showOperation('Event save failed', error.message, 'bad'));
        }
        function savePlantRecord() {
            const experimentId = document.getElementById('experiment-event-id')?.value || document.getElementById('experiment-id')?.value || '';
            const payload = {
                plant_id: document.getElementById('plant-record-id').value.trim(),
                experiment_id: experimentId,
                tray_id: document.getElementById('plant-record-id').value.trim(),
                plant_name: document.getElementById('plant-record-name').value.trim(),
                variety: document.getElementById('plant-record-variety').value.trim(),
                planted_date: document.getElementById('plant-record-planted').value,
                first_germination_at: document.getElementById('plant-record-germination').value,
                device_id: document.getElementById('plant-record-device').value.trim(),
                segment_id: document.getElementById('plant-record-segment').value.trim(),
                irrigation: document.getElementById('plant-record-irrigation').value.trim(),
                fertilizer: document.getElementById('plant-record-fertilizer').value.trim(),
                harvest: document.getElementById('plant-record-harvest').value.trim(),
                parent_mother: document.getElementById('plant-record-mother').value.trim(),
                parent_father: document.getElementById('plant-record-father').value.trim(),
                other: document.getElementById('plant-record-notes').value.trim(),
            };
            fetch('/plant_records', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(async r => {
                const body = await r.json();
                if (!r.ok) throw new Error(body.message || 'Plant save failed');
                return body;
            }).then(() => {
                showOperation('Plant/tray saved', `${payload.plant_name || payload.plant_id}: record updated.`);
                loadExperiments();
            }).catch(error => showOperation('Plant save failed', error.message, 'bad'));
        }
        setInterval(() => {
            if (document.hidden) {
                pauseHiddenVideos();
                return;
            }
            refreshVisibleFrames();
            updateStats();
            loadActivity();
            if (activeView === 'experiments') loadExperiments();
        }, 30000);
        document.addEventListener('visibilitychange', () => {
            pauseHiddenVideos();
            if (!document.hidden) {
                refreshVisibleFrames();
                updateStats();
                loadActivity();
            }
        });
        loadDeviceSettings();
        loadAdbTransportStatus();
        loadNetworkCameraStatus(true);
        loadExperiments();
        loadActivity();
        loadSegments();
        updateStats();
        setInterval(loadActivity, 2500);
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
    load_device_aliases()
    load_device_metadata()
    load_video_manifest()
    load_night_skip_state()
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
