import subprocess
import time
import os
import threading
from datetime import datetime
import json
from pt.core.utils.path_utils import get_images_dir

ADB = "adb"
REMOTE_DIR = "/sdcard/PTCaptures"
LOCAL_DIR = get_images_dir()

APK_CANDIDATES = [
    os.path.join(os.getcwd(), "ptcapture.apk"),
    r"C:\Program Files\pt\ptcapture.apk",
    r"C:\Program Files\pt\app-debug.apk",
    r"C:\Users\c_r_a\AndroidStudioProjects\PTCapture\app\build\outputs\apk\debug\app-debug.apk",
]


def resolve_apk_path():
    env_path = os.getenv("PT_CAPTURE_APK")
    if env_path and os.path.exists(env_path):
        return env_path

    for apk_path in APK_CANDIDATES:
        if os.path.exists(apk_path):
            print(f"Using APK: {apk_path}")
            return apk_path

    print("WARNING: No APK found in default locations. Please place ptcapture.apk in C:\\Program Files\\pt or set PT_CAPTURE_APK.")
    return APK_CANDIDATES[0]

APK_PATH = None


def get_apk_path():
    global APK_PATH
    if APK_PATH is None:
        APK_PATH = resolve_apk_path()
    return APK_PATH

os.makedirs(LOCAL_DIR, exist_ok=True)

# -------------------------
# ADB helpers
# -------------------------
def adb(cmd, device=None):
    base = [ADB]
    if device:
        base += ["-s", device]
    result = subprocess.run(base + cmd, capture_output=True, text=True)
    return result.stdout, result.stderr

def get_devices():
    out, _ = adb(["devices"])
    return [line.split()[0] for line in out.splitlines() if "\tdevice" in line]

# -------------------------
# APK management
# -------------------------
def is_installed(device):
    out, _ = adb(["shell", "pm", "list", "packages"], device)
    return "com.pt.capture" in out

def install_apk(device):
    apk_path = get_apk_path()
    if not os.path.exists(apk_path):
        print(f"ERROR: APK not found at {apk_path}")
        return False

    print(f"Applying bulletproof automation settings to {device}...")

    # 1. Disable all network connectivity to block Play Protect "calling home" during install
    adb(["shell", "svc", "wifi", "disable"], device)
    adb(["shell", "svc", "data", "disable"], device)

    # 2. Disable package verifiers and background scanning
    # global settings
    adb(["shell", "settings", "put", "global", "verifier_verify_adb_installs", "0"], device)
    adb(["shell", "settings", "put", "global", "package_verifier_enable", "0"], device)
    adb(["shell", "settings", "put", "global", "verifier_timeout", "0"], device)

    # secure/system settings (varies by Android version)
    adb(["shell", "settings", "put", "secure", "install_non_market_apps", "1"], device)
    adb(["shell", "settings", "put", "secure", "package_verifier_user_consent", "0"], device)

    print(f"Installing APK on {device}...")
    # -r: replace existing
    # -t: allow test packages
    # -g: grant all runtime permissions (essential for Camera/Storage)
    # -d: allow version code downgrade
    out, err = adb(["install", "-r", "-t", "-g", "-d", apk_path], device)

    if "Success" in out:
        print(f"APK installed successfully on {device}")
        return True
    else:
        # Fallback for older devices (Pre-Android 6.0 or limited shells)
        print(f"Aggressive install failed, trying legacy fallback...")
        out2, err2 = adb(["install", "-r", "-t", apk_path], device)
        # Manually grant permissions for older devices
        adb(["shell", "pm", "grant", "com.pt.capture", "android.permission.CAMERA"], device)
        adb(["shell", "pm", "grant", "com.pt.capture", "android.permission.WRITE_EXTERNAL_STORAGE"], device)

        if is_installed(device):
            print(f"APK installed successfully on {device} (legacy mode)")
            return True
        else:
            print(f"APK install failed on {device}: {out} {err} / {out2} {err2}")
            return False


def ensure_installed(device):
    if is_installed(device):
        return True
    return install_apk(device)

def uninstall_apk(device):
    print(f"Uninstalling old APK from {device}...")
    adb(["uninstall", "com.pt.capture"], device)

# -------------------------
# Capture engine
# -------------------------
def capture_on_device(device, filename, mode="jpg", delay=5000, exposure=-1, iso="auto", zoom_percent=0, focus_mode="continuous-picture", antibanding="60hz"):
    """Capture photo on single device using custom APK with advanced options"""
    try:
        # 1. Force stop app to prevent "Fail to connect to camera"
        adb(["shell", "am", "force-stop", "com.pt.capture"], device)

        # 2. Installation and automation settings are setup work, not per-capture work.
        if not ensure_installed(device):
            return False

        # 3. Create remote directory
        adb(["shell", "mkdir", "-p", REMOTE_DIR], device)

        # 4. Launch capture activity
        print(f"Starting {mode} capture on {device} (delay={delay}ms)...")

        # Wake up device and dismiss keyguard if possible (best effort for automation)
        adb(["shell", "input", "keyevent", "WAKEUP"], device)
        adb(["shell", "input", "keyevent", "82"], device) # MENU/Unlock
        # Horizontal swipe often unlocks many older/stock Android lockscreens
        adb(["shell", "input", "swipe", "200", "800", "800", "800"], device)

        cmd = ["shell", "am", "start", "-n", "com.pt.capture/.CaptureActivity",
               "--es", "name", filename,
               "--es", "mode", mode,
               "--ei", "delay", str(delay),
               "--ei", "zoomPercent", str(int(zoom_percent)),
               "--ei", "exposureCompensation", str(int(exposure)),
               "--es", "iso", str(iso),
               "--es", "focusMode", focus_mode,
               "--es", "antibanding", antibanding]

        # Clear logcat before starting to avoid reading old completion signals
        adb(["logcat", "-c"], device)
        adb(cmd, device)

        # 5. Wait for logcat signal
        return wait_for_capture_complete(device, filename)

    except Exception as e:
        print(f"Error capturing on {device}: {e}")
        return False

def wait_for_capture_complete(device, filename):
    """Wait for logcat signal that capture is complete"""
    import time
    start_time = time.time()
    print(f"Waiting for capture completion on {device}...")

    while time.time() - start_time < 30:  # 30 second timeout
        # Check logcat for completion signal
        out, _ = adb(["logcat", "-d", "-s", "PTCapture"], device)

        if f"CAPTURE_COMPLETE:{filename}" in out:
            print(f"Capture completed on {device}: {filename}")
            return True

        time.sleep(0.5)

    print(f"Timeout waiting for capture on {device}")
    return False

def pull_photo(device, filename):
    """Pull captured photo from device"""
    remote_path = f"{REMOTE_DIR}/{filename}"
    local_path = os.path.join(LOCAL_DIR, f"{device}_{filename}")

    out, err = adb(["pull", remote_path, local_path], device)
    if os.path.exists(local_path):
        print(f"Photo pulled: {local_path}")
        # Clean up remote file
        adb(["shell", "rm", remote_path], device)
        return local_path
    else:
        print(f"Failed to pull photo from {device}: {out} {err}")
        return None

def capture_focus_stack(device, base_filename, steps=5):
    """Capture focus stack on single device"""
    results = []
    for i in range(steps):
        filename = f"{base_filename}_focus_{i}.jpg"
        # In a real implementation, we'd adjust focus distance
        # For now, just capture multiple times
        if capture_on_device(device, filename):
            local_path = pull_photo(device, filename)
            if local_path:
                results.append(local_path)
        time.sleep(0.5)  # Brief delay between shots
    return results

def capture_synchronized(devices, filename_base):
    """Capture on multiple devices with synchronized timing"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def capture_device(device):
        filename = f"{filename_base}_{device}_{timestamp}.jpg"
        return capture_on_device(device, filename)

    # Start all captures simultaneously
    threads = []
    for device in devices:
        thread = threading.Thread(target=capture_device, args=(device,))
        threads.append(thread)
        thread.start()

    # Wait for completion
    results = []
    for thread in threads:
        thread.join()
        # In a real implementation, we'd collect results from threads
        # For now, just pull all files
        pass

    # Pull all captured photos
    pulled = []
    for device in devices:
        pattern = f"{filename_base}_{device}_{timestamp}.jpg"
        local_path = pull_photo(device, pattern)
        if local_path:
            pulled.append((device, local_path))

    return pulled

# -------------------------
# Main interface
# -------------------------
def capture_all_devices(mode="jpg", delay=5000, exposure=-1, iso=-1):
    """Capture photos on all connected devices in parallel with advanced options"""
    devices = get_devices()
    if not devices:
        print("No devices connected")
        return []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    def capture_device(device):
        filename = f"capture_{timestamp}"
        if capture_on_device(device, filename, mode, delay, exposure, iso):
            local_path = pull_photo(device, filename + "." + mode)
            if local_path:
                results.append((device, local_path))

    # Start parallel captures
    threads = []
    for device in devices:
        thread = threading.Thread(target=capture_device, args=(device,))
        threads.append(thread)
        thread.start()

    # Wait for all to complete
    for thread in threads:
        thread.join()

    return results

def run():
    """Main capture function - drop-in replacement for existing capture logic"""
    print("PT Capture System - Starting multi-device capture...")

    results = capture_all_devices()

    if results:
        print(f"Successfully captured {len(results)} photos:")
        for device, path in results:
            print(f"  {device}: {path}")
    else:
        print("No photos captured")

    return results

# -------------------------
# Integration helpers
# -------------------------
def get_device_info():
    """Get info about connected devices"""
    devices = get_devices()
    info = {}
    for device in devices:
        info[device] = {
            "installed": is_installed(device),
            "model": get_device_model(device)
        }
    return info

def get_device_model(device):
    """Get device model name"""
    out, _ = adb(["shell", "getprop", "ro.product.model"], device)
    return out.strip()

if __name__ == "__main__":
    run()
