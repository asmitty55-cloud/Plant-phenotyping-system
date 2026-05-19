import subprocess
import time
import os
from pt.device.calibration.phone_logger import PhoneLogger
from pt.device.capture_service import capture

def run_adb(cmd):
    """Run an ADB command and return stdout, stderr."""
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

def interrogate_phone(device_id):
    """
    APK-based interrogation.
    Verifies that PTCapture.apk can trigger and save a file to /sdcard/PTCaptures.
    """
    logger = PhoneLogger(device_id)
    logger.log("Starting interrogation (APK-based)", major=True)
    print(f"[INTERROGATION] Starting APK-based interrogation for device {device_id}...")

    # The custom APK always saves to this fixed location
    target_folder = "/sdcard/PTCaptures"

    profile = {
        "device_id": device_id,
        "save_folder": target_folder,
        "extension": "jpg",
        "delay": 5.0,
        "sd_card": False,
        "ok_required": False,
        "ok_button_coords": None,
        "shutter_success": False,
    }

    # 1. Ensure APK is installed
    if not capture.is_installed(device_id):
        logger.log("APK not found, installing...", major=False)
        if not capture.install_apk(device_id):
            logger.log("Failed to install APK during interrogation.", major=True)
            return profile

    # 2. Trigger a test capture
    test_filename = f"interrogate_{int(time.time())}.jpg"
    logger.log(f"Triggering test capture: {test_filename}", major=False)
    
    if capture.capture_on_device(device_id, test_filename):
        logger.log("Capture signal successful.", major=True)
        
        # 3. Verify file exists on device
        # Use 'ls' and check output
        out, _ = run_adb(["adb", "-s", device_id, "shell", f"ls {target_folder}/{test_filename}"])
        if test_filename in out:
            logger.log(f"Verified test file in {target_folder}", major=True)
            profile["shutter_success"] = True
            # Clean up
            run_adb(["adb", "-s", device_id, "shell", f"rm {target_folder}/{test_filename}"])
            print(f"[INTERROGATION] Success for {device_id}")
        else:
            logger.log(f"Capture signal OK, but file {test_filename} not found in {target_folder}.", major=True)
            # Sometimes a small delay is needed for storage to sync
            time.sleep(2)
            out, _ = run_adb(["adb", "-s", device_id, "shell", f"ls {target_folder}/{test_filename}"])
            if test_filename in out:
                profile["shutter_success"] = True
                run_adb(["adb", "-s", device_id, "shell", f"rm {target_folder}/{test_filename}"])
    else:
        logger.log("Capture trigger failed (timeout or error).", major=True)

    return profile
