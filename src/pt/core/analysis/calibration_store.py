import os
import json
from pt.core.utils.path_utils import get_data_root

CALIBRATION_FILE = os.path.join(get_data_root(), "debug", "calibration.json")

class CalibrationStore:
    def __init__(self):
        self.data = self._load()

    def _load(self):
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "marker_overrides": {},  # marker_id -> size_mm
            "device_overrides": {},  # device_id -> { "marker_id": size_mm, "ignore_regions": [[x1,y1,x2,y2], ...] }
            "charuco_overrides": {},
            "camera_params": {},     # device_id -> { "K": [[...]], "dist": [...], "R": [...], "T": [...] }
            "world_origin_device": None # The device ID that defines the origin
        }

    def _ensure_defaults(self):
        self.data.setdefault("marker_overrides", {})
        self.data.setdefault("device_overrides", {})
        self.data.setdefault("charuco_overrides", {})
        self.data.setdefault("camera_params", {})

    def save(self):
        os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(self.data, f, indent=4)

    def get_marker_size(self, marker_id, device_id=None):
        self._ensure_defaults()
        # 1. Check device-specific override
        if device_id and device_id in self.data["device_overrides"]:
            dev_over = self.data["device_overrides"][device_id]
            if str(marker_id) in dev_over.get("marker_overrides", {}):
                return dev_over["marker_overrides"][str(marker_id)]

        # 2. Check global marker override
        if str(marker_id) in self.data["marker_overrides"]:
            return self.data["marker_overrides"][str(marker_id)]

        return None

    def get_ignore_regions(self, device_id):
        self._ensure_defaults()
        if device_id in self.data["device_overrides"]:
            return self.data["device_overrides"][device_id].get("ignore_regions", [])
        return []

    def set_marker_size(self, marker_id, size_mm, device_id=None):
        self._ensure_defaults()
        if device_id:
            if device_id not in self.data["device_overrides"]:
                self.data["device_overrides"][device_id] = {}
            if "marker_overrides" not in self.data["device_overrides"][device_id]:
                self.data["device_overrides"][device_id]["marker_overrides"] = {}
            self.data["device_overrides"][device_id]["marker_overrides"][str(marker_id)] = float(size_mm)
        else:
            self.data["marker_overrides"][str(marker_id)] = float(size_mm)
        self.save()

    def add_ignore_region(self, device_id, region):
        """region is [x1, y1, x2, y2]"""
        self._ensure_defaults()
        if device_id not in self.data["device_overrides"]:
            self.data["device_overrides"][device_id] = {}
        if "ignore_regions" not in self.data["device_overrides"][device_id]:
            self.data["device_overrides"][device_id]["ignore_regions"] = []
        self.data["device_overrides"][device_id]["ignore_regions"].append(region)
        self.save()

    def clear_ignore_regions(self, device_id):
        self._ensure_defaults()
        if device_id in self.data["device_overrides"]:
            self.data["device_overrides"][device_id]["ignore_regions"] = []
            self.save()

    def get_charuco_target(self, default_target, device_id=None):
        self._ensure_defaults()
        target = dict(default_target)
        if "global" in self.data["charuco_overrides"]:
            target.update(self.data["charuco_overrides"]["global"])
        if device_id and device_id in self.data["charuco_overrides"]:
            target.update(self.data["charuco_overrides"][device_id])
        return target

    def set_charuco_target(self, target_update, device_id=None):
        self._ensure_defaults()
        key = device_id or "global"
        current = self.data["charuco_overrides"].setdefault(key, {})
        for field in ("square_size_mm", "marker_size_mm"):
            if field in target_update:
                current[field] = float(target_update[field])
        self.save()

    def set_camera_params(self, device_id, K, dist, R=None, T=None):
        self._ensure_defaults()
        self.data["camera_params"][device_id] = {
            "K": K.tolist() if hasattr(K, "tolist") else K,
            "dist": dist.tolist() if hasattr(dist, "tolist") else dist,
            "R": R.tolist() if hasattr(R, "tolist") else R,
            "T": T.tolist() if hasattr(T, "tolist") else T
        }
        self.save()

    def get_camera_params(self, device_id):
        self._ensure_defaults()
        return self.data["camera_params"].get(device_id)

calib_store = CalibrationStore()
