import os
import json
from pt.core.analysis.charuco_catalog import wall_marker_size
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
            "manual_markers": {},
            "color_boards": {},
            "color_reference": {},
            "camera_params": {},     # device_id -> { "K": [[...]], "dist": [...], "R": [...], "T": [...] }
            "landmarks": {},         # dictionary:id -> physical size and world pose
            "world_origin_device": None # The device ID that defines the origin
        }

    def _ensure_defaults(self):
        self.data.setdefault("marker_overrides", {})
        self.data.setdefault("device_overrides", {})
        self.data.setdefault("charuco_overrides", {})
        self.data.setdefault("manual_markers", {})
        self.data.setdefault("color_boards", {})
        self.data.setdefault("color_reference", {})
        self.data.setdefault("camera_params", {})
        self.data.setdefault("landmarks", {})

    def save(self):
        os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(self.data, f, indent=4)

    @staticmethod
    def marker_key(dictionary, marker_id):
        dictionary = str(dictionary or "4X4_50").replace("DICT_", "")
        return f"{dictionary}:{int(marker_id)}"

    def get_marker_size(self, marker_id, device_id=None, dictionary="4X4_50"):
        self._ensure_defaults()
        marker_text = str(marker_id)
        marker_key = self.marker_key(dictionary, marker_id)
        # 1. Check device-specific override
        if device_id and device_id in self.data["device_overrides"]:
            dev_over = self.data["device_overrides"][device_id]
            marker_overrides = dev_over.get("marker_overrides", {})
            if marker_key in marker_overrides:
                return marker_overrides[marker_key]
            if dictionary == "4X4_50" and marker_text in marker_overrides:
                return marker_overrides[marker_text]

        # 2. Check global marker override
        if marker_key in self.data["marker_overrides"]:
            return self.data["marker_overrides"][marker_key]
        if dictionary == "4X4_50" and marker_text in self.data["marker_overrides"]:
            return self.data["marker_overrides"][marker_text]

        landmark = self.data["landmarks"].get(marker_key)
        if landmark and landmark.get("enabled", True):
            return landmark.get("size_mm")

        catalog_size = wall_marker_size(marker_id) if dictionary == "4X4_50" else None
        if catalog_size:
            return catalog_size

        return None

    def get_ignore_regions(self, device_id):
        self._ensure_defaults()
        if device_id in self.data["device_overrides"]:
            return self.data["device_overrides"][device_id].get("ignore_regions", [])
        return []

    def set_marker_size(self, marker_id, size_mm, device_id=None, dictionary="4X4_50"):
        self._ensure_defaults()
        marker_key = self.marker_key(dictionary, marker_id)
        if device_id:
            if device_id not in self.data["device_overrides"]:
                self.data["device_overrides"][device_id] = {}
            if "marker_overrides" not in self.data["device_overrides"][device_id]:
                self.data["device_overrides"][device_id]["marker_overrides"] = {}
            self.data["device_overrides"][device_id]["marker_overrides"][marker_key] = float(size_mm)
        else:
            self.data["marker_overrides"][marker_key] = float(size_mm)
        self.save()

    def upsert_landmark(self, landmark):
        self._ensure_defaults()
        dictionary = str(landmark.get("dictionary") or "6X6_250").replace("DICT_", "")
        marker_id = int(landmark["id"])
        limits = {"4X4_50": 50, "6X6_250": 250}
        if dictionary not in limits:
            raise ValueError(f"Unsupported landmark dictionary: {dictionary}")
        if not 0 <= marker_id < limits[dictionary]:
            raise ValueError(f"{dictionary} marker ID must be between 0 and {limits[dictionary] - 1}")
        key = self.marker_key(dictionary, marker_id)
        current = self.data["landmarks"].get(key, {})
        size_mm = float(landmark.get("size_mm", current.get("size_mm", 52.0)))
        if size_mm <= 0:
            raise ValueError("size_mm must be greater than zero")
        current.update({
            "dictionary": dictionary,
            "id": marker_id,
            "label": str(landmark.get("label") or current.get("label") or key),
            "size_mm": size_mm,
            "position_mm": [float(value) for value in landmark.get("position_mm", current.get("position_mm", [0, 0, 0]))],
            "rotation_deg": [float(value) for value in landmark.get("rotation_deg", current.get("rotation_deg", [0, 0, 0]))],
            "enabled": bool(landmark.get("enabled", current.get("enabled", True))),
        })
        if len(current["position_mm"]) != 3 or len(current["rotation_deg"]) != 3:
            raise ValueError("position_mm and rotation_deg must each contain three values")
        self.data["landmarks"][key] = current
        self.save()
        return dict(current)

    def get_landmark(self, dictionary, marker_id):
        self._ensure_defaults()
        landmark = self.data["landmarks"].get(self.marker_key(dictionary, marker_id))
        return dict(landmark) if landmark else None

    def list_landmarks(self, enabled_only=False):
        self._ensure_defaults()
        landmarks = [dict(value) for value in self.data["landmarks"].values()]
        if enabled_only:
            landmarks = [landmark for landmark in landmarks if landmark.get("enabled", True)]
        return sorted(landmarks, key=lambda item: (item.get("dictionary", ""), int(item.get("id", 0))))

    def delete_landmark(self, dictionary, marker_id):
        self._ensure_defaults()
        removed = self.data["landmarks"].pop(self.marker_key(dictionary, marker_id), None)
        self.save()
        return removed is not None

    def add_ignore_region(self, device_id, region):
        """region is [x1, y1, x2, y2]"""
        self._ensure_defaults()
        if device_id not in self.data["device_overrides"]:
            self.data["device_overrides"][device_id] = {}
        if "ignore_regions" not in self.data["device_overrides"][device_id]:
            self.data["device_overrides"][device_id]["ignore_regions"] = []
        self.data["device_overrides"][device_id]["ignore_regions"].append(region)
        self.save()

    def add_ignore_polygon(self, device_id, polygon):
        self._ensure_defaults()
        if device_id not in self.data["device_overrides"]:
            self.data["device_overrides"][device_id] = {}
        self.data["device_overrides"][device_id].setdefault("ignore_polygons", [])
        self.data["device_overrides"][device_id]["ignore_polygons"].append(polygon)
        self.save()

    def get_ignore_polygons(self, device_id):
        self._ensure_defaults()
        if device_id in self.data["device_overrides"]:
            return self.data["device_overrides"][device_id].get("ignore_polygons", [])
        return []

    def clear_ignore_regions(self, device_id):
        self._ensure_defaults()
        if device_id in self.data["device_overrides"]:
            self.data["device_overrides"][device_id]["ignore_regions"] = []
            self.data["device_overrides"][device_id]["ignore_polygons"] = []
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
        if "ids" in target_update and isinstance(target_update["ids"], list):
            current["ids"] = [int(v) for v in target_update["ids"]]
        if "name" in target_update:
            current["name"] = str(target_update["name"])
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

    def add_manual_marker(self, device_id, corners, size_mm, marker_id="manual"):
        self._ensure_defaults()
        marker = {
            "uid": f"manual_{len(self.data['manual_markers'].get(device_id, [])) + 1}",
            "id": str(marker_id),
            "corners": [[float(p[0]), float(p[1])] for p in corners],
            "size_mm": float(size_mm),
        }
        self.data["manual_markers"].setdefault(device_id, []).append(marker)
        self.save()
        return marker

    def get_manual_markers(self, device_id):
        self._ensure_defaults()
        return self.data["manual_markers"].get(device_id, [])

    def clear_manual_markers(self, device_id):
        self._ensure_defaults()
        self.data["manual_markers"][device_id] = []
        self.save()

    def delete_manual_marker(self, device_id, uid):
        self._ensure_defaults()
        markers = self.data["manual_markers"].get(device_id, [])
        self.data["manual_markers"][device_id] = [
            marker for idx, marker in enumerate(markers)
            if marker.get("uid") != uid and str(idx) != str(uid)
        ]
        self.save()

    def add_color_patch(self, device_id, board_name, patch_name, code, role, polygon):
        self._ensure_defaults()
        boards = self.data["color_boards"].setdefault(device_id, [])
        board = next((b for b in boards if b.get("name") == board_name), None)
        if board is None:
            board = {"name": str(board_name or "Color Board"), "patches": []}
            boards.append(board)
        patch = {
            "uid": f"patch_{len(board.get('patches', [])) + 1}",
            "name": str(patch_name or f"Patch {len(board.get('patches', [])) + 1}"),
            "code": str(code or ""),
            "role": str(role or ""),
            "polygon": [[float(p[0]), float(p[1])] for p in polygon],
        }
        board.setdefault("patches", []).append(patch)
        self.save()
        return patch

    def get_color_boards(self, device_id):
        self._ensure_defaults()
        return self.data["color_boards"].get(device_id, [])

    def clear_color_boards(self, device_id):
        self._ensure_defaults()
        self.data["color_boards"][device_id] = []
        self.save()

    def delete_color_patch(self, device_id, board_name, uid):
        self._ensure_defaults()
        boards = self.data["color_boards"].get(device_id, [])
        for board in boards:
            if board.get("name") == board_name:
                board["patches"] = [
                    patch for idx, patch in enumerate(board.get("patches", []))
                    if patch.get("uid") != uid and str(idx) != str(uid)
                ]
        self.save()

    def get_color_reference(self, device_id):
        self._ensure_defaults()
        default = {"enabled": True, "mode": "off", "baseline": None}
        current = self.data["color_reference"].get(device_id, {})
        merged = dict(default)
        if isinstance(current, dict):
            merged.update(current)
        if merged.get("mode") not in ("off", "simple", "advanced"):
            merged["mode"] = "off"
        merged["enabled"] = bool(merged.get("enabled", True))
        return merged

    def set_color_reference(self, device_id, enabled=None, mode=None, baseline=None):
        self._ensure_defaults()
        current = self.data["color_reference"].setdefault(device_id, {})
        if enabled is not None:
            current["enabled"] = bool(enabled)
        if mode is not None and mode in ("off", "simple", "advanced"):
            current["mode"] = mode
        if baseline is not None:
            current["baseline"] = baseline
        self.save()
        return self.get_color_reference(device_id)

calib_store = CalibrationStore()
