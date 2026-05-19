import json
import os

from pt.core.utils.path_utils import get_data_root


class SegmentationStore:
    def __init__(self):
        self.path = os.path.join(get_data_root(), "debug", "segments.json")
        self.data = {"segments": {}}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f) or {"segments": {}}
            except (OSError, json.JSONDecodeError):
                self.data = {"segments": {}}
        self.data.setdefault("segments", {})

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def list(self, device_id=None):
        if device_id:
            return self.data["segments"].get(device_id, [])
        return self.data["segments"]

    def add(self, device_id, name, region=None, polygon=None):
        self.data["segments"].setdefault(device_id, [])
        if polygon:
            xs = [int(p[0]) for p in polygon]
            ys = [int(p[1]) for p in polygon]
            region = [min(xs), min(ys), max(xs), max(ys)]
        region = region or [0, 0, 0, 0]
        segment = {
            "id": f"seg_{len(self.data['segments'][device_id]) + 1}_{int(region[0])}_{int(region[1])}",
            "name": name,
            "region": [int(v) for v in region],
        }
        if polygon:
            segment["polygon"] = [[int(p[0]), int(p[1])] for p in polygon]
        self.data["segments"][device_id].append(segment)
        self.save()
        return segment

    def delete(self, device_id, segment_id):
        self.load()
        segments = self.data["segments"].get(device_id, [])
        remaining = [s for s in segments if s.get("id") != segment_id]
        deleted = len(segments) - len(remaining)
        self.data["segments"][device_id] = remaining
        if deleted:
            self.save()
        return deleted


segmentation_store = SegmentationStore()
