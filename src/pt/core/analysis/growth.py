"""Growth-rate and centroid-motion analysis module."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from pt.core.analysis.modules import AnalysisObservation, AnalysisResult


def _elapsed_hours(previous: dict[str, Any], observation: AnalysisObservation) -> float:
    previous_set = observation.context.get("previous_observation")
    capture_time_utc = observation.context.get("capture_time_utc")
    if previous_set and capture_time_utc:
        start = datetime.fromisoformat(previous_set["captured_at_utc"])
        end = datetime.fromisoformat(capture_time_utc)
        return (end - start).total_seconds() / 3600.0
    timestamp = previous.get("timestamp")
    if not timestamp:
        return 0.0
    start = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timestamp()
    return (float(observation.context.get("now_timestamp") or 0) - start) / 3600.0


class GrowthModule:
    name = "growth"
    version = "1.0.0"

    def analyze(self, observation: AnalysisObservation) -> AnalysisResult:
        current_area = float(observation.metrics.get("plant_area_mm2") or 0.0)
        movement = {
            "centroid_px": (observation.metrics.get("canopy_bounding_box") or {}).get("center"),
            "displacement_mm": 0.0,
            "speed_mm_hr": 0.0,
        }
        previous = observation.previous
        hours = 0.0
        try:
            if previous:
                hours = _elapsed_hours(previous, observation)
        except (KeyError, TypeError, ValueError, OverflowError):
            hours = 0.0

        growth_rate = 0.0
        alerts = []
        if previous and hours > 0.01:
            growth_rate = (current_area - float(previous.get("area") or 0.0)) / hours
            current_centroid = movement.get("centroid_px")
            previous_centroid = (previous.get("movement") or {}).get("centroid_px")
            active_scale = float(observation.metrics.get("scale_px_per_mm") or 0.0)
            if current_centroid and previous_centroid and active_scale > 0:
                displacement_px = float(np.linalg.norm(
                    np.asarray(current_centroid, dtype=np.float32)
                    - np.asarray(previous_centroid, dtype=np.float32)
                ))
                movement["displacement_mm"] = displacement_px / active_scale
                movement["speed_mm_hr"] = movement["displacement_mm"] / hours

        history = observation.history or []
        latest = movement if observation.metrics.get("canopy_bounding_box") else (
            (history[-1] if history else {}).get("movement") or movement
        )
        prior = (history[-6] if len(history) >= 6 else {}).get("movement") or {}
        latest_centroid = latest.get("centroid_px")
        prior_centroid = prior.get("centroid_px")
        if (
            isinstance(latest_centroid, list)
            and isinstance(prior_centroid, list)
            and len(latest_centroid) == 2
            and len(prior_centroid) == 2
        ):
            y_shift = float(latest_centroid[1]) - float(prior_centroid[1])
            speed = float(latest.get("speed_mm_hr") or 0)
            if y_shift > 10 and speed > 0.5:
                alerts.append({
                    "severity": "warn",
                    "type": "droop",
                    "device_id": observation.device_id,
                    "message": f"Canopy centroid moved downward {y_shift:.0f}px with {speed:.2f} mm/hr movement speed.",
                    "trigger_module": self.name,
                    "metric_basis": {
                        "name": "canopy_centroid_y_shift_px_and_movement_speed_mm_hr",
                        "y_shift_px": y_shift,
                        "movement_speed_mm_hr": speed,
                    },
                    "threshold_basis": {
                        "rule": "y_shift_px > 10 and movement_speed_mm_hr > 0.5",
                        "minimum_y_shift_px": 10,
                        "minimum_movement_speed_mm_hr": 0.5,
                    },
                })

        return AnalysisResult(metrics={
            "growth_rate_mm2_hr": growth_rate,
            "movement": movement,
            "previous_capture_set_id": previous.get("capture_set_id") if observation.context.get("previous_observation") and previous else None,
        }, alerts=alerts)