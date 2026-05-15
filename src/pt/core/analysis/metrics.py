"""Canonical plant metrics snapshot helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "plant_metrics.v1"


def build_metrics_snapshot(
    *,
    plant_id: str,
    device_id: str,
    filename: str,
    analysis: dict[str, Any],
    growth_rate_mm2_hr: float,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Map image analysis output into the versioned plant metrics schema."""
    timestamp = captured_at or datetime.now(timezone.utc)
    scale = analysis.get("scale_px_per_mm")
    canopy_area = float(analysis.get("plant_area_mm2") or 0.0)

    return {
        "schema_version": SCHEMA_VERSION,
        "plant_id": plant_id,
        "device_id": device_id,
        "captured_at": timestamp.isoformat(),
        "source_image": filename,
        "growth": {
            "canopy_area_mm2": canopy_area,
            "growth_rate_mm2_hr": float(growth_rate_mm2_hr),
            "scale_px_per_mm": float(scale) if scale else None,
            "height_mm": None,
            "leaf_count": None,
        },
        "stress": {
            "score": None,
            "signals": {
                "chlorosis_ratio": None,
                "necrosis_ratio": None,
                "wilting_index": None,
                "color_shift_index": None,
            },
        },
        "circadian": {
            "phase_hour": None,
            "amplitude": None,
            "period_hours": None,
            "light_cycle": None,
        },
        "quality": {
            "markers_found": int(analysis.get("markers_found") or 0),
            "marker_dictionary": analysis.get("dictionary"),
            "analysis_method": analysis.get("method"),
            "debug_image": analysis.get("debug_image"),
        },
    }
