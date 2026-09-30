"""Baseline-relative color and greenness analysis module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pt.core.analysis.modules import AnalysisObservation, AnalysisResult


def median_metric(entries, path):
    values = []
    for entry in entries:
        current = entry
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, (int, float)):
            values.append(float(current))
    return float(np.median(values)) if values else None


def build_color_baseline(history, existing_baseline=None, max_samples=8):
    valid = [
        entry for entry in history
        if entry.get("color_metrics") and entry.get("area", 0) > 0
    ]
    if len(valid) < 3:
        return existing_baseline or {"status": "collecting", "samples": len(valid)}

    samples = valid[:max_samples]
    return {
        "status": "ready",
        "samples": len(samples),
        "canopy_area_mm2": median_metric(samples, ["area"]),
        "green_index": median_metric(samples, ["color_metrics", "green_index"]),
        "chlorosis_ratio": median_metric(samples, ["color_metrics", "chlorosis_ratio"]),
        "mean_hue": median_metric(samples, ["color_metrics", "mean_hue"]),
        "mean_saturation": median_metric(samples, ["color_metrics", "mean_saturation"]),
        "mean_value": median_metric(samples, ["color_metrics", "mean_value"]),
    }


def evaluate_nutrient_flags(color_metrics, baseline):
    if not color_metrics or not baseline or baseline.get("status") != "ready":
        return {
            "status": "baseline_collecting",
            "severity": "none",
            "score": 0.0,
            "flags": [],
            "deltas": {},
        }

    deltas = {}
    flags = []
    score = 0.0
    green_base = baseline.get("green_index")
    if green_base not in (None, 0):
        green_drop = (green_base - color_metrics.get("green_index", green_base)) / max(abs(green_base), 0.01)
        deltas["green_index_drop_fraction"] = float(green_drop)
        if green_drop > 0.18:
            flags.append("green_index_drop")
            score += min(green_drop, 0.5)

    chlorosis_base = baseline.get("chlorosis_ratio")
    if chlorosis_base is not None:
        chlorosis_rise = color_metrics.get("chlorosis_ratio", chlorosis_base) - chlorosis_base
        deltas["chlorosis_ratio_delta"] = float(chlorosis_rise)
        if chlorosis_rise > 0.12:
            flags.append("chlorosis_increase")
            score += min(chlorosis_rise * 2.5, 0.5)

    saturation_base = baseline.get("mean_saturation")
    if saturation_base not in (None, 0):
        saturation_drop = (saturation_base - color_metrics.get("mean_saturation", saturation_base)) / saturation_base
        deltas["saturation_drop_fraction"] = float(saturation_drop)
        if saturation_drop > 0.15:
            flags.append("desaturation")
            score += min(saturation_drop, 0.35)

    severity = "none"
    if score >= 0.55:
        severity = "high"
    elif score >= 0.3:
        severity = "medium"
    elif flags:
        severity = "low"
    return {
        "status": "ready",
        "severity": severity,
        "score": float(min(score, 1.0)),
        "flags": flags,
        "deltas": deltas,
    }


class ColorIndexModule:
    name = "color_index"
    version = "1.0.0"

    def analyze(self, observation: AnalysisObservation) -> AnalysisResult:
        baseline = build_color_baseline(
            observation.history,
            observation.context.get("existing_baseline"),
        )
        flags = evaluate_nutrient_flags(
            observation.metrics.get("color_metrics"),
            baseline,
        )
        alerts = []
        history = observation.history or []
        recent = [entry for entry in history[-12:] if not entry.get("ignored")]
        prior = [entry for entry in history[-60:-12] if not entry.get("ignored")]
        recent_green = [
            (entry.get("color_metrics") or {}).get("green_index")
            for entry in recent
            if isinstance((entry.get("color_metrics") or {}).get("green_index"), (int, float))
        ]
        prior_green = [
            (entry.get("color_metrics") or {}).get("green_index")
            for entry in prior
            if isinstance((entry.get("color_metrics") or {}).get("green_index"), (int, float))
        ]
        if recent_green and prior_green:
            recent_average = float(np.mean(recent_green))
            prior_average = float(np.mean(prior_green))
            if prior_average and recent_average < prior_average * 0.85:
                drop = ((prior_average - recent_average) / abs(prior_average)) * 100
                alerts.append({
                    "severity": "warn",
                    "type": "green_index_drop",
                    "device_id": observation.device_id,
                    "message": f"Green index dropped {drop:.0f}% from recent baseline.",
                    "trigger_module": self.name,
                    "metric_basis": {
                        "name": "green_index",
                        "baseline_mean": prior_average,
                        "recent_mean": recent_average,
                    },
                    "threshold_basis": {
                        "rule": "recent_mean < baseline_mean * 0.85",
                        "baseline_multiplier": 0.85,
                    },
                })
        return AnalysisResult(metrics={"baseline": baseline, "nutrient_deficiency": flags}, alerts=alerts)