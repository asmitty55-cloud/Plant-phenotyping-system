"""Circadian rhythm metrics and rhythm-shift alerts."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from pt.core.analysis.modules import AnalysisObservation, AnalysisResult


def dominant_frequency_hz(history):
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
    uniform_values -= np.mean(uniform_values)
    amplitude = float(np.std(uniform_values))
    if amplitude <= 0:
        return None
    spectrum = np.abs(np.fft.rfft(uniform_values))
    freqs = np.fft.rfftfreq(len(uniform_values), d=step)
    mask = (freqs >= 1.0 / (48 * 3600)) & (freqs <= 1.0 / (2 * 3600))
    if not np.any(mask):
        return None
    masked_spectrum = spectrum[mask]
    masked_freqs = freqs[mask]
    index = int(np.argmax(masked_spectrum))
    frequency = float(masked_freqs[index])
    strength = float(masked_spectrum[index] / max(1e-9, np.sum(masked_spectrum)))
    return {
        "frequency_hz": frequency,
        "period_hours": (1.0 / frequency) / 3600.0 if frequency > 0 else None,
        "amplitude_px": amplitude,
        "strength": strength,
        "samples": len(samples),
        "span_hours": span / 3600.0,
        "fingerprint": f"{frequency:.9g}Hz:{amplitude:.2f}px:{strength:.2f}",
    }


class CircadianModule:
    name = "circadian"
    version = "1.0.0"

    def analyze(self, observation: AnalysisObservation) -> AnalysisResult:
        history = observation.history or []
        rhythm = dominant_frequency_hz(history)
        alerts = []
        if rhythm:
            midpoint = len(history) // 2
            if midpoint >= 8:
                old = dominant_frequency_hz(history[:midpoint])
                new = dominant_frequency_hz(history[midpoint:])
                if old and new and old.get("frequency_hz"):
                    shift = abs(new["frequency_hz"] - old["frequency_hz"]) / old["frequency_hz"]
                    if shift >= 0.25:
                        alerts.append({
                            "severity": "warn",
                            "type": "rhythm_shift",
                            "device_id": observation.device_id,
                            "message": f"Circadian rhythm shifted {shift * 100:.0f}% ({old['frequency_hz']:.3g} Hz to {new['frequency_hz']:.3g} Hz).",
                            "trigger_module": self.name,
                            "metric_basis": {
                                "name": "dominant_frequency_hz",
                                "previous_frequency_hz": old["frequency_hz"],
                                "recent_frequency_hz": new["frequency_hz"],
                                "relative_shift": shift,
                            },
                            "threshold_basis": {
                                "rule": "relative_frequency_shift >= 0.25",
                                "minimum_relative_shift": 0.25,
                            },
                        })
        return AnalysisResult(metrics={"rhythm": rhythm} if rhythm else {}, alerts=alerts)
