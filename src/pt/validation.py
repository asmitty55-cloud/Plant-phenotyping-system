"""Reference-dataset validation for plant phenotyping measurements."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA_VERSION = "plant_validation_report.v1"
MANIFEST_SCHEMA_VERSION = "plant_validation.v1"


def mask_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    reference_mask = np.asarray(reference) > 0
    prediction_mask = np.asarray(prediction) > 0
    if reference_mask.shape != prediction_mask.shape:
        raise ValueError(
            f"Mask dimensions differ: reference {reference_mask.shape}, prediction {prediction_mask.shape}"
        )
    true_positive = int(np.count_nonzero(reference_mask & prediction_mask))
    false_positive = int(np.count_nonzero(~reference_mask & prediction_mask))
    false_negative = int(np.count_nonzero(reference_mask & ~prediction_mask))
    union = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "iou": float(true_positive / union) if union else 1.0,
        "precision": float(true_positive / precision_denominator) if precision_denominator else 1.0,
        "recall": float(true_positive / recall_denominator) if recall_denominator else 1.0,
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
    }


def regression_metrics(pairs: list[tuple[float, float]]) -> dict[str, float | int | None]:
    if not pairs:
        return {"count": 0, "mae": None, "rmse": None, "mean_absolute_percent_error": None}
    reference = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    predicted = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    absolute_error = np.abs(predicted - reference)
    valid_percent = np.abs(reference) > 1e-12
    percent_errors = absolute_error[valid_percent] / np.abs(reference[valid_percent]) * 100.0
    return {
        "count": int(len(pairs)),
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(np.square(predicted - reference)))),
        "mean_absolute_percent_error": float(np.mean(percent_errors)) if len(percent_errors) else None,
    }


def _resolve_dataset_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Dataset path escapes validation root: {relative_path}") from exc
    return candidate


def _aggregate_masks(mask_scores: list[dict[str, float | int]]) -> dict[str, Any]:
    report = {}
    for key in ("iou", "precision", "recall"):
        values = [float(score[key]) for score in mask_scores]
        report[key] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "min": float(np.min(values)) if values else None,
        }
    return report


def _acceptance_failures(metrics: dict[str, Any], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    checks = [
        ("mask_iou", "mean", "mask_iou_min", "min"),
        ("mask_precision", "mean", "mask_precision_min", "min"),
        ("mask_recall", "mean", "mask_recall_min", "min"),
        ("area_cm2", "mean_absolute_percent_error", "area_mean_absolute_percent_error_max", "max"),
        ("coverage", "mae", "coverage_mae_max", "max"),
        ("volume_cm3", "mean_absolute_percent_error", "volume_mean_absolute_percent_error_max", "max"),
    ]
    failures = []
    for metric_name, value_name, threshold_name, direction in checks:
        threshold = thresholds.get(threshold_name)
        value = (metrics.get(metric_name) or {}).get(value_name)
        if threshold is None or value is None:
            continue
        passed = value >= threshold if direction == "min" else value <= threshold
        if not passed:
            failures.append({
                "metric": metric_name,
                "observed": value,
                "threshold": threshold,
                "operator": ">=" if direction == "min" else "<=",
            })

    color_threshold = thresholds.get("color_mean_absolute_percent_error_max")
    if color_threshold is not None:
        for key, summary in (metrics.get("color") or {}).items():
            value = summary.get("mean_absolute_percent_error")
            if value is not None and value > color_threshold:
                failures.append({
                    "metric": f"color.{key}",
                    "observed": value,
                    "threshold": color_threshold,
                    "operator": "<=",
                })
    return failures


def run_validation(
    manifest_path: str | os.PathLike[str],
    acceptance_path: str | os.PathLike[str] | None = None,
    report_path: str | os.PathLike[str] | None = None,
    analyzer=None,
    reconstructor=None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    root = manifest_file.parent
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION!r}")
    samples = manifest.get("samples") or []
    reconstructions = manifest.get("reconstructions") or []
    thresholds = json.loads(Path(acceptance_path).read_text(encoding="utf-8")) if acceptance_path else {}

    if analyzer is None:
        from pt.core.analysis.image_analysis import analyze_image

        analyzer = analyze_image
    if reconstructor is None:
        from pt.core.analysis.volumetric import reconstruct_visual_hull

        reconstructor = reconstruct_visual_hull

    calibration_path = manifest.get("calibration_snapshot")
    original_calibration = None
    calibration_store = None
    if calibration_path:
        from pt.core.analysis.calibration_store import calib_store

        calibration_store = calib_store
        original_calibration = calib_store.data
        calib_store.data = json.loads(
            _resolve_dataset_path(root, calibration_path).read_text(encoding="utf-8")
        )

    errors = []
    image_results = {}
    mask_scores = []
    area_pairs = []
    coverage_pairs = []
    color_pairs: dict[str, list[tuple[float, float]]] = {}
    volume_pairs = []

    try:
        import cv2

        for sample in samples:
            sample_id = str(sample.get("id") or sample.get("image") or "unknown")
            try:
                image_path = _resolve_dataset_path(root, sample["image"])
                device_id = str(sample.get("device_id") or "validation")
                result = analyzer(
                    str(image_path), device_id=device_id,
                    persist_artifacts=False, persist_state=False,
                )
                if not result:
                    raise ValueError("analysis returned no result")
                image_results[sample_id] = {
                    "result": result,
                    "device_id": device_id,
                    "image_path": str(image_path),
                }

                reference_mask_path = sample.get("validated_mask")
                reference_mask = None
                if reference_mask_path:
                    reference_mask = cv2.imread(
                        str(_resolve_dataset_path(root, reference_mask_path)), cv2.IMREAD_GRAYSCALE
                    )
                    if reference_mask is None:
                        raise ValueError(f"Could not read validated mask: {reference_mask_path}")
                    if result.get("mask") is None:
                        raise ValueError("analysis produced no canopy mask")
                    mask_scores.append(mask_metrics(reference_mask, result["mask"]))

                if sample.get("true_area_cm2") is not None:
                    predicted_area_cm2 = float(result.get("plant_area_mm2") or 0.0) / 100.0
                    area_pairs.append((float(sample["true_area_cm2"]), predicted_area_cm2))

                reference_coverage = sample.get("reference_coverage")
                if reference_coverage is None and reference_mask is not None:
                    reference_coverage = float(np.count_nonzero(reference_mask) / reference_mask.size)
                if reference_coverage is not None:
                    coverage_pairs.append((
                        float(reference_coverage),
                        float(result.get("canopy_coverage") or 0.0),
                    ))

                for metric_name, reference_value in (sample.get("color_reference") or {}).items():
                    predicted_value = (result.get("color_metrics") or {}).get(metric_name)
                    if predicted_value is not None:
                        color_pairs.setdefault(metric_name, []).append((float(reference_value), float(predicted_value)))
                    else:
                        errors.append({"sample_id": sample_id, "error": f"missing predicted color metric {metric_name!r}"})
            except Exception as exc:
                errors.append({"sample_id": sample_id, "error": str(exc)})

        for reconstruction in reconstructions:
            reconstruction_id = str(reconstruction.get("id") or "unknown")
            try:
                masks = {}
                for image in reconstruction.get("images") or []:
                    image_path = _resolve_dataset_path(root, image["image"])
                    device_id = str(image["device_id"])
                    cache_key = f"{device_id}:{image_path}"
                    if cache_key not in image_results:
                        result = analyzer(
                            str(image_path), device_id=device_id,
                            persist_artifacts=False, persist_state=False,
                        )
                        if not result:
                            raise ValueError(f"analysis returned no result for {image_path}")
                        image_results[cache_key] = {
                            "result": result,
                            "device_id": device_id,
                            "image_path": str(image_path),
                        }
                    mask = image_results[cache_key]["result"].get("mask")
                    if mask is not None:
                        masks[device_id] = mask
                if len(masks) < 2:
                    raise ValueError("at least two analyzable camera masks are required")
                reconstructed = reconstructor(masks)
                if reconstructed.get("status") != "ok" or reconstructed.get("volume_mm3") is None:
                    raise ValueError(reconstructed.get("message") or "reconstruction returned no volume")
                reference_volume = reconstruction.get("reference_volume_cm3")
                if reference_volume is not None:
                    volume_pairs.append((float(reference_volume), float(reconstructed["volume_mm3"]) / 1000.0))
            except Exception as exc:
                errors.append({"reconstruction_id": reconstruction_id, "error": str(exc)})
    finally:
        if calibration_store is not None:
            calibration_store.data = original_calibration

    aggregated_masks = _aggregate_masks(mask_scores)
    metrics = {
        "mask_iou": aggregated_masks["iou"],
        "mask_precision": aggregated_masks["precision"],
        "mask_recall": aggregated_masks["recall"],
        "area_cm2": regression_metrics(area_pairs),
        "coverage": regression_metrics(coverage_pairs),
        "color": {name: regression_metrics(pairs) for name, pairs in sorted(color_pairs.items())},
        "volume_cm3": regression_metrics(volume_pairs),
    }
    failures = _acceptance_failures(metrics, thresholds)
    evaluated_count = sum([
        len(mask_scores), len(area_pairs), len(coverage_pairs), len(volume_pairs),
        sum(len(pairs) for pairs in color_pairs.values()),
    ])
    status = "not_configured" if evaluated_count == 0 and not errors else "failed" if errors or failures else "passed"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": manifest.get("dataset_name"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "status": status,
        "sample_count": len(samples),
        "reconstruction_count": len(reconstructions),
        "evaluated_measurement_count": evaluated_count,
        "metrics": metrics,
        "acceptance_thresholds": thresholds,
        "acceptance_failures": failures,
        "errors": errors,
    }
    if report_path:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate phenotyping outputs against curated reference data.")
    parser.add_argument("--manifest", default="validation/manifest.json")
    parser.add_argument("--acceptance", default="validation/acceptance.json")
    parser.add_argument("--report", default="validation/validation_report.json")
    parser.add_argument("--data-root", help="Optional PT_DATA_ROOT used by image analysis and calibration loaders.")
    args = parser.parse_args(argv)
    if args.data_root:
        os.environ["PT_DATA_ROOT"] = str(Path(args.data_root).resolve())
    report = run_validation(args.manifest, args.acceptance, args.report)
    print(json.dumps({
        "status": report["status"],
        "report": str(Path(args.report)),
        "evaluated_measurement_count": report["evaluated_measurement_count"],
        "acceptance_failures": report["acceptance_failures"],
        "errors": report["errors"],
    }, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
