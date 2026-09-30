import json

import cv2
import numpy as np

from pt.validation import mask_metrics, regression_metrics, run_validation


def test_mask_metrics_compute_iou_precision_and_recall():
    reference = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    predicted = np.array([[255, 0], [255, 0]], dtype=np.uint8)

    result = mask_metrics(reference, predicted)

    assert result["iou"] == 1 / 3
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5


def test_regression_metrics_compute_mae_rmse_and_percent_error():
    result = regression_metrics([(2.0, 1.0), (4.0, 6.0)])

    assert result["count"] == 2
    assert result["mae"] == 1.5
    assert np.isclose(result["rmse"], np.sqrt(2.5))
    assert result["mean_absolute_percent_error"] == 50.0


def test_empty_reference_manifest_is_not_configured_not_a_science_pass(tmp_path):
    manifest = tmp_path / "manifest.json"
    report_path = tmp_path / "validation_report.json"
    manifest.write_text(json.dumps({
        "schema_version": "plant_validation.v1",
        "dataset_name": "empty-fixture",
        "samples": [],
        "reconstructions": [],
    }), encoding="utf-8")

    report = run_validation(manifest, report_path=report_path)

    assert report["status"] == "not_configured"
    assert report["evaluated_measurement_count"] == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "not_configured"


def test_runner_scores_image_and_optional_reconstruction_references(tmp_path):
    source_image = tmp_path / "source.jpg"
    source_image.write_bytes(b"placeholder input for injected analyzer")
    reference_mask = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    mask_path = tmp_path / "reference-mask.png"
    assert cv2.imwrite(str(mask_path), reference_mask)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "plant_validation.v1",
        "dataset_name": "synthetic-metric-fixture",
        "samples": [{
            "id": "sample-1",
            "image": "source.jpg",
            "validated_mask": "reference-mask.png",
            "device_id": "camera-a",
            "true_area_cm2": 2.0,
            "reference_coverage": 0.25,
            "color_reference": {"green_index": 0.5},
        }],
        "reconstructions": [{
            "id": "volume-1",
            "images": [
                {"device_id": "camera-a", "image": "source.jpg"},
                {"device_id": "camera-b", "image": "source.jpg"},
            ],
            "reference_volume_cm3": 1.0,
        }],
    }), encoding="utf-8")

    def analyzer(image_path, device_id=None, persist_artifacts=True, persist_state=True):
        return {
            "mask": reference_mask.copy(),
            "plant_area_mm2": 200.0,
            "canopy_coverage": 0.25,
            "color_metrics": {"green_index": 0.5},
        }

    def reconstructor(masks):
        assert set(masks) == {"camera-a", "camera-b"}
        return {"status": "ok", "volume_mm3": 1000.0}

    acceptance = {
        "mask_iou_min": 0.99,
        "mask_precision_min": 0.99,
        "mask_recall_min": 0.99,
        "area_mean_absolute_percent_error_max": 1.0,
        "coverage_mae_max": 0.01,
        "color_mean_absolute_percent_error_max": 1.0,
        "volume_mean_absolute_percent_error_max": 1.0,
    }
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")

    report = run_validation(
        manifest,
        acceptance_path=acceptance_path,
        analyzer=analyzer,
        reconstructor=reconstructor,
    )

    assert report["status"] == "passed"
    assert report["metrics"]["mask_iou"]["mean"] == 1.0
    assert report["metrics"]["area_cm2"]["mae"] == 0.0
    assert report["metrics"]["color"]["green_index"]["rmse"] == 0.0
    assert report["metrics"]["volume_cm3"]["mean_absolute_percent_error"] == 0.0
    assert not report["acceptance_failures"]
