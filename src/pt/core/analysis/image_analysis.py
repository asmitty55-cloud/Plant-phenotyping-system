import cv2
import numpy as np
import os
import json
import time
import re
from pt.core.utils.path_utils import get_data_root
from pt.core.analysis.metrics import build_metrics_snapshot
from pt.core.analysis.calibration_store import calib_store

# Common ArUco dictionaries to check
DICTIONARIES = {
    "4X4_50": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    "4X4_250": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    "6X6_250": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
    "APRILTAG_36h11": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
}

CHARUCO_WALL_TARGET = {
    "name": "back_wall_charuco_letter_8_5x11",
    "squares_x": 3,
    "squares_y": 5,
    "square_size_mm": 51.28,
    "marker_size_mm": 41.18,
    "dictionary": "4X4_50",
}

MARKER_SIZE_MM = CHARUCO_WALL_TARGET["marker_size_mm"]
CAPTURE_IMAGE_RE = re.compile(r"^capture_\d{8}_\d{6}\.(jpg|jpeg|png)$", re.IGNORECASE)


def get_charuco_target(device_id=None):
    return calib_store.get_charuco_target(CHARUCO_WALL_TARGET, device_id)


def is_capture_image(filename):
    return bool(CAPTURE_IMAGE_RE.match(filename))

def get_detector_params():
    params = cv2.aruco.DetectorParameters()
    # Balanced for 3D prints: allow some error but keep geometry strict
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.04
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    params.maxErroneousBitsInBorderRate = 0.25 # Balanced for 3D print artifacts
    params.errorCorrectionRate = 0.6
    return params

def try_detect(img, dict_name, aruco_dict, params):
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, rejected = detector.detectMarkers(img)
    return corners, ids


def get_charuco_board(device_id=None):
    target = get_charuco_target(device_id)
    return cv2.aruco.CharucoBoard(
        (target["squares_x"], target["squares_y"]),
        target["square_size_mm"],
        target["marker_size_mm"],
        DICTIONARIES[target["dictionary"]],
    )


def estimate_scale_from_charuco_corners(charuco_corners, charuco_ids, device_id=None):
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 2:
        return None

    points_by_id = {
        int(charuco_ids[i][0]): charuco_corners[i][0]
        for i in range(len(charuco_ids))
    }
    target = get_charuco_target(device_id)
    squares_x = target["squares_x"]
    square_size_mm = target["square_size_mm"]
    scales = []

    for corner_id, point in points_by_id.items():
        right_id = corner_id + 1
        below_id = corner_id + (squares_x - 1)
        if right_id in points_by_id and corner_id // (squares_x - 1) == right_id // (squares_x - 1):
            scales.append(float(np.linalg.norm(point - points_by_id[right_id]) / square_size_mm))
        if below_id in points_by_id:
            scales.append(float(np.linalg.norm(point - points_by_id[below_id]) / square_size_mm))

    return float(np.mean(scales)) if scales else None


def charuco_bbox(charuco_corners):
    if charuco_corners is None or len(charuco_corners) == 0:
        return None
    points = charuco_corners.reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(points.astype(np.float32))
    return {
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
        "center": [float(x + w / 2), float(y + h / 2)],
    }


def detect_charuco_target(gray, params, device_id=None):
    target = get_charuco_target(device_id)
    dictionary_name = target["dictionary"]
    aruco_dict = DICTIONARIES[dictionary_name]
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        return None

    board = get_charuco_board(device_id)
    charuco_corners = None
    charuco_ids = None
    charuco_scale = None

    try:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        charuco_scale = estimate_scale_from_charuco_corners(charuco_corners, charuco_ids, device_id)
    except cv2.error:
        charuco_corners = None
        charuco_ids = None

    return {
        "dictionary": dictionary_name,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "charuco_scale_px_per_mm": charuco_scale,
        "charuco_bbox": charuco_bbox(charuco_corners),
        "target": target,
    }

def calculate_canopy_metrics(frame, px_per_mm, device_id=None):
    """Segment canopy pixels and report calibrated area plus color features."""
    if px_per_mm is None or px_per_mm == 0:
        return {
            "canopy_area_mm2": 0.0,
            "canopy_pixels": 0,
            "canopy_coverage": 0.0,
            "bounding_box": None,
            "color_metrics": {},
            "mask": None,
        }

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply ignore regions if any
    if device_id:
        ignore_regions = calib_store.get_ignore_regions(device_id)
        for x1, y1, x2, y2 in ignore_regions:
            # Ensure within bounds
            y1, y2 = max(0, y1), min(frame.shape[0], y2)
            x1, x2 = max(0, x1), min(frame.shape[1], x2)
            if y2 > y1 and x2 > x1:
                # Black out ignored region in HSV (or set S=0, V=0)
                hsv[y1:y2, x1:x2] = 0
                frame[y1:y2, x1:x2] = 0

    b, g, r = cv2.split(frame.astype(np.float32))

    exg = (2 * g) - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, exg_mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lower_green = np.array([25, 20, 20])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_yellow_green = np.array([18, 35, 35])
    upper_yellow_green = np.array([105, 255, 255])
    broad_vegetation_mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)

    mask = cv2.bitwise_or(green_mask, cv2.bitwise_and(exg_mask, broad_vegetation_mask))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    canopy_pixels = int(cv2.countNonZero(mask))
    area_mm2 = float(canopy_pixels / (px_per_mm ** 2))
    coverage = float(canopy_pixels / (frame.shape[0] * frame.shape[1]))
    bbox = None
    color_metrics = {}

    if canopy_pixels > 0:
        x, y, w, h = cv2.boundingRect(mask)
        bbox = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        selected_hsv = hsv[mask > 0]
        selected_bgr = frame[mask > 0].astype(np.float32)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        selected_lab = lab[mask > 0].astype(np.float32)

        selected_b = selected_bgr[:, 0]
        selected_g = selected_bgr[:, 1]
        selected_r = selected_bgr[:, 2]
        green_index = np.mean((2 * selected_g - selected_r - selected_b) / 255.0)
        yellow_ratio = np.mean(
            (selected_hsv[:, 0] >= 18)
            & (selected_hsv[:, 0] <= 42)
            & (selected_hsv[:, 1] >= 40)
        )
        dark_ratio = np.mean(selected_hsv[:, 2] < 55)

        color_metrics = {
            "mean_hue": float(np.mean(selected_hsv[:, 0])),
            "mean_saturation": float(np.mean(selected_hsv[:, 1])),
            "mean_value": float(np.mean(selected_hsv[:, 2])),
            "mean_lab_a": float(np.mean(selected_lab[:, 1])),
            "mean_lab_b": float(np.mean(selected_lab[:, 2])),
            "green_index": float(green_index),
            "chlorosis_ratio": float(yellow_ratio),
            "dark_tissue_ratio": float(dark_ratio),
        }

    return {
        "canopy_area_mm2": area_mm2,
        "canopy_pixels": canopy_pixels,
        "canopy_coverage": coverage,
        "bounding_box": bbox,
        "color_metrics": color_metrics,
        "mask": mask,
    }


def calculate_plant_area(frame, px_per_mm):
    canopy = calculate_canopy_metrics(frame, px_per_mm)
    return canopy["canopy_area_mm2"], canopy["mask"]


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
    baseline = {
        "status": "ready",
        "samples": len(samples),
        "canopy_area_mm2": median_metric(samples, ["area"]),
        "green_index": median_metric(samples, ["color_metrics", "green_index"]),
        "chlorosis_ratio": median_metric(samples, ["color_metrics", "chlorosis_ratio"]),
        "mean_hue": median_metric(samples, ["color_metrics", "mean_hue"]),
        "mean_saturation": median_metric(samples, ["color_metrics", "mean_saturation"]),
        "mean_value": median_metric(samples, ["color_metrics", "mean_value"]),
    }
    return baseline


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


def make_json_safe(value):
    """Convert OpenCV/NumPy analysis output into JSON-friendly Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def serialize_analysis_results(results):
    serializable = make_json_safe({k: v for k, v in results.items() if k != "mask"})
    mask_path = results.get("mask_path")
    if mask_path:
        serializable["mask_path"] = mask_path
    return serializable

def analyze_image(image_path, device_id=None):
    """
    Detects ArUco markers with aggressive multi-channel fallback for 3D prints & grow lights.
    """
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return None

        # Pre-processing variants to handle reflections/purple light
        processing_variants = []

        if len(frame.shape) == 3:
            b, g, r = cv2.split(frame)
            # 1. Pure Green (Best for Blurple)
            processing_variants.append(("Green", g))
            # 2. Pure Blue (Sometimes better if Green saturates)
            processing_variants.append(("Blue", b))
            # 3. Standard Grayscale
            processing_variants.append(("Gray", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        else:
            processing_variants.append(("Source", frame))

        params = get_detector_params()
        best_corners = None
        best_ids = None
        detected_dict = None
        used_method = None
        charuco_detection = None

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

        # Create a persistent debug frame for drawing
        debug_frame = frame.copy()

        for method_name, img in processing_variants:
            # 1. Enhance Contrast
            enhanced = clahe.apply(img)

            # 2. Smooth 3D Print Noise (striations)
            # A slight blur helps remove layer lines that look like false "bits"
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

            # 3. Sharpen Edges
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(blurred, -1, kernel)

            if charuco_detection is None:
                charuco_detection = detect_charuco_target(sharpened, params, device_id)

            for dict_name, aruco_dict in DICTIONARIES.items():
                detector = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners, ids, rejected = detector.detectMarkers(sharpened)

                # Draw rejected candidates for debugging if nothing found yet
                if (ids is None or len(ids) == 0) and rejected is not None:
                    cv2.aruco.drawDetectedMarkers(debug_frame, rejected, borderColor=(0, 0, 255))

                # Validation: ArUco markers must be roughly square and have a reasonable ID
                valid_corners = []
                valid_ids = []

                if ids is not None:
                    for i, marker_corners in enumerate(corners):
                        c = marker_corners[0]
                        # Check "squareness" (ratio of side lengths)
                        s1 = np.linalg.norm(c[0] - c[1])
                        s2 = np.linalg.norm(c[1] - c[2])
                        if s1 > 0 and 0.8 < (s1 / s2) < 1.2:
                            valid_corners.append(marker_corners)
                            valid_ids.append(ids[i])

                if valid_ids:
                    best_corners = valid_corners
                    best_ids = np.array(valid_ids)
                    detected_dict = dict_name
                    used_method = method_name
                    break
            if best_ids is not None: break

        # Results packaging
        results = {
            "markers_found": 0,
            "markers": [],
            "scale_px_per_mm": None,
            "dictionary": detected_dict,
            "method": used_method,
            "calibration_target": get_charuco_target(device_id),
        }

        # Setup debug frame
        debug_frame = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if best_ids is not None:
            results["markers_found"] = len(best_ids)
            scales = []
            for i in range(len(best_ids)):
                marker_id = int(best_ids[i][0])
                marker_corners = best_corners[i][0]
                # Calculate side lengths
                s1 = float(np.linalg.norm(marker_corners[0] - marker_corners[1]))
                s2 = float(np.linalg.norm(marker_corners[1] - marker_corners[2]))
                s3 = float(np.linalg.norm(marker_corners[2] - marker_corners[3]))
                s4 = float(np.linalg.norm(marker_corners[3] - marker_corners[0]))

                avg_side_px = (s1 + s2 + s3 + s4) / 4.0

                # Check for marker size override
                marker_size_mm = calib_store.get_marker_size(marker_id, device_id) or MARKER_SIZE_MM

                px_per_mm = float(avg_side_px / marker_size_mm)
                scales.append(px_per_mm)

                center = np.mean(marker_corners, axis=0)
                results["markers"].append({
                    "id": marker_id,
                    "center": [float(center[0]), float(center[1])],
                    "corners": marker_corners.astype(float).tolist(),
                    "px_per_mm": px_per_mm,
                    "size_mm": marker_size_mm
                })
            marker_scale = float(np.mean(scales))
            results["scale_px_per_mm"] = marker_scale
            results["scale_source"] = "charuco_marker_size"

            if charuco_detection and charuco_detection["charuco_scale_px_per_mm"]:
                results["scale_px_per_mm"] = charuco_detection["charuco_scale_px_per_mm"]
                results["scale_source"] = "charuco_corner_spacing"

            if charuco_detection:
                results["charuco_corners_found"] = (
                    0 if charuco_detection["charuco_ids"] is None
                    else int(len(charuco_detection["charuco_ids"]))
                )
                results["charuco_target"] = charuco_detection.get("target")
                results["charuco_bbox"] = charuco_detection.get("charuco_bbox")

            canopy = calculate_canopy_metrics(frame, results["scale_px_per_mm"], device_id)
            plant_area = canopy["canopy_area_mm2"]
            plant_mask = canopy["mask"]
            results.update({
                "plant_area_mm2": plant_area,
                "canopy_area_mm2": plant_area,
                "canopy_pixels": canopy["canopy_pixels"],
                "canopy_coverage": canopy["canopy_coverage"],
                "canopy_bounding_box": canopy["bounding_box"],
                "color_metrics": canopy["color_metrics"],
                "mask": plant_mask,
            })

            # Highlight plant in debug view (subtle green tint)
            if plant_mask is not None:
                plant_overlay = np.zeros_like(debug_frame)
                plant_overlay[plant_mask > 0] = [0, 255, 0]
                cv2.addWeighted(debug_frame, 1.0, plant_overlay, 0.3, 0, debug_frame)
                if canopy["bounding_box"]:
                    box = canopy["bounding_box"]
                    cv2.rectangle(
                        debug_frame,
                        (box["x"], box["y"]),
                        (box["x"] + box["width"], box["y"] + box["height"]),
                        (0, 200, 255),
                        2,
                    )

            cv2.aruco.drawDetectedMarkers(debug_frame, best_corners, best_ids)
            if charuco_detection and charuco_detection["charuco_corners"] is not None:
                cv2.aruco.drawDetectedCornersCharuco(
                    debug_frame,
                    charuco_detection["charuco_corners"],
                    charuco_detection["charuco_ids"],
                )
            cv2.putText(debug_frame, f"ChArUco wall: {detected_dict} via {used_method}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Canopy: {plant_area:.1f} mm^2", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Scale: {results['scale_px_per_mm']:.3f} px/mm ({results['scale_source']})", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print(f"[ANALYSIS] Failed to find markers in {image_path} after trying Green, Blue, and Gray channels.")

        # Save debug
        debug_dir = os.path.join(get_data_root(), "analysis_debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, os.path.basename(image_path))
        cv2.imwrite(debug_path, debug_frame)
        results["debug_image"] = debug_path

        # Save mask for volumetric reconstruction
        if results.get("mask") is not None:
            mask_path = image_path.rsplit(".", 1)[0] + "_mask.png"
            cv2.imwrite(mask_path, results["mask"])
            results["mask_path"] = mask_path

        return results
    except Exception as e:
        print(f"[ANALYSIS] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_latest_captures(captures_dir):
    stats_file = os.path.join(get_data_root(), "plant_stats.json")
    all_stats = {}
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f: all_stats = json.load(f)
        except: pass

    # Track growth rates to find the winner
    fastest_rate = -1.0
    winner_id = None

    for device_id in os.listdir(captures_dir):
        device_path = os.path.join(captures_dir, device_id)
        if not os.path.isdir(device_path): continue
        files = sorted([f for f in os.listdir(device_path) if is_capture_image(f)])
        if not files: continue

        results = analyze_image(os.path.join(device_path, files[-1]), device_id=device_id)
        if results:
            if device_id not in all_stats: all_stats[device_id] = {"history": []}

            # Reset fastest tag initially
            all_stats[device_id]["is_fastest"] = False

            history = all_stats[device_id].get("history", [])
            current_area = float(results.get("plant_area_mm2", 0))
            baseline = build_color_baseline(history, all_stats[device_id].get("baseline"))
            deficiency = evaluate_nutrient_flags(results.get("color_metrics"), baseline)

            # Calculate Growth Rate (mm2 / hour)
            growth_rate = 0.0
            if len(history) >= 1:
                last_entry = history[-1]
                try:
                    t1 = time.mktime(time.strptime(last_entry["timestamp"], "%Y-%m-%d %H:%M:%S"))
                    t2 = time.time()
                    hours = (t2 - t1) / 3600.0
                    if hours > 0.01:
                        growth_rate = (current_area - (last_entry.get("area") or 0)) / hours
                except: pass

            all_stats[device_id].update({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "filename": files[-1],
                "data": serialize_analysis_results(results),
                "growth_rate_mm2_hr": growth_rate,
                "baseline": baseline,
                "nutrient_deficiency": deficiency,
                "metrics": build_metrics_snapshot(
                    plant_id=device_id,
                    device_id=device_id,
                    filename=files[-1],
                    analysis={**serialize_analysis_results(results), "nutrient_deficiency": deficiency},
                    growth_rate_mm2_hr=growth_rate,
                )
            })

            if not history or history[-1]["filename"] != files[-1]:
                history.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": files[-1],
                    "scale": float(results["scale_px_per_mm"]) if results["scale_px_per_mm"] else None,
                    "area": current_area,
                    "canopy_coverage": results.get("canopy_coverage"),
                    "color_metrics": results.get("color_metrics"),
                    "nutrient_deficiency": deficiency,
                })
                all_stats[device_id]["history"] = history[-100:]

            if growth_rate > fastest_rate:
                fastest_rate = growth_rate
                winner_id = device_id

    # Mark the winner
    if winner_id and fastest_rate > 0:
        all_stats[winner_id]["is_fastest"] = True

    with open(stats_file, 'w') as f: json.dump(make_json_safe(all_stats), f, indent=4)
    return all_stats
