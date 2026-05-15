import cv2
import numpy as np
import os
import json
import time
from pt.core.utils.path_utils import get_data_root
from pt.core.analysis.metrics import build_metrics_snapshot

# Common ArUco dictionaries to check
DICTIONARIES = {
    "4X4_50": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    "4X4_250": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
    "6X6_250": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
    "APRILTAG_36h11": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
}

MARKER_SIZE_MM = 60.0

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

def calculate_plant_area(frame, px_per_mm):
    """
    Isolates green pixels across a wide variety of shades (light to dark)
    and calculates area in mm^2.
    """
    if px_per_mm is None or px_per_mm == 0:
        return 0.0, None

    # Convert to HSV for robust color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Expanded range:
    # Hue: 25 (yellow-green) to 95 (deep forest/blue-green)
    # Saturation/Value: Lowered to 20 to catch dark shadows and pale leaves
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up soil noise and fill leaf gaps
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Remove small soil dots
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fill small leaf holes

    green_pixels = cv2.countNonZero(mask)
    area_mm2 = float(green_pixels / (px_per_mm ** 2))
    return area_mm2, mask

def analyze_image(image_path):
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
            "method": used_method
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
                px_per_mm = float(avg_side_px / MARKER_SIZE_MM)
                scales.append(px_per_mm)

                center = np.mean(marker_corners, axis=0)
                results["markers"].append({
                    "id": marker_id,
                    "center": [float(center[0]), float(center[1])],
                    "px_per_mm": px_per_mm
                })
            results["scale_px_per_mm"] = float(np.mean(scales))

            # --- NEW: Plant Growth Analysis ---
            plant_area, plant_mask = calculate_plant_area(frame, results["scale_px_per_mm"])
            results["plant_area_mm2"] = plant_area

            # Highlight plant in debug view (subtle green tint)
            plant_overlay = np.zeros_like(debug_frame)
            plant_overlay[plant_mask > 0] = [0, 255, 0]
            cv2.addWeighted(debug_frame, 1.0, plant_overlay, 0.3, 0, debug_frame)

            cv2.aruco.drawDetectedMarkers(debug_frame, best_corners, best_ids)
            cv2.putText(debug_frame, f"{detected_dict} via {used_method}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Area: {plant_area:.1f} mm^2", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print(f"[ANALYSIS] Failed to find markers in {image_path} after trying Green, Blue, and Gray channels.")

        # Save debug
        debug_dir = os.path.join(get_data_root(), "analysis_debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, os.path.basename(image_path))
        cv2.imwrite(debug_path, debug_frame)
        results["debug_image"] = debug_path
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
        files = sorted([f for f in os.listdir(device_path) if f.endswith(".jpg")])
        if not files: continue

        results = analyze_image(os.path.join(device_path, files[-1]))
        if results:
            if device_id not in all_stats: all_stats[device_id] = {"history": []}

            # Reset fastest tag initially
            all_stats[device_id]["is_fastest"] = False

            history = all_stats[device_id].get("history", [])
            current_area = float(results.get("plant_area_mm2", 0))

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
                "data": results,
                "growth_rate_mm2_hr": growth_rate,
                "metrics": build_metrics_snapshot(
                    plant_id=device_id,
                    device_id=device_id,
                    filename=files[-1],
                    analysis=results,
                    growth_rate_mm2_hr=growth_rate,
                )
            })

            if not history or history[-1]["filename"] != files[-1]:
                history.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename": files[-1],
                    "scale": float(results["scale_px_per_mm"]) if results["scale_px_per_mm"] else None,
                    "area": current_area
                })
                all_stats[device_id]["history"] = history[-100:]

            if growth_rate > fastest_rate:
                fastest_rate = growth_rate
                winner_id = device_id

    # Mark the winner
    if winner_id and fastest_rate > 0:
        all_stats[winner_id]["is_fastest"] = True

    with open(stats_file, 'w') as f: json.dump(all_stats, f, indent=4)
    return all_stats
