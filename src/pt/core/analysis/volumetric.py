import cv2
import numpy as np
import os
import json
from pt.core.analysis.calibration_store import calib_store
from pt.core.analysis.image_analysis import get_charuco_board, get_detector_params, DICTIONARIES, get_charuco_target


def _camera_intrinsics(device_id, gray):
    stored_params = calib_store.get_camera_params(device_id)
    if stored_params and "K" in stored_params:
        return np.array(stored_params["K"]), np.array(stored_params["dist"])
    h, w = gray.shape
    f = w
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32), np.zeros(5, dtype=np.float32)


def _landmark_world_corners(landmark):
    size = float(landmark["size_mm"])
    half = size / 2.0
    local = np.array([
        [-half, half, 0],
        [half, half, 0],
        [half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)
    rx, ry, rz = np.radians(np.asarray(landmark.get("rotation_deg", [0, 0, 0]), dtype=np.float64))
    rotation_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    rotation_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    rotation_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    rotation = rotation_z @ rotation_y @ rotation_x
    position = np.asarray(landmark.get("position_mm", [0, 0, 0]), dtype=np.float64)
    return (local @ rotation.T + position).astype(np.float32)


def estimate_pose_from_landmarks(gray, device_id, params):
    object_points = []
    image_points = []
    landmarks_used = []
    by_dictionary = {}
    for landmark in calib_store.list_landmarks(enabled_only=True):
        by_dictionary.setdefault(landmark["dictionary"], {})[int(landmark["id"])] = landmark
    for dictionary_name, landmarks in by_dictionary.items():
        aruco_dict = DICTIONARIES.get(dictionary_name)
        if aruco_dict is None:
            continue
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if ids is None:
            continue
        for index, raw_id in enumerate(ids.reshape(-1)):
            landmark = landmarks.get(int(raw_id))
            if not landmark:
                continue
            object_points.extend(_landmark_world_corners(landmark))
            image_points.extend(corners[index][0])
            landmarks_used.append(f"{dictionary_name}:{int(raw_id)}")
    if len(object_points) < 4:
        return None
    K, dist = _camera_intrinsics(device_id, gray)
    valid, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=5.0,
    )
    if not valid:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return {"K": K, "dist": dist, "R": R, "T": tvec, "landmarks": landmarks_used, "inliers": 0 if inliers is None else len(inliers)}

def calibrate_extrinsics(device_images):
    """
    device_images: dict of device_id -> image_path
    Uses the same board to find relative poses.
    """
    params = get_detector_params()

    # We'll use the board's coordinate system as the world origin
    # or one camera as origin.

    results = {}

    for device_id, img_path in device_images.items():
        target = get_charuco_target(device_id)
        board = get_charuco_board(device_id)
        aruco_dict = DICTIONARIES[target["dictionary"]]
        img = cv2.imread(img_path)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect markers
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        if marker_ids is not None and len(marker_ids) > 0:
            # Interpolate corners
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )

            if charuco_ids is not None and len(charuco_ids) >= 4:
                # Need camera intrinsics. If not calibrated, we might have to estimate or use defaults.
                # For now, let's assume we have them or use a heuristic.
                K, dist = _camera_intrinsics(device_id, gray)

                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, board, K, dist, None, None
                )

                if valid:
                    # Convert rvec to rotation matrix
                    R, _ = cv2.Rodrigues(rvec)
                    calib_store.set_camera_params(device_id, K, dist, R, tvec)
                    results[device_id] = {"R": R.tolist(), "T": tvec.tolist(), "source": "charuco"}
                    continue

        landmark_pose = estimate_pose_from_landmarks(gray, device_id, params)
        if landmark_pose:
            calib_store.set_camera_params(
                device_id,
                landmark_pose["K"],
                landmark_pose["dist"],
                landmark_pose["R"],
                landmark_pose["T"],
            )
            results[device_id] = {
                "R": landmark_pose["R"].tolist(),
                "T": landmark_pose["T"].tolist(),
                "source": "mapped_landmarks",
                "landmarks": landmark_pose["landmarks"],
                "inliers": landmark_pose["inliers"],
            }

    return results

def reconstruct_visual_hull(device_masks, voxel_size=(100, 100, 100), world_bounds=None):
    """
    device_masks: dict of device_id -> mask (binary image)
    """
    # 1. Define voxel grid
    # For now, a 1m^3 box centered at origin (board)
    if world_bounds is None:
        # [xmin, ymin, zmin, xmax, ymax, zmax] in mm
        world_bounds = [-200, -200, 0, 200, 200, 400]

    x = np.linspace(world_bounds[0], world_bounds[3], voxel_size[0])
    y = np.linspace(world_bounds[1], world_bounds[4], voxel_size[1])
    z = np.linspace(world_bounds[2], world_bounds[5], voxel_size[2])

    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    voxels = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3) # (N, 3)

    # Occupancy grid
    occupancy = np.ones(len(voxels), dtype=bool)
    cameras_used = []
    skipped_devices = []

    for device_id, mask in device_masks.items():
        params = calib_store.get_camera_params(device_id)
        if not params or not params.get("R") or not params.get("T") or not params.get("K"):
            skipped_devices.append(device_id)
            continue
        cameras_used.append(device_id)

        K = np.array(params["K"])
        R = np.array(params["R"])
        T = np.array(params["T"])
        dist = np.array(params["dist"])

        # Project voxels to image plane
        # world to camera: P_c = R * P_w + T
        # Since the board IS the world origin in our estimatePoseCharucoBoard call
        img_pts, _ = cv2.projectPoints(voxels, R, T, K, dist)
        img_pts = img_pts.reshape(-1, 2).astype(int)

        # Check if projected points are in mask
        h, w = mask.shape
        valid_idx = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) & \
                    (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)

        # For points inside image, check mask
        in_mask = np.zeros(len(voxels), dtype=bool)
        if np.any(valid_idx):
            pts_in_img = img_pts[valid_idx]
            in_mask[valid_idx] = mask[pts_in_img[:, 1], pts_in_img[:, 0]] > 0

        # Voxel must be in ALL silhouettes
        occupancy &= in_mask

    if not cameras_used:
        return {
            "status": "not_ready",
            "message": "No calibrated camera poses are available. Run 3D calibration after ChArUco corners are visible.",
            "volume_mm3": None,
            "occupied_voxels": 0,
            "grid_shape": list(voxel_size),
            "cameras_used": [],
            "skipped_devices": skipped_devices,
        }

    # Final volume
    occupied_count = np.sum(occupancy)
    voxel_vol = (x[1]-x[0]) * (y[1]-y[0]) * (z[1]-z[0])
    total_volume_mm3 = occupied_count * voxel_vol
    occupied_points = voxels[occupancy]
    if len(occupied_points) > 2500:
        step = max(1, len(occupied_points) // 2500)
        occupied_points = occupied_points[::step][:2500]

    return {
        "status": "ok",
        "volume_mm3": float(total_volume_mm3),
        "occupied_voxels": int(occupied_count),
        "grid_shape": list(voxel_size),
        "cameras_used": cameras_used,
        "skipped_devices": skipped_devices,
        "world_bounds": world_bounds,
        "preview_points": occupied_points.astype(float).tolist(),
    }
