import json
import os
from datetime import datetime

import cv2
import numpy as np

from pt.core.experiments import add_event, load_plant_records, upsert_plant_record
from pt.core.utils.path_utils import get_data_root


TRAYS_PATH = os.path.join(get_data_root(), "debug", "tray_cells.json")
GERMINATION_PERSISTENCE = 3
GERMINATION_MIN_PIXELS = 80
GERMINATION_MIN_COVERAGE_DELTA = 0.004


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_id(value):
    text = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value or "").strip())
    return text.strip("_") or f"id_{int(datetime.now().timestamp() * 1000)}"


def _empty_data():
    return {"trays": []}


def load_trays():
    if not os.path.exists(TRAYS_PATH):
        return []
    try:
        with open(TRAYS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle) or _empty_data()
    except (OSError, json.JSONDecodeError):
        return []
    trays = data.get("trays", [])
    return trays if isinstance(trays, list) else []


def save_trays(trays):
    os.makedirs(os.path.dirname(TRAYS_PATH), exist_ok=True)
    with open(TRAYS_PATH, "w", encoding="utf-8") as handle:
        json.dump({"trays": trays}, handle, indent=2)


def list_trays(device_id=None):
    trays = load_trays()
    if device_id:
        return [tray for tray in trays if tray.get("device_id") == device_id]
    return trays


def upsert_tray(payload):
    tray_id = _safe_id(payload.get("tray_id") or payload.get("id") or payload.get("name"))
    now = _now()
    record = {
        "tray_id": tray_id,
        "experiment_id": str(payload.get("experiment_id") or "").strip(),
        "device_id": str(payload.get("device_id") or "").strip(),
        "name": str(payload.get("name") or tray_id).strip(),
        "variety": str(payload.get("variety") or "").strip(),
        "rows": int(max(1, min(50, int(payload.get("rows") or 1)))),
        "cols": int(max(1, min(50, int(payload.get("cols") or 1)))),
        "planted_at": str(payload.get("planted_at") or "").strip(),
        "notes": str(payload.get("notes") or "").strip(),
        "updated_at": now,
    }
    trays = load_trays()
    for index, existing in enumerate(trays):
        if existing.get("tray_id") == tray_id:
            record["created_at"] = existing.get("created_at") or now
            record["cells"] = existing.get("cells") or []
            trays[index] = {**existing, **record}
            break
    else:
        record["created_at"] = now
        record["cells"] = []
        trays.append(record)
    save_trays(trays)
    return record


def add_cell(payload):
    tray_id = str(payload.get("tray_id") or "").strip()
    if not tray_id:
        raise ValueError("tray_id is required.")
    region = payload.get("region")
    polygon = payload.get("polygon")
    if polygon:
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError("polygon must contain at least 3 points.")
        xs = [int(point[0]) for point in polygon]
        ys = [int(point[1]) for point in polygon]
        region = [min(xs), min(ys), max(xs), max(ys)]
    if not isinstance(region, list) or len(region) != 4:
        raise ValueError("region must be [x1, y1, x2, y2].")

    trays = load_trays()
    for tray in trays:
        if tray.get("tray_id") != tray_id:
            continue
        cells = tray.setdefault("cells", [])
        row = int(payload.get("row") or 0)
        col = int(payload.get("col") or 0)
        cell_id = _safe_id(payload.get("cell_id") or f"{tray_id}_r{row + 1}_c{col + 1}")
        cell = {
            "cell_id": cell_id,
            "row": row,
            "col": col,
            "name": str(payload.get("name") or cell_id).strip(),
            "region": [int(v) for v in region],
            "status": str(payload.get("status") or "empty").strip() or "empty",
            "candidate_count": 0,
            "created_at": _now(),
            "updated_at": _now(),
        }
        if polygon:
            cell["polygon"] = [[int(point[0]), int(point[1])] for point in polygon]
        cells.append(cell)
        save_trays(trays)
        return cell
    raise ValueError(f"Unknown tray_id: {tray_id}")


def delete_cell(tray_id, cell_id):
    trays = load_trays()
    deleted = 0
    for tray in trays:
        if tray.get("tray_id") != tray_id:
            continue
        cells = tray.get("cells") or []
        remaining = [cell for cell in cells if cell.get("cell_id") != cell_id]
        deleted = len(cells) - len(remaining)
        tray["cells"] = remaining
        tray["updated_at"] = _now()
        break
    if deleted:
        save_trays(trays)
    return deleted


def _plant_id_for_cell(tray, cell, plant_id=None):
    return _safe_id(plant_id or cell.get("plant_id") or f"{tray.get('tray_id')}_{cell.get('cell_id')}")


def _upsert_germinated_plant(tray, cell, device_id, timestamp, filename=None, plant_id=None):
    plant_id = _plant_id_for_cell(tray, cell, plant_id)
    upsert_plant_record({
        "plant_id": plant_id,
        "experiment_id": tray.get("experiment_id"),
        "tray_id": tray.get("tray_id"),
        "device_id": device_id,
        "cell_id": cell.get("cell_id"),
        "plant_name": cell.get("name") or plant_id,
        "planted_date": tray.get("planted_at"),
        "first_germination_at": timestamp,
        "other": f"Auto-locked from cell {cell.get('cell_id')} after {GERMINATION_PERSISTENCE} germination detections." if filename else f"Manually confirmed from cell {cell.get('cell_id')}.",
    })
    if tray.get("experiment_id"):
        add_event({
            "experiment_id": tray.get("experiment_id"),
            "type": "germination",
            "timestamp": timestamp,
            "device_id": device_id,
            "plant_id": plant_id,
            "tray_id": tray.get("tray_id"),
            "cell_id": cell.get("cell_id"),
            "notes": (
                f"Auto germination lock for {cell.get('cell_id')} from {filename}."
                if filename else f"Manual germination confirmation for {cell.get('cell_id')}."
            ),
        })
    return plant_id


def set_cell_status(tray_id, cell_id, status, plant_id=None, timestamp=None):
    trays = load_trays()
    changed = None
    timestamp = timestamp or _now()
    for tray in trays:
        if tray.get("tray_id") != tray_id:
            continue
        for cell in tray.get("cells") or []:
            if cell.get("cell_id") != cell_id:
                continue
            cell["status"] = status
            if status == "germinated":
                cell["plant_id"] = _upsert_germinated_plant(tray, cell, tray.get("device_id"), timestamp, plant_id=plant_id)
                cell.setdefault("germinated_at", timestamp)
            elif status == "empty":
                for key in ("plant_id", "germinated_at", "first_detected_at"):
                    cell.pop(key, None)
            elif plant_id is not None:
                cell["plant_id"] = plant_id
            cell["candidate_count"] = 0
            cell["updated_at"] = _now()
            changed = cell
            break
    if changed:
        save_trays(trays)
    return changed


def _mask_for_cell(mask, cell):
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = cell.get("region", [0, 0, 0, 0])
    x1, x2 = max(0, min(w, int(x1))), max(0, min(w, int(x2)))
    y1, y2 = max(0, min(h, int(y1))), max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None, [x1, y1, x2, y2], 0
    roi = mask[y1:y2, x1:x2]
    polygon = cell.get("polygon")
    if polygon and len(polygon) >= 3:
        poly_mask = np.zeros_like(mask)
        cv2.fillPoly(poly_mask, [np.array(polygon, dtype=np.int32)], 255)
        roi = cv2.bitwise_and(mask, poly_mask)[y1:y2, x1:x2]
    return roi, [x1, y1, x2, y2], max(1, (x2 - x1) * (y2 - y1))


def analyze_device_cells(device_id, mask, px_per_mm, timestamp, filename):
    if mask is None:
        return []
    trays = load_trays()
    changed = False
    metrics = []
    plant_ids = {record.get("plant_id") for record in load_plant_records()}
    for tray in trays:
        if tray.get("device_id") != device_id:
            continue
        for cell in tray.get("cells") or []:
            roi, region, cell_pixels = _mask_for_cell(mask, cell)
            if roi is None:
                continue
            pixels = int(cv2.countNonZero(roi))
            coverage = float(pixels / cell_pixels)
            area_mm2 = float(pixels / (px_per_mm ** 2)) if px_per_mm else None
            baseline = int(cell.get("baseline_pixels") if cell.get("baseline_pixels") is not None else pixels)
            baseline_coverage = float(cell.get("baseline_coverage") if cell.get("baseline_coverage") is not None else coverage)
            if cell.get("baseline_pixels") is None:
                cell["baseline_pixels"] = pixels
                cell["baseline_coverage"] = coverage
                changed = True
            delta_pixels = max(0, pixels - baseline)
            delta_coverage = max(0.0, coverage - baseline_coverage)
            threshold = max(GERMINATION_MIN_PIXELS, int(baseline * 1.75) + 20)
            candidate = (
                cell.get("status") not in ("germinated", "failed")
                and delta_pixels >= threshold
                and delta_coverage >= GERMINATION_MIN_COVERAGE_DELTA
            )
            if candidate:
                cell["candidate_count"] = int(cell.get("candidate_count") or 0) + 1
                cell.setdefault("first_detected_at", timestamp)
                if cell["candidate_count"] >= GERMINATION_PERSISTENCE:
                    plant_id = cell.get("plant_id") or _plant_id_for_cell(tray, cell)
                    cell["status"] = "germinated"
                    cell["plant_id"] = plant_id
                    cell["germinated_at"] = timestamp
                    if plant_id not in plant_ids:
                        _upsert_germinated_plant(tray, cell, device_id, timestamp, filename)
                        plant_ids.add(plant_id)
            elif cell.get("status") not in ("germinated", "failed"):
                cell["candidate_count"] = max(0, int(cell.get("candidate_count") or 0) - 1)

            cell["last_pixels"] = pixels
            cell["last_delta_pixels"] = delta_pixels
            cell["last_coverage"] = coverage
            cell["last_area_mm2"] = area_mm2
            cell["last_seen_at"] = timestamp
            cell["last_filename"] = filename
            cell["updated_at"] = _now()
            changed = True
            metrics.append({
                "tray_id": tray.get("tray_id"),
                "cell_id": cell.get("cell_id"),
                "plant_id": cell.get("plant_id"),
                "name": cell.get("name") or cell.get("cell_id"),
                "region": region,
                "polygon": cell.get("polygon"),
                "status": cell.get("status") or "empty",
                "candidate_count": int(cell.get("candidate_count") or 0),
                "canopy_pixels": pixels,
                "delta_pixels": delta_pixels,
                "coverage": coverage,
                "area_mm2": area_mm2,
            })
    if changed:
        save_trays(trays)
    return metrics
