import json
import os
from datetime import datetime

import yaml

from pt.core.utils.path_utils import get_data_root


CONFIG_PATH = os.path.join(get_data_root(), "configs", "experiments.yaml")
EVENTS_PATH = os.path.join(get_data_root(), "debug", "experiment_events.json")
PLANTS_PATH = os.path.join(get_data_root(), "debug", "plant_records.json")


def load_experiments():
    if not os.path.exists(CONFIG_PATH):
        return []
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return []
    return [item for item in data.get("experiments", []) if isinstance(item, dict)]


def save_experiments(experiments):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump({"experiments": experiments}, handle, sort_keys=False)


def upsert_experiment(payload):
    experiment_id = str(payload.get("id") or "").strip()
    if not experiment_id:
        raise ValueError("Experiment ID is required.")
    experiments = load_experiments()
    record = {
        "id": experiment_id,
        "name": str(payload.get("name") or experiment_id).strip(),
        "status": str(payload.get("status") or "active").strip(),
        "hypothesis": str(payload.get("hypothesis") or "").strip(),
        "devices": [str(value) for value in payload.get("devices") or []],
        "plants": [str(value) for value in payload.get("plants") or []],
        "started_at": str(payload.get("started_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "notes": str(payload.get("notes") or "").strip(),
    }
    for index, existing in enumerate(experiments):
        if existing.get("id") == experiment_id:
            record["started_at"] = existing.get("started_at") or record["started_at"]
            experiments[index] = {**existing, **record}
            break
    else:
        experiments.append(record)
    save_experiments(experiments)
    return record


def load_events():
    if not os.path.exists(EVENTS_PATH):
        return []
    try:
        with open(EVENTS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def load_plant_records():
    if not os.path.exists(PLANTS_PATH):
        return []
    try:
        with open(PLANTS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_plant_records(records):
    os.makedirs(os.path.dirname(PLANTS_PATH), exist_ok=True)
    with open(PLANTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(records[-5000:], handle, indent=2)


def upsert_plant_record(payload):
    plant_id = str(payload.get("plant_id") or payload.get("id") or "").strip()
    if not plant_id:
        plant_id = f"plant_{int(datetime.now().timestamp() * 1000)}"
    record = {
        "plant_id": plant_id,
        "experiment_id": str(payload.get("experiment_id") or "").strip(),
        "tray_id": str(payload.get("tray_id") or "").strip(),
        "device_id": str(payload.get("device_id") or "").strip(),
        "segment_id": str(payload.get("segment_id") or "").strip(),
        "cell_id": str(payload.get("cell_id") or "").strip(),
        "plant_name": str(payload.get("plant_name") or payload.get("plant") or "").strip(),
        "variety": str(payload.get("variety") or "").strip(),
        "parent_mother": str(payload.get("parent_mother") or "").strip(),
        "parent_father": str(payload.get("parent_father") or "").strip(),
        "planted_date": str(payload.get("planted_date") or "").strip(),
        "first_germination_at": str(payload.get("first_germination_at") or "").strip(),
        "irrigation": str(payload.get("irrigation") or "").strip(),
        "fertilizer": str(payload.get("fertilizer") or "").strip(),
        "harvest": str(payload.get("harvest") or "").strip(),
        "other": str(payload.get("other") or payload.get("notes") or "").strip(),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    records = load_plant_records()
    for index, existing in enumerate(records):
        if existing.get("plant_id") == plant_id:
            record["created_at"] = existing.get("created_at") or record["updated_at"]
            records[index] = {**existing, **record}
            break
    else:
        record["created_at"] = record["updated_at"]
        records.append(record)
    save_plant_records(records)
    return record


def add_event(payload):
    experiment_id = str(payload.get("experiment_id") or "").strip()
    event_type = str(payload.get("type") or "observation").strip()
    if not experiment_id:
        raise ValueError("Choose an experiment before logging an event.")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    event = {
        "id": f"event_{int(datetime.now().timestamp() * 1000)}",
        "experiment_id": experiment_id,
        "timestamp": str(payload.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "type": event_type,
        "device_id": str(payload.get("device_id") or "").strip(),
        "plant_id": str(payload.get("plant_id") or "").strip(),
        "tray_id": str(payload.get("tray_id") or "").strip(),
        "segment_id": str(payload.get("segment_id") or "").strip(),
        "cell_id": str(payload.get("cell_id") or "").strip(),
        "value": payload.get("value"),
        "fresh_weight_g": data.get("fresh_weight_g", payload.get("fresh_weight_g")),
        "dry_weight_g": data.get("dry_weight_g", payload.get("dry_weight_g")),
        "cut_height_mm": data.get("cut_height_mm", payload.get("cut_height_mm")),
        "area_removed_mm2": data.get("area_removed_mm2", payload.get("area_removed_mm2")),
        "data": data,
        "notes": str(payload.get("notes") or "").strip(),
    }
    events = load_events()
    events.append(event)
    os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)
    with open(EVENTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(events[-5000:], handle, indent=2)
    return event
