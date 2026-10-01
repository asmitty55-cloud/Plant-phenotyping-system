import json
import hashlib
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from pt.core.utils.path_utils import get_data_root


DB_FILENAME = "plant_observatory.sqlite3"
MAX_DASHBOARD_POINTS = 10000
_init_lock = threading.Lock()
_initialized = False
_initialized_path = None


def db_path():
    return os.path.join(get_data_root(), DB_FILENAME)


@contextmanager
def connect():
    database_path = db_path()
    ensure_db(database_path)
    conn = sqlite3.connect(database_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_db(database_path=None):
    global _initialized, _initialized_path
    database_path = database_path or db_path()
    if _initialized and _initialized_path == database_path:
        return
    with _init_lock:
        if _initialized and _initialized_path == database_path:
            return
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
        conn = sqlite3.connect(database_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    area REAL,
                    growth_rate_mm2_hr REAL,
                    scale REAL,
                    detected_scale REAL,
                    scale_rejected INTEGER DEFAULT 0,
                    canopy_coverage REAL,
                    green_index REAL,
                    color_metrics_json TEXT,
                    color_correction_json TEXT,
                    segments_json TEXT,
                    ignored_segments_json TEXT,
                    nutrient_json TEXT,
                    movement_json TEXT,
                    capture_set_id TEXT,
                    analysis_run_id TEXT,
                    previous_capture_set_id TEXT,
                    ignored INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(device_id, filename)
                )
                """
            )
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(metric_history)").fetchall()
            }
            if "color_correction_json" not in existing_columns:
                conn.execute("ALTER TABLE metric_history ADD COLUMN color_correction_json TEXT")
            if "movement_json" not in existing_columns:
                conn.execute("ALTER TABLE metric_history ADD COLUMN movement_json TEXT")
            if "capture_set_id" not in existing_columns:
                conn.execute("ALTER TABLE metric_history ADD COLUMN capture_set_id TEXT")
            if "analysis_run_id" not in existing_columns:
                conn.execute("ALTER TABLE metric_history ADD COLUMN analysis_run_id TEXT")
            if "previous_capture_set_id" not in existing_columns:
                conn.execute("ALTER TABLE metric_history ADD COLUMN previous_capture_set_id TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_rollups (
                    device_id TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    period TEXT NOT NULL,
                    point_count INTEGER NOT NULL,
                    avg_area REAL,
                    max_area REAL,
                    min_area REAL,
                    avg_growth_rate REAL,
                    avg_green_index REAL,
                    PRIMARY KEY(device_id, period, bucket)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backfill_status (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    running INTEGER DEFAULT 0,
                    started_at TEXT,
                    finished_at TEXT,
                    current_device TEXT,
                    processed INTEGER DEFAULT 0,
                    total INTEGER DEFAULT 0,
                    message TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plant_metric_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plant_id TEXT NOT NULL,
                    tray_id TEXT,
                    cell_id TEXT,
                    device_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    area REAL,
                    coverage REAL,
                    canopy_pixels INTEGER,
                    delta_pixels INTEGER,
                    status TEXT,
                    capture_set_id TEXT,
                    analysis_run_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(plant_id, filename)
                )
                """
            )
            plant_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(plant_metric_history)").fetchall()
            }
            if "capture_set_id" not in plant_columns:
                conn.execute("ALTER TABLE plant_metric_history ADD COLUMN capture_set_id TEXT")
            if "analysis_run_id" not in plant_columns:
                conn.execute("ALTER TABLE plant_metric_history ADD COLUMN analysis_run_id TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS capture_sets (
                    capture_set_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    captured_at_utc TEXT,
                    created_at_utc TEXT NOT NULL,
                    source TEXT NOT NULL,
                    notes TEXT,
                    sync_quality_ms REAL,
                    device_count INTEGER,
                    successful_devices_json TEXT NOT NULL DEFAULT '[]',
                    failed_devices_json TEXT NOT NULL DEFAULT '[]',
                    capture_duration_ms REAL,
                    image_count INTEGER NOT NULL DEFAULT 0,
                    qc_status TEXT NOT NULL DEFAULT 'not_assessed',
                    reconstruction_status TEXT NOT NULL DEFAULT 'not_run'
                )
                """
            )
            capture_set_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(capture_sets)").fetchall()
            }
            capture_set_migrations = {
                "sync_quality_ms": "REAL",
                "device_count": "INTEGER",
                "successful_devices_json": "TEXT NOT NULL DEFAULT '[]'",
                "failed_devices_json": "TEXT NOT NULL DEFAULT '[]'",
                "capture_duration_ms": "REAL",
                "image_count": "INTEGER NOT NULL DEFAULT 0",
                "qc_status": "TEXT NOT NULL DEFAULT 'not_assessed'",
                "reconstruction_status": "TEXT NOT NULL DEFAULT 'not_run'",
            }
            for column, definition in capture_set_migrations.items():
                if column not in capture_set_columns:
                    conn.execute(f"ALTER TABLE capture_sets ADD COLUMN {column} {definition}")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS capture_images (
                    device_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    capture_set_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    captured_at_utc TEXT,
                    image_sha256 TEXT,
                    camera_metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY(device_id, filename)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calibration_snapshots (
                    calibration_snapshot_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    payload_hash TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    analysis_run_id TEXT PRIMARY KEY,
                    capture_set_id TEXT NOT NULL,
                    calibration_snapshot_id TEXT NOT NULL,
                    pipeline_version TEXT NOT NULL,
                    parameter_hash TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    enabled_modules_json TEXT NOT NULL DEFAULT '[]',
                    module_versions_json TEXT NOT NULL DEFAULT '{}',
                    created_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'complete',
                    UNIQUE(capture_set_id, calibration_snapshot_id, pipeline_version, parameter_hash)
                )
                """
            )
            run_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(analysis_runs)").fetchall()
            }
            if "enabled_modules_json" not in run_columns:
                conn.execute("ALTER TABLE analysis_runs ADD COLUMN enabled_modules_json TEXT NOT NULL DEFAULT '[]'")
            if "module_versions_json" not in run_columns:
                conn.execute("ALTER TABLE analysis_runs ADD COLUMN module_versions_json TEXT NOT NULL DEFAULT '{}'")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS capture_set_reconstructions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    capture_set_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_capture_images_set ON capture_images(capture_set_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_set ON analysis_runs(capture_set_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lighting_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    from_mode TEXT,
                    to_mode TEXT NOT NULL,
                    filename TEXT,
                    luma REAL,
                    confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(device_id, timestamp, to_mode)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_history_device_time ON metric_history(device_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_history_device_ignored ON metric_history(device_id, ignored)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_plant_metric_history_plant_time ON plant_metric_history(plant_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lighting_transitions_device_time ON lighting_transitions(device_id, timestamp)")
            conn.execute(
                """
                INSERT OR IGNORE INTO backfill_status
                (id, running, processed, total, message)
                VALUES (1, 0, 0, 0, 'idle')
                """
            )
            conn.commit()
        finally:
            conn.close()
        _initialized = True
        _initialized_path = database_path


def _json(value):
    return json.dumps(value if value is not None else None, separators=(",", ":"))


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def create_capture_set(capture_set_id=None, experiment_id=None, captured_at_utc=None, source="analysis", notes=None):
    capture_set_id = capture_set_id or f"CS_{uuid.uuid4().hex}"
    with connect() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO capture_sets
            (capture_set_id, experiment_id, captured_at_utc, created_at_utc, source, notes)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (capture_set_id, experiment_id, captured_at_utc, _utc_now(), source, notes),
        )
    return capture_set_id


def update_capture_set_qc(capture_set_id, device_count=None, successful_devices=None,
                          failed_devices=None, sync_quality_ms=None, capture_duration_ms=None):
    successful_devices = sorted(set(successful_devices or []))
    failed_devices = sorted(set(failed_devices or []))
    if not successful_devices:
        qc_status = "invalid"
    elif failed_devices:
        qc_status = "marginal"
    elif sync_quality_ms is None:
        qc_status = "not_assessed"
    elif sync_quality_ms <= 50:
        qc_status = "good"
    elif sync_quality_ms <= 200:
        qc_status = "marginal"
    else:
        qc_status = "invalid"
    with connect() as conn:
        conn.execute(
            """UPDATE capture_sets SET
                   device_count=COALESCE(?, device_count),
                   successful_devices_json=?, failed_devices_json=?,
                   sync_quality_ms=?,
                   capture_duration_ms=COALESCE(?, capture_duration_ms),
                   image_count=(SELECT COUNT(*) FROM capture_images WHERE capture_set_id=?),
                   qc_status=?
               WHERE capture_set_id=?""",
            (device_count, _canonical_json(successful_devices), _canonical_json(failed_devices),
             sync_quality_ms, capture_duration_ms, capture_set_id, qc_status, capture_set_id),
        )
    return qc_status


def list_capture_sets(limit=100):
    with connect() as conn:
        rows = conn.execute(
            """SELECT cs.*,
                      (SELECT COUNT(*) FROM capture_images ci WHERE ci.capture_set_id=cs.capture_set_id) AS actual_image_count,
                      (SELECT COUNT(*) FROM analysis_runs ar WHERE ar.capture_set_id=cs.capture_set_id) AS analysis_run_count,
                      (SELECT MAX(created_at_utc) FROM capture_set_reconstructions cr WHERE cr.capture_set_id=cs.capture_set_id) AS reconstruction_updated_at_utc
               FROM capture_sets cs
               ORDER BY COALESCE(cs.captured_at_utc, cs.created_at_utc) DESC
               LIMIT ?""",
            (max(1, min(int(limit), 1000)),),
        ).fetchall()
    records = []
    for row in rows:
        record = dict(row)
        record["successful_devices"] = json.loads(record.pop("successful_devices_json") or "[]")
        record["failed_devices"] = json.loads(record.pop("failed_devices_json") or "[]")
        record["image_count"] = record.pop("actual_image_count")
        records.append(record)
    return records


def capture_set_detail(capture_set_id):
    with connect() as conn:
        set_row = conn.execute(
            """SELECT cs.*,
                      (SELECT COUNT(*) FROM capture_images ci WHERE ci.capture_set_id=cs.capture_set_id) AS actual_image_count,
                      (SELECT COUNT(*) FROM analysis_runs ar WHERE ar.capture_set_id=cs.capture_set_id) AS analysis_run_count
               FROM capture_sets cs WHERE cs.capture_set_id=?""",
            (capture_set_id,),
        ).fetchone()
        if not set_row:
            return None
        images = conn.execute(
            """SELECT ci.*, mh.area, mh.growth_rate_mm2_hr, mh.scale, mh.canopy_coverage,
                      mh.analysis_run_id, mh.previous_capture_set_id
               FROM capture_images ci
               LEFT JOIN metric_history mh ON mh.device_id=ci.device_id AND mh.filename=ci.filename
               WHERE ci.capture_set_id=? ORDER BY ci.device_id""",
            (capture_set_id,),
        ).fetchall()
        image_records = []
        for image in images:
            item = dict(image)
            item["camera_metadata"] = json.loads(item.pop("camera_metadata_json") or "{}")
            item["plants"] = [dict(plant) for plant in conn.execute(
                """SELECT plant_id, tray_id, cell_id, area, coverage, status
                   FROM plant_metric_history WHERE device_id=? AND filename=?""",
                (item["device_id"], item["filename"]),
            ).fetchall()]
            image_records.append(item)
        runs = [dict(row) for row in conn.execute(
            """SELECT ar.analysis_run_id, ar.calibration_snapshot_id, ar.pipeline_version,
                      ar.parameter_hash, ar.parameters_json, ar.enabled_modules_json,
                      ar.module_versions_json, ar.created_at_utc,
                      cal.payload_hash AS calibration_hash, cal.payload_json AS calibration_json
               FROM analysis_runs ar
               LEFT JOIN calibration_snapshots cal ON cal.calibration_snapshot_id=ar.calibration_snapshot_id
               WHERE ar.capture_set_id=? ORDER BY ar.created_at_utc DESC""",
            (capture_set_id,),
        ).fetchall()]
        for run in runs:
            run["parameters"] = json.loads(run.pop("parameters_json") or "{}")
            run["enabled_modules"] = json.loads(run.pop("enabled_modules_json") or "[]")
            run["module_versions"] = json.loads(run.pop("module_versions_json") or "{}")
            run["calibration_snapshot"] = json.loads(run.pop("calibration_json") or "{}")
        reconstruction = conn.execute(
            """SELECT status, result_json, created_at_utc FROM capture_set_reconstructions
               WHERE capture_set_id=? ORDER BY id DESC LIMIT 1""",
            (capture_set_id,),
        ).fetchone()
    record = dict(set_row)
    record["successful_devices"] = json.loads(record.pop("successful_devices_json") or "[]")
    record["failed_devices"] = json.loads(record.pop("failed_devices_json") or "[]")
    record["image_count"] = record.pop("actual_image_count")
    record["images"] = image_records
    record["analysis_runs"] = runs
    record["reconstruction"] = None if reconstruction is None else {
        **dict(reconstruction), "result": json.loads(reconstruction["result_json"]),
    }
    return record


def record_capture_set_reconstruction(capture_set_id, status, result):
    with connect() as conn:
        conn.execute(
            """INSERT INTO capture_set_reconstructions
               (capture_set_id, status, result_json, created_at_utc) VALUES (?, ?, ?, ?)""",
            (capture_set_id, status, _canonical_json(result), _utc_now()),
        )
        conn.execute(
            "UPDATE capture_sets SET reconstruction_status=? WHERE capture_set_id=?",
            (status, capture_set_id),
        )


def capture_set_context(capture_set_id):
    with connect() as conn:
        row = conn.execute(
            "SELECT experiment_id, captured_at_utc FROM capture_sets WHERE capture_set_id=?",
            (capture_set_id,),
        ).fetchone()
    return dict(row) if row else None


def previous_metric_in_experiment(capture_set_id, device_id):
    with connect() as conn:
        current = conn.execute(
            "SELECT experiment_id, captured_at_utc FROM capture_sets WHERE capture_set_id=?",
            (capture_set_id,),
        ).fetchone()
        if not current or not current["experiment_id"] or not current["captured_at_utc"]:
            return None
        row = conn.execute(
            """SELECT mh.area, mh.timestamp, mh.movement_json, mh.scale,
                      cs.capture_set_id, cs.captured_at_utc
               FROM metric_history mh
               JOIN capture_images ci ON ci.device_id=mh.device_id AND ci.filename=mh.filename
               JOIN capture_sets cs ON cs.capture_set_id=ci.capture_set_id
               WHERE mh.device_id=? AND cs.experiment_id=?
                 AND cs.captured_at_utc < ?
               ORDER BY cs.captured_at_utc DESC LIMIT 1""",
            (device_id, current["experiment_id"], current["captured_at_utc"]),
        ).fetchone()
    if not row:
        return None
    result = dict(row)
    result["movement"] = json.loads(result.pop("movement_json") or "{}")
    return result


def register_capture_image(capture_set_id, device_id, filename, image_path, captured_at_utc=None, camera_metadata=None):
    if not os.path.isfile(image_path):
        image_sha256 = None
    else:
        digest = hashlib.sha256()
        with open(image_path, "rb") as image_file:
            for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(chunk)
        image_sha256 = digest.hexdigest()
    with connect() as conn:
        conn.execute(
            """INSERT INTO capture_images
            (device_id, filename, capture_set_id, image_path, captured_at_utc, image_sha256, camera_metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id, filename) DO UPDATE SET
                capture_set_id=excluded.capture_set_id,
                image_path=excluded.image_path,
                captured_at_utc=COALESCE(excluded.captured_at_utc, capture_images.captured_at_utc),
                image_sha256=COALESCE(excluded.image_sha256, capture_images.image_sha256),
                camera_metadata_json=CASE
                    WHEN excluded.camera_metadata_json = '{}' THEN capture_images.camera_metadata_json
                    ELSE excluded.camera_metadata_json
                END""",
            (device_id, filename, capture_set_id, os.path.abspath(image_path), captured_at_utc,
             image_sha256, _canonical_json(camera_metadata or {})),
        )


def ensure_capture_image(device_id, filename, image_path, capture_set_id=None, experiment_id=None,
                         captured_at_utc=None, camera_metadata=None, source="analysis"):
    if capture_set_id is None:
        with connect() as conn:
            existing = conn.execute(
                "SELECT capture_set_id FROM capture_images WHERE device_id = ? AND filename = ?",
                (device_id, filename),
            ).fetchone()
        if existing:
            capture_set_id = existing["capture_set_id"]
        else:
            stable_key = hashlib.sha256(f"{device_id}\0{filename}".encode("utf-8")).hexdigest()[:20]
            capture_set_id = f"CS_{stable_key}"
    create_capture_set(capture_set_id, experiment_id, captured_at_utc, source)
    register_capture_image(capture_set_id, device_id, filename, image_path, captured_at_utc, camera_metadata)
    return capture_set_id


def create_calibration_snapshot(payload):
    payload_json = _canonical_json(payload)
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    snapshot_id = f"CAL_{payload_hash[:20]}"
    with connect() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO calibration_snapshots
            (calibration_snapshot_id, created_at_utc, payload_hash, payload_json)
            VALUES (?, ?, ?, ?)""",
            (snapshot_id, _utc_now(), payload_hash, payload_json),
        )
    return snapshot_id


def create_analysis_run(capture_set_id, calibration_snapshot_id, pipeline_version, parameters,
                        enabled_modules=None, module_versions=None):
    parameters_json = _canonical_json(parameters)
    enabled_modules_json = _canonical_json(sorted(set(enabled_modules or [])))
    module_versions_json = _canonical_json(module_versions or {})
    run_configuration = _canonical_json({
        "parameters": parameters,
        "enabled_modules": json.loads(enabled_modules_json),
        "module_versions": json.loads(module_versions_json),
    })
    parameter_hash = hashlib.sha256(run_configuration.encode("utf-8")).hexdigest()
    analysis_run_id = f"AR_{uuid.uuid4().hex}"
    with connect() as conn:
        existing = conn.execute(
            """SELECT analysis_run_id FROM analysis_runs
               WHERE capture_set_id = ? AND calibration_snapshot_id = ?
                  AND pipeline_version = ? AND parameter_hash = ?""",
              (capture_set_id, calibration_snapshot_id, pipeline_version, parameter_hash),
        ).fetchone()
        if existing:
            return existing["analysis_run_id"]
        conn.execute(
            """INSERT OR IGNORE INTO analysis_runs
            (analysis_run_id, capture_set_id, calibration_snapshot_id, pipeline_version,
             parameter_hash, parameters_json, enabled_modules_json, module_versions_json, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (analysis_run_id, capture_set_id, calibration_snapshot_id, pipeline_version,
             parameter_hash, parameters_json, enabled_modules_json, module_versions_json, _utc_now()),
        )
    return analysis_run_id


def analysis_run_detail(analysis_run_id):
    with connect() as conn:
        row = conn.execute(
            """SELECT ar.*, cal.payload_hash AS calibration_hash, cal.payload_json AS calibration_json,
                      cs.experiment_id, cs.captured_at_utc,
                      (SELECT COUNT(*) FROM capture_images ci WHERE ci.capture_set_id=ar.capture_set_id) AS image_count
               FROM analysis_runs ar
               LEFT JOIN calibration_snapshots cal ON cal.calibration_snapshot_id=ar.calibration_snapshot_id
               LEFT JOIN capture_sets cs ON cs.capture_set_id=ar.capture_set_id
               WHERE ar.analysis_run_id=?""",
            (analysis_run_id,),
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        for field, fallback in (
            ("parameters_json", {}),
            ("enabled_modules_json", []),
            ("module_versions_json", {}),
            ("calibration_json", {}),
        ):
            raw = record.pop(field, None)
            key = field.removesuffix("_json")
            record[key] = json.loads(raw or _canonical_json(fallback))
        record["images"] = [dict(image) for image in conn.execute(
            """SELECT device_id, filename, image_path, image_sha256, captured_at_utc
               FROM capture_images WHERE capture_set_id=? ORDER BY device_id""",
            (record["capture_set_id"],),
        ).fetchall()]
        record["metrics"] = [dict(metric) for metric in conn.execute(
            """SELECT device_id, filename, area, growth_rate_mm2_hr, scale, canopy_coverage
               FROM metric_history WHERE analysis_run_id=?""",
            (analysis_run_id,),
        ).fetchall()]
        record["plant_metrics"] = [dict(metric) for metric in conn.execute(
            """SELECT plant_id, tray_id, cell_id, device_id, filename, area, coverage, status
               FROM plant_metric_history WHERE analysis_run_id=?""",
            (analysis_run_id,),
        ).fetchall()]
        reconstruction_rows = conn.execute(
            """SELECT status, result_json, created_at_utc FROM capture_set_reconstructions
               WHERE capture_set_id=? ORDER BY id DESC""",
            (record["capture_set_id"],),
        ).fetchall()
        record["reconstruction"] = None
        for reconstruction_row in reconstruction_rows:
            reconstruction_result = json.loads(reconstruction_row["result_json"] or "{}")
            if reconstruction_result.get("analysis_run_id") == analysis_run_id:
                record["reconstruction"] = {
                    **dict(reconstruction_row),
                    "result": reconstruction_result,
                }
                break
    return record


def calibration_counts():
    with connect() as conn:
        snapshots = conn.execute("SELECT COUNT(*) FROM calibration_snapshots").fetchone()[0]
    return {"snapshots": int(snapshots)}


def provenance_for_image(device_id, filename):
    with connect() as conn:
        row = conn.execute(
            """SELECT ci.*, cs.experiment_id, cs.source, cs.notes AS capture_notes,
                      ar.analysis_run_id, ar.pipeline_version, ar.parameter_hash,
                      ar.parameters_json, ar.enabled_modules_json, ar.module_versions_json,
                      ar.created_at_utc AS analysis_created_at_utc,
                      cal.calibration_snapshot_id, cal.payload_hash AS calibration_hash,
                      cal.payload_json AS calibration_json, mh.area, mh.growth_rate_mm2_hr,
                      mh.scale, mh.canopy_coverage
               FROM capture_images ci
               JOIN capture_sets cs ON cs.capture_set_id = ci.capture_set_id
               LEFT JOIN metric_history mh_link ON mh_link.device_id = ci.device_id AND mh_link.filename = ci.filename
               LEFT JOIN analysis_runs ar ON ar.analysis_run_id = mh_link.analysis_run_id
               LEFT JOIN calibration_snapshots cal ON cal.calibration_snapshot_id = ar.calibration_snapshot_id
               LEFT JOIN metric_history mh ON mh.device_id = ci.device_id AND mh.filename = ci.filename
               WHERE ci.device_id = ? AND ci.filename = ?
               ORDER BY ar.created_at_utc DESC LIMIT 1""",
            (device_id, filename),
        ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["camera_metadata"] = json.loads(result.pop("camera_metadata_json") or "{}")
        result["parameters"] = json.loads(result.pop("parameters_json") or "{}") if result.get("parameters_json") else None
        result["enabled_modules"] = json.loads(result.pop("enabled_modules_json") or "[]") if result.get("enabled_modules_json") else []
        result["module_versions"] = json.loads(result.pop("module_versions_json") or "{}") if result.get("module_versions_json") else {}
        result["calibration_snapshot"] = json.loads(result.pop("calibration_json") or "{}") if result.get("calibration_json") else None
        result["plants"] = [dict(item) for item in conn.execute(
            """SELECT plant_id, tray_id, cell_id, area, coverage, status
               FROM plant_metric_history WHERE device_id = ? AND filename = ?""",
            (device_id, filename),
        ).fetchall()]
    return result


def upsert_history_point(device_id, entry):
    color = entry.get("color_metrics") or {}
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO metric_history (
                device_id, timestamp, filename, area, growth_rate_mm2_hr, scale,
                detected_scale, scale_rejected, canopy_coverage, green_index,
                color_metrics_json, color_correction_json, segments_json, ignored_segments_json,
                nutrient_json, movement_json, ignored
                , capture_set_id, analysis_run_id, previous_capture_set_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id, filename) DO UPDATE SET
                timestamp=excluded.timestamp,
                area=excluded.area,
                growth_rate_mm2_hr=excluded.growth_rate_mm2_hr,
                scale=excluded.scale,
                detected_scale=excluded.detected_scale,
                scale_rejected=excluded.scale_rejected,
                canopy_coverage=excluded.canopy_coverage,
                green_index=excluded.green_index,
                color_metrics_json=excluded.color_metrics_json,
                color_correction_json=excluded.color_correction_json,
                segments_json=excluded.segments_json,
                ignored_segments_json=excluded.ignored_segments_json,
                nutrient_json=excluded.nutrient_json,
                movement_json=excluded.movement_json,
                capture_set_id=excluded.capture_set_id,
                analysis_run_id=excluded.analysis_run_id,
                previous_capture_set_id=excluded.previous_capture_set_id
            """,
            (
                device_id,
                entry.get("timestamp"),
                entry.get("filename"),
                entry.get("area"),
                entry.get("growth_rate_mm2_hr"),
                entry.get("scale"),
                entry.get("detected_scale"),
                1 if entry.get("scale_rejected") else 0,
                entry.get("canopy_coverage"),
                color.get("green_index"),
                _json(entry.get("color_metrics")),
                _json(entry.get("color_correction")),
                _json(entry.get("segments") or []),
                _json(entry.get("ignored_segments") or []),
                _json(entry.get("nutrient_deficiency")),
                _json(entry.get("movement") or {}),
                1 if entry.get("ignored") else 0,
                entry.get("capture_set_id"),
                entry.get("analysis_run_id"),
                entry.get("previous_capture_set_id"),
            ),
        )


def upsert_plant_metric_points(device_id, timestamp, filename, cells, capture_set_id=None, analysis_run_id=None):
    rows = [cell for cell in (cells or []) if cell.get("plant_id")]
    if not rows:
        return 0
    with connect() as conn:
        for cell in rows:
            conn.execute(
                """
                INSERT INTO plant_metric_history (
                    plant_id, tray_id, cell_id, device_id, timestamp, filename,
                    area, coverage, canopy_pixels, delta_pixels, status
                    , capture_set_id, analysis_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plant_id, filename) DO UPDATE SET
                    tray_id=excluded.tray_id,
                    cell_id=excluded.cell_id,
                    device_id=excluded.device_id,
                    timestamp=excluded.timestamp,
                    area=excluded.area,
                    coverage=excluded.coverage,
                    canopy_pixels=excluded.canopy_pixels,
                    delta_pixels=excluded.delta_pixels,
                    status=excluded.status,
                    capture_set_id=excluded.capture_set_id,
                    analysis_run_id=excluded.analysis_run_id
                """,
                (
                    cell.get("plant_id"),
                    cell.get("tray_id"),
                    cell.get("cell_id"),
                    device_id,
                    timestamp,
                    filename,
                    cell.get("area_mm2"),
                    cell.get("coverage"),
                    cell.get("canopy_pixels"),
                    cell.get("delta_pixels"),
                    cell.get("status"),
                    capture_set_id,
                    analysis_run_id,
                ),
            )
    return len(rows)


def history_for_plant(plant_id, max_points=MAX_DASHBOARD_POINTS):
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM plant_metric_history
            WHERE plant_id = ?
            ORDER BY timestamp
            LIMIT ?
            """,
            (plant_id, max_points),
        ).fetchall()
    return [dict(row) for row in rows]


def latest_lighting_transition(device_id):
    with connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM lighting_transitions
            WHERE device_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (device_id,),
        ).fetchone()
    return dict(row) if row else None


def record_lighting_transition(device_id, timestamp, to_mode, filename=None, luma=None, confidence=None):
    previous = latest_lighting_transition(device_id)
    from_mode = previous.get("to_mode") if previous else None
    if from_mode == to_mode:
        return None
    with connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO lighting_transitions (
                device_id, timestamp, from_mode, to_mode, filename, luma, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (device_id, timestamp, from_mode, to_mode, filename, luma, confidence),
        )
    return {
        "device_id": device_id,
        "timestamp": timestamp,
        "from_mode": from_mode,
        "to_mode": to_mode,
        "filename": filename,
        "luma": luma,
        "confidence": confidence,
    }


def lighting_transitions_for_device(device_id, limit=200):
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM lighting_transitions
            WHERE device_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (device_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def all_lighting_transitions(limit=100000):
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM lighting_transitions
            ORDER BY device_id, timestamp
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def all_plant_history(limit=100000):
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM plant_metric_history
            ORDER BY plant_id, timestamp
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def row_to_history(row):
    def load_json(key, fallback):
        raw = row[key]
        if raw is None:
            return fallback
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return fallback

    return {
        "device_id": row["device_id"],
        "timestamp": row["timestamp"],
        "filename": row["filename"],
        "scale": row["scale"],
        "detected_scale": row["detected_scale"],
        "scale_rejected": bool(row["scale_rejected"]),
        "area": row["area"],
        "growth_rate_mm2_hr": row["growth_rate_mm2_hr"],
        "segments": load_json("segments_json", []),
        "ignored_segments": load_json("ignored_segments_json", []),
        "canopy_coverage": row["canopy_coverage"],
        "color_metrics": load_json("color_metrics_json", {}),
        "color_correction": load_json("color_correction_json", {}),
        "nutrient_deficiency": load_json("nutrient_json", {}),
        "movement": load_json("movement_json", {}),
        "capture_set_id": row["capture_set_id"],
        "analysis_run_id": row["analysis_run_id"],
        "previous_capture_set_id": row["previous_capture_set_id"],
        "ignored": bool(row["ignored"]),
    }


def list_devices():
    with connect() as conn:
        rows = conn.execute("SELECT DISTINCT device_id FROM metric_history ORDER BY device_id").fetchall()
    return [row["device_id"] for row in rows]


def all_device_history(limit_per_device=100000):
    rows = []
    for device_id in list_devices():
        rows.extend(history_for_device(device_id, max_points=limit_per_device))
    return rows


def history_for_device(device_id, max_points=MAX_DASHBOARD_POINTS):
    with connect() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS count FROM metric_history WHERE device_id = ?",
            (device_id,),
        ).fetchone()["count"]
        if count <= max_points:
            rows = conn.execute(
                "SELECT * FROM metric_history WHERE device_id = ? ORDER BY timestamp",
                (device_id,),
            ).fetchall()
        else:
            stride = max(1, count // max_points)
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (ORDER BY timestamp) AS rn
                    FROM metric_history
                    WHERE device_id = ?
                )
                WHERE rn = 1 OR rn % ? = 0
                ORDER BY timestamp
                """,
                (device_id, stride),
            ).fetchall()
    return [row_to_history(row) for row in rows]


def latest_for_device(device_id):
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM metric_history WHERE device_id = ? ORDER BY timestamp DESC LIMIT 1",
            (device_id,),
        ).fetchone()
    return row_to_history(row) if row else None


def ignore_point(device_id, timestamp=None, filename=None, segment_id=None):
    if not timestamp and not filename:
        return 0
    with connect() as conn:
        if segment_id:
            row = conn.execute(
                """
                SELECT id, ignored_segments_json FROM metric_history
                WHERE device_id = ? AND (timestamp = ? OR filename = ?)
                LIMIT 1
                """,
                (device_id, timestamp, filename),
            ).fetchone()
            if not row:
                return 0
            try:
                ignored = set(json.loads(row["ignored_segments_json"] or "[]"))
            except json.JSONDecodeError:
                ignored = set()
            ignored.add(segment_id)
            conn.execute(
                "UPDATE metric_history SET ignored_segments_json = ? WHERE id = ?",
                (_json(sorted(ignored)), row["id"]),
            )
            return 1
        cur = conn.execute(
            """
            UPDATE metric_history
            SET ignored = 1
            WHERE device_id = ? AND (timestamp = ? OR filename = ?)
            """,
            (device_id, timestamp, filename),
        )
        return cur.rowcount


def delete_point(device_id, timestamp=None, filename=None, segment_id=None):
    if not timestamp and not filename:
        return 0
    with connect() as conn:
        if segment_id:
            row = conn.execute(
                """
                SELECT id, segments_json FROM metric_history
                WHERE device_id = ? AND (timestamp = ? OR filename = ?)
                LIMIT 1
                """,
                (device_id, timestamp, filename),
            ).fetchone()
            if not row:
                return 0
            try:
                segments = json.loads(row["segments_json"] or "[]")
            except json.JSONDecodeError:
                segments = []
            remaining = [seg for seg in segments if seg.get("id") != segment_id]
            conn.execute("UPDATE metric_history SET segments_json = ? WHERE id = ?", (_json(remaining), row["id"]))
            return 1 if len(remaining) != len(segments) else 0
        cur = conn.execute(
            """
            DELETE FROM metric_history
            WHERE device_id = ? AND (timestamp = ? OR filename = ?)
            """,
            (device_id, timestamp, filename),
        )
        return cur.rowcount


def reset_device_history(device_id):
    with connect() as conn:
        cur = conn.execute("DELETE FROM metric_history WHERE device_id = ?", (device_id,))
        conn.execute("DELETE FROM metric_rollups WHERE device_id = ?", (device_id,))
        return cur.rowcount


def clear_all_history():
    with connect() as conn:
        metric_rows = conn.execute("DELETE FROM metric_history").rowcount
        conn.execute("DELETE FROM metric_rollups")
        conn.execute(
            """
            UPDATE backfill_status
            SET running = 0, started_at = NULL, finished_at = CURRENT_TIMESTAMP,
                current_device = NULL, processed = 0, total = 0, message = 'cleared'
            WHERE id = 1
            """
        )
        return metric_rows


def refresh_rollups(device_id):
    with connect() as conn:
        conn.execute("DELETE FROM metric_rollups WHERE device_id = ?", (device_id,))
        for period, bucket_expr in (
            ("hour", "substr(timestamp, 1, 13) || ':00:00'"),
            ("day", "substr(timestamp, 1, 10)"),
        ):
            conn.execute(
                f"""
                INSERT INTO metric_rollups (
                    device_id, bucket, period, point_count, avg_area, max_area,
                    min_area, avg_growth_rate, avg_green_index
                )
                SELECT
                    device_id,
                    {bucket_expr} AS bucket,
                    ? AS period,
                    COUNT(*) AS point_count,
                    AVG(area),
                    MAX(area),
                    MIN(area),
                    AVG(growth_rate_mm2_hr),
                    AVG(green_index)
                FROM metric_history
                WHERE device_id = ? AND ignored = 0
                GROUP BY device_id, bucket
                """,
                (period, device_id),
            )


def set_backfill_status(**fields):
    allowed = {"running", "started_at", "finished_at", "current_device", "processed", "total", "message"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return
    with connect() as conn:
        assignments = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE backfill_status SET {assignments} WHERE id = 1", tuple(updates.values()))


def get_backfill_status():
    with connect() as conn:
        row = conn.execute("SELECT * FROM backfill_status WHERE id = 1").fetchone()
    return dict(row) if row else {"running": 0, "message": "idle", "processed": 0, "total": 0}
