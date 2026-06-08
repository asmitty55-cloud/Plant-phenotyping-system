import json
import os
import sqlite3
import threading
from contextlib import contextmanager

from pt.core.utils.path_utils import get_data_root


DB_FILENAME = "plant_observatory.sqlite3"
MAX_DASHBOARD_POINTS = 10000
_init_lock = threading.Lock()
_initialized = False


def db_path():
    return os.path.join(get_data_root(), DB_FILENAME)


@contextmanager
def connect():
    ensure_db()
    conn = sqlite3.connect(db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def ensure_db():
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        os.makedirs(get_data_root(), exist_ok=True)
        conn = sqlite3.connect(db_path(), timeout=30)
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
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(plant_id, filename)
                )
                """
            )
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


def _json(value):
    return json.dumps(value if value is not None else None, separators=(",", ":"))


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
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                movement_json=excluded.movement_json
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
            ),
        )


def upsert_plant_metric_points(device_id, timestamp, filename, cells):
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
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plant_id, filename) DO UPDATE SET
                    tray_id=excluded.tray_id,
                    cell_id=excluded.cell_id,
                    device_id=excluded.device_id,
                    timestamp=excluded.timestamp,
                    area=excluded.area,
                    coverage=excluded.coverage,
                    canopy_pixels=excluded.canopy_pixels,
                    delta_pixels=excluded.delta_pixels,
                    status=excluded.status
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
