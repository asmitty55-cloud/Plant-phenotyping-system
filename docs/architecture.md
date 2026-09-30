# Repository Architecture

This repository stores source, configuration, schemas, deterministic scripts, and docs.
It should not be used as a device sync folder or dataset store.

## Canonical Layout

- `src/pt/core`: image processing, metrics, shared utilities.
- `src/pt/device`: Android capture integration, interrogation, calibration, and device source.
- `src/pt/api`: Flask dashboard and API routes.
- `src/pt/pipeline`: orchestration entrypoints.
- `src/pt/schemas`: versioned JSON schemas for analysis outputs.
- `scripts`: thin CLI wrappers only.
- `configs`: committed configuration templates.
- `docs`: operating notes and project documentation.
- `tests`: automated tests.
- `examples`: small, deterministic sample inputs or usage snippets.

## Artifact Boundary

Do not commit generated files, device-specific state, binary build outputs, captures,
videos, databases, debug logs, or dataset-like exports. Use `PT_DATA_ROOT` to point
runtime output at durable storage outside the repo.

## Calibration

Current analysis expects the back-wall ChArUco target documented in
`docs/calibration.md`.

## Measurement Provenance

SQLite stores capture sets, per-device source images, content-addressed calibration
snapshots, and parameter-hashed analysis runs alongside the existing metric tables.
Metric-history rows link to a capture set and analysis run; plant-cell rows retain
their plant, tray, and cell identifiers with the same links. Calibration snapshots
preserve the calibration payload used by an analysis, while capture-image records
store the image path, SHA-256 digest, UTC capture time when known, and camera metadata.

New scheduled phone captures record their effective camera settings. A device is
associated automatically with an experiment only when exactly one active experiment
lists it; otherwise the capture set remains unassigned for explicit follow-up.
Synchronized volumetric captures share a capture-set ID and record the per-device
monotonic target and measured clock offset. Historical backfills create provenance
records too, but mark capture settings as unavailable because those settings were not
recorded at acquisition time. Existing metric rows are migrated with nullable links
and are not silently represented as fully traceable until processed again.

Use `GET /api/provenance/<device_id>/<filename>` to retrieve an image's capture-set,
experiment, source-image digest, camera metadata, analysis parameters and hash,
calibration snapshot, device-level metric, and linked plant-cell measurements.
`GET /api/analysis-runs/<analysis_run_id>` exposes the run's active module names,
module versions, parameter hash, calibration snapshot, images, metrics, experiment,
and plants. `GET /api/system/manifest` reports the runtime pipeline/schema versions,
module enabled states, calibration counts, and active experiments.

CaptureSets are also the observation unit exposed by the dashboard. `GET
/api/capture-sets` lists observations; `GET /api/capture-sets/<id>` returns its
images, metrics, analysis runs, QC, and latest reconstruction; source images are
served through `/api/capture-images/<device_id>/<filename>`. Use
`POST /api/capture-sets/<id>/reconstruct` to reconstruct only that set's images.
The compatibility `/reconstruct` route selects the newest set with at least two
registered images, rather than combining the latest image from each device.

Growth rates for experiment-assigned CaptureSets compare against the prior metric
for the same device and experiment, and persist `previous_capture_set_id`. Sets
without an experiment assignment retain the existing per-device history behavior.
Synchronized-capture `sync_quality_ms` is a conservative pairwise uncertainty
estimate from the lowest ADB clock-probe round trips, not observed shutter skew.
`capture_duration_ms` measures the capture request through local sync completion for
phone captures, the frame request for network cameras, and the concurrent capture
plus sync interval for synchronized sets. Historical and unattempted captures may
have no duration value.
Current operational QC labels are good at 50 ms or below, marginal above 50 ms
through 200 ms or when a device fails, and invalid above 200 ms or when no device
captures. These thresholds are operational defaults, not scientifically validated
acceptance limits; unsynchronized single-camera sets are labeled not assessed.

## Analysis Modules

`pt.core.analysis.modules` defines `AnalysisObservation`, `AnalysisResult`, and the
`AnalysisModule` protocol. `pt.core.analysis.registry.ANALYSIS_REGISTRY` registers
the canopy, growth, circadian, color-index, and volumetric modules. Modules return
metric mappings and alert records; existing history rows, canonical snapshots,
biology-state JSON, and reconstruction responses retain their current shapes.

Modules can be toggled independently at runtime, for example:

```python
from pt.core.analysis.registry import ANALYSIS_REGISTRY

ANALYSIS_REGISTRY.disable("circadian")
ANALYSIS_REGISTRY.enable("circadian")
```

Image-level canopy segmentation, persisted growth/color history, biological alert
aggregation, and CaptureSet reconstruction invoke the relevant registry entries.
The registry is in-process configuration; toggles are not persisted across restarts.

Biology-state alerts include `trigger_module`, `analysis_run_id`, `capture_set_id`,
`experiment_id`, `plant_ids`, `metric_basis`, and `threshold_basis` when the source
metric has stored provenance. Alert objects remain computed from current histories;
they are not a durable alert-event log.

## Android Transport

Android capture accepts both USB ADB serials and optional Wi-Fi ADB endpoints.
The shared transport and hotspot setup are documented in `docs/wifi_adb.md`.
