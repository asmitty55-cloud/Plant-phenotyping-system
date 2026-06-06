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

## Android Transport

Android capture accepts both USB ADB serials and optional Wi-Fi ADB endpoints.
The shared transport and hotspot setup are documented in `docs/wifi_adb.md`.
