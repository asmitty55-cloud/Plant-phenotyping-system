# Scientific validation dataset

This directory is the reference-data contract for measurement accuracy checks. Do not add synthetic or estimated ground truth: each reference should come from a documented manual annotation or an independent measurement procedure.

## Layout

- `canopy_masks/source_images/`: source RGB captures.
- `canopy_masks/reference_masks/`: binary canopy annotations, same pixel dimensions as the source image; nonzero means canopy.
- `area_reference/`: supporting area measurement records and protocol notes.
- `color_reference/`: independently measured patch/canopy color references and acquisition notes.
- `reconstruction_reference/`: synchronized multi-view source captures and independent volume measurements.
- `manifest.json`: the active dataset index. `manifest.example.json` shows the record shape.
- `acceptance.json`: provisional, version-controlled acceptance thresholds.

Keep the first curated set small (about 20-50 captures), but cover the intended working range: species/cultivar, canopy size, camera/device, distance, lighting, and calibration conditions. Record who annotated each mask, when, the annotation protocol, how area and volume were measured, and the units. Resolve disagreements before calling a reference final. Avoid putting treatment or outcome labels into the image IDs if they could bias annotators.

## Manifest records

Each `samples` record can include:

- `id`, `image`, `device_id`
- `validated_mask` (optional, relative to this directory)
- `true_area_cm2` (optional; independently measured area)
- `reference_coverage` (optional; otherwise derived from mask foreground fraction)
- `color_reference` (optional mapping from output color-metric name to independent numeric value)

Each `reconstructions` record groups at least two `{device_id, image}` views and can include `reference_volume_cm3`. When a dataset requires calibration, point `calibration_snapshot` at a frozen JSON calibration payload. Keep source images and masks dimension-aligned and preserve the original captures.

## Run

From the repository root:

```powershell
python validate.py
```

The runner uses the configured `PT_DATA_ROOT` for runtime state unless `--data-root` is supplied. It disables debug-image/mask writes and tray-state updates during validation. The generated `validation/validation_report.json` contains the manifest hash, dataset counts, IoU, precision/recall, area and coverage MAE/RMSE/percent error, color metric errors, optional volume errors, and threshold failures.

An empty manifest produces `status: not_configured`, not a false pass. CI starts running the same command now; once curated references are added, threshold failures make the workflow fail. Threshold values in `acceptance.json` are initial contracts only. Review them against measurement uncertainty and inter-annotator agreement before treating them as scientific acceptance limits.
