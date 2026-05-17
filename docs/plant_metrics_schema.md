# Plant Metrics Schema

`src/pt/schemas/plant_metrics.schema.json` defines the canonical metrics snapshot
for downstream analysis. The first version covers three tracks:

- `growth`: canopy area, growth rate, scale, and future height/leaf count fields.
- `stress`: normalized stress score plus chlorosis, necrosis, wilting, and color-shift signals.
- `circadian`: phase, amplitude, period, and optional light-cycle label.

Runtime metrics should be written to the configured data directory, not the git repo.
Set `PT_DATA_ROOT` when you want captures, debug frames, and `plant_stats.json` in a
specific storage location.

`process_latest_captures()` writes a `metrics` object beside the legacy analysis fields,
so existing dashboard code can keep working while newer consumers read the schema-shaped
record.

## Segmentation & Feature Extraction Framework

The current vision stack already has the architectural groundwork for machine
vision phenotyping. The pipeline detects calibration markers, estimates scale,
extracts a canopy mask, writes debug overlays, and persists frame-level metrics
for dashboard and downstream analysis.

Planned and partially implemented feature families:

- Plant segmentation and mask extraction.
- Contour analysis and canopy bounding boxes.
- Green pixel analysis using HSV thresholds and Excess Green.
- Calibrated canopy metrics in square millimeters.
- Shape descriptors for comparing posture and morphology.

Derived biological features should include:

- Projected leaf area.
- Convex hull area.
- Canopy centroid.
- Canopy density.
- Excess Green (ExG).
- Texture or mask entropy.

These features are useful individually, but become much more valuable as time
series. A single frame can estimate canopy state; repeated frames can estimate
growth, recovery, stress response, and daily movement.

## Circadian Analysis

The platform can support circadian phenotyping because it captures repeated
images under a known photoperiod. Once each plant has reliable segmentation
over time, the system can track movement rhythms across light and dark cycles.

Trackable circadian signals:

- Leaf angle changes.
- Centroid movement.
- Canopy width oscillation.
- Posture variation.

Applications:

- Circadian entrainment.
- Stress disruption.
- Photoperiod response.

The next implementation layer should add optical flow and motion field analysis.
Optical flow can estimate pixel-level movement between adjacent frames, while
motion field summaries can convert that movement into biological descriptors:
directional bias, movement amplitude, quiescent periods, and phase shifts after
light, irrigation, harvest, or stress events.
