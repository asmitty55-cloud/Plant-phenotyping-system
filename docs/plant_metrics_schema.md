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
