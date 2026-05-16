# Calibration Target

The active wall reference is a 3 by 5 ChArUco-style board printed full page on
8.5 x 11 inch paper and mounted on the back wall.

Measured geometry:

- Chessboard square size: `51.28 mm`
- ArUco marker block size: `41.18 mm`
- Dictionary: `DICT_4X4_50`

The analyzer first attempts ChArUco corner spacing. If OpenCV cannot interpolate
the chessboard corners from a frame, it falls back to averaging detected ArUco
marker side lengths using the measured `41.18 mm` block size.

Because this target is mounted on the back wall, the resulting scale is a wall-plane
scale. Keep plants at a consistent depth relative to the wall, or add a plant-plane
target if absolute area measurements need to account for depth.

## Canopy and Deficiency Tracking

Canopy measurement uses the wall-plane scale plus a vegetation mask built from HSV
color thresholds and an excess-green index. Each capture records:

- canopy area in square millimeters
- canopy pixel coverage
- canopy bounding box
- color metrics such as green index, chlorosis ratio, saturation, and brightness

Nutrient deficiency flags are baseline-relative. The system collects at least three
valid captures per device before flagging drift. This is intentional: different crops
and cultivars can have different normal colors, so the first healthy captures define
the comparison point for later yellowing, desaturation, and green-index loss.
