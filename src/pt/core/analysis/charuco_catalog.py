import json
import os
from pathlib import Path

from pt.core.utils.path_utils import get_data_root


DEFAULT_SQUARE_MM = 52.0
DEFAULT_MARKER_MM = 39.0
DEFAULT_DICTIONARY = "4X4_50"
DEFAULT_SQUARES_X = 3
DEFAULT_SQUARES_Y = 5


def _repo_root():
    return Path(__file__).resolve().parents[4]


def _catalog_candidates():
    rel = Path("images") / "charuco_boards_3x5_with_rulers_letterfit" / "opencv_charuco_config.json"
    return [
        Path(get_data_root()) / rel,
        _repo_root() / rel,
    ]


def _dict_name(value):
    return str(value or DEFAULT_DICTIONARY).replace("DICT_", "")


def load_catalog():
    for path in _catalog_candidates():
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            boards = []
            for idx, board in enumerate(data.get("boards") or [], start=1):
                ids = [int(v) for v in board.get("ids", [])]
                if not ids:
                    continue
                label = f"board_{idx}_ids_{ids[0]}_{ids[-1]}"
                boards.append({
                    "name": board.get("name") or label,
                    "ids": ids,
                    "file": board.get("file"),
                    "squares_x": int(data.get("squaresX") or DEFAULT_SQUARES_X),
                    "squares_y": int(data.get("squaresY") or DEFAULT_SQUARES_Y),
                    "square_size_mm": float(data.get("design_square_mm") or DEFAULT_SQUARE_MM),
                    "marker_size_mm": float(data.get("design_marker_mm") or DEFAULT_MARKER_MM),
                    "dictionary": _dict_name(data.get("dictionary")),
                })
            if boards:
                return boards
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue

    return [
        {
            "name": "board_ids_7_13",
            "ids": [7, 8, 9, 10, 11, 12, 13],
            "squares_x": DEFAULT_SQUARES_X,
            "squares_y": DEFAULT_SQUARES_Y,
            "square_size_mm": DEFAULT_SQUARE_MM,
            "marker_size_mm": DEFAULT_MARKER_MM,
            "dictionary": DEFAULT_DICTIONARY,
        }
    ]


def default_target():
    return dict(load_catalog()[0])


def wall_marker_size(marker_id):
    try:
        marker_int = int(marker_id)
    except (TypeError, ValueError):
        return None
    for target in load_catalog():
        if marker_int in set(target.get("ids") or []):
            return float(target["marker_size_mm"])
    return None
