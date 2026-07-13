"""Generate letter-fit ArUco sheets and supplemental ChArUco color boards."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


DPI = 600
SQUARE_MM = 52.0
MARKER_MM = 39.0
COLS = 3
ROWS = 5
BOARD_WIDTH_PX = round(COLS * SQUARE_MM / 25.4 * DPI)
BOARD_HEIGHT_PX = round(ROWS * SQUARE_MM / 25.4 * DPI)

DICTIONARIES = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
}


SHEET_LAYOUTS = {
    39: {"marker_mm": 39.0, "cell_mm": 52.0, "cols": 3, "rows": 5, "landscape": False, "name": "compact"},
    52: {"marker_mm": 52.0, "cell_mm": 65.0, "cols": 3, "rows": 4, "landscape": False, "name": "standard"},
    75: {"marker_mm": 75.0, "cell_mm": 90.0, "cols": 3, "rows": 2, "landscape": True, "name": "long_range"},
}


def marker_image(dictionary, marker_id, marker_px):
    image = np.zeros((marker_px, marker_px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px, image, 1)
    return Image.fromarray(image, mode="L")


def save_letter_pdf(images, output_path, width_mm, height_mm, use_landscape=False):
    page_size = landscape(letter) if use_landscape else letter
    pdf = canvas.Canvas(str(output_path), pagesize=page_size)
    page_width, page_height = page_size
    board_width = width_mm * mm
    board_height = height_mm * mm
    x = (page_width - board_width) / 2
    y = (page_height - board_height) / 2
    for image_path in images:
        pdf.drawImage(str(image_path), x, y, width=board_width, height=board_height, mask="auto")
        pdf.showPage()
    pdf.save()


def generate_standalone_sheets(dictionary_name, output_dir, layout=None):
    layout = dict(layout or SHEET_LAYOUTS[39])
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[dictionary_name])
    marker_count = int(dictionary.bytesList.shape[0])
    marker_mm = float(layout["marker_mm"])
    cell_mm = float(layout["cell_mm"])
    cols = int(layout["cols"])
    rows = int(layout["rows"])
    tags_per_page = cols * rows
    marker_px = round(marker_mm / 25.4 * DPI)
    cell_px = round(cell_mm / 25.4 * DPI)
    page_width_px = cols * cell_px
    page_height_px = rows * cell_px
    font = ImageFont.load_default(size=max(24, round(DPI * 0.055)))
    pages = []
    manifest_pages = []

    size_label = str(int(marker_mm) if marker_mm.is_integer() else marker_mm).replace(".", "p")
    for page_index, start_id in enumerate(range(0, marker_count, tags_per_page), start=1):
        ids = list(range(start_id, min(start_id + tags_per_page, marker_count)))
        page = Image.new("L", (page_width_px, page_height_px), 255)
        draw = ImageDraw.Draw(page)
        placements = []
        for index, marker_id in enumerate(ids):
            row, col = divmod(index, cols)
            x = round(col * cell_px + (cell_px - marker_px) / 2)
            y = round(row * cell_px + (cell_px - marker_px) / 2)
            page.paste(marker_image(dictionary, marker_id, marker_px), (x, y))
            label = f"{dictionary_name} ID {marker_id}"
            label_box = draw.textbbox((0, 0), label, font=font)
            label_width = label_box[2] - label_box[0]
            label_y = min(round((row + 1) * cell_px - font.size - 8), y + marker_px + 8)
            draw.text((round(col * cell_px + (cell_px - label_width) / 2), label_y), label, fill=0, font=font)
            draw.rectangle((col * cell_px, row * cell_px, (col + 1) * cell_px - 1, (row + 1) * cell_px - 1), outline=210, width=2)
            placements.append({"id": marker_id, "row": row, "column": col})

        page_path = output_dir / f"aruco_{dictionary_name.lower()}_{size_label}mm_{layout['name']}_page_{page_index:02d}.png"
        page.save(page_path, dpi=(DPI, DPI))
        pages.append(page_path)
        manifest_pages.append({"page": page_index, "file": page_path.name, "ids": ids, "placements": placements})

    pdf_path = output_dir / f"aruco_{dictionary_name.lower()}_{size_label}mm_{layout['name']}_all_tags_letter.pdf"
    save_letter_pdf(pages, pdf_path, cols * cell_mm, rows * cell_mm, layout.get("landscape", False))
    manifest = {
        "dictionary": f"DICT_{dictionary_name}",
        "marker_count": marker_count,
        "marker_size_mm": marker_mm,
        "cell_size_mm": cell_mm,
        "layout": {"columns": cols, "rows": rows, "tags_per_full_page": tags_per_page, "landscape": bool(layout.get("landscape"))},
        "dpi": DPI,
        "pdf": pdf_path.name,
        "pages": manifest_pages,
    }
    manifest_path = output_dir / f"aruco_{dictionary_name.lower()}_{size_label}mm_{layout['name']}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def generate_supplemental_color_boards(repo_root, output_dir):
    source_path = repo_root / "images" / "charuco_boards_3x5_color_reference_letterfit" / "opencv_charuco_color_reference_config.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    used_ids = {int(marker_id) for board in source["boards"] for marker_id in board["ids"]}
    remaining = [marker_id for marker_id in range(50) if marker_id not in used_ids]
    board_groups = [remaining[index:index + 7] for index in range(0, 21, 7)]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    color_by_cell = {
        tuple(patch["cell"]): tuple(patch["rgb"])
        for patch in source["boards"][0].get("color_patches", [])
    }
    board_paths = []
    board_records = []

    def ids_label(ids):
        runs = []
        start = previous = ids[0]
        for marker_id in ids[1:]:
            if marker_id == previous + 1:
                previous = marker_id
                continue
            runs.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = marker_id
        runs.append(str(start) if start == previous else f"{start}-{previous}")
        return "_".join(runs)

    for index, ids in enumerate(board_groups, start=1):
        board = cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE_MM, MARKER_MM, dictionary, np.asarray(ids, dtype=np.int32))
        gray = board.generateImage((BOARD_WIDTH_PX, BOARD_HEIGHT_PX), marginSize=0, borderBits=1)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cell_width = BOARD_WIDTH_PX / COLS
        cell_height = BOARD_HEIGHT_PX / ROWS
        for (col, row), rgb in color_by_cell.items():
            x1, x2 = round(col * cell_width), round((col + 1) * cell_width)
            y1, y2 = round(row * cell_height), round((row + 1) * cell_height)
            image[y1:y2, x1:x2] = rgb
        path = output_dir / f"charuco_3x5_ids_{ids_label(ids)}_52mm_color_ref.png"
        Image.fromarray(image).save(path, dpi=(DPI, DPI))
        board_paths.append(path)
        board_records.append({"ids": ids, "file": path.name, "color_patches": source["boards"][0].get("color_patches", [])})

    pdf_path = output_dir / "charuco_color_reference_supplemental_4x4_50.pdf"
    save_letter_pdf(board_paths, pdf_path, COLS * SQUARE_MM, ROWS * SQUARE_MM)
    result = {
        "dictionary": "DICT_4X4_50",
        "squaresX": COLS,
        "squaresY": ROWS,
        "design_square_mm": SQUARE_MM,
        "design_marker_mm": MARKER_MM,
        "existing_ids": sorted(used_ids),
        "boards": board_records,
        "remaining_standalone_id": remaining[21],
        "pdf": pdf_path.name,
    }
    (output_dir / "supplemental_charuco_config.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", choices=sorted(DICTIONARIES), default="4X4_50")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-charuco", action="store_true")
    parser.add_argument("--sizes", nargs="*", type=int, choices=sorted(SHEET_LAYOUTS))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output or repo_root / "images" / "aruco_print_sheets"
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = args.sizes or ([39, 52, 75] if args.dictionary == "6X6_250" else [39])
    standalone_sets = [generate_standalone_sheets(args.dictionary, output_dir, SHEET_LAYOUTS[size]) for size in sizes]
    if args.dictionary == "4X4_50" and not args.skip_charuco:
        supplemental = generate_supplemental_color_boards(repo_root, output_dir)
        print(f"Generated {len(supplemental['boards'])} supplemental color boards; ID {supplemental['remaining_standalone_id']} remains standalone.")
    for standalone in standalone_sets:
        print(f"Generated {standalone['marker_count']} {args.dictionary} tags at {standalone['marker_size_mm']} mm across {len(standalone['pages'])} letter pages in {output_dir}")


if __name__ == "__main__":
    main()
