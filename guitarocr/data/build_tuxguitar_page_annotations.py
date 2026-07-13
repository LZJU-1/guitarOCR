from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from guitarocr.paths import DATABASE_ROOT


PAGE_WIDTH = 550.0
PAGE_HEIGHT = 800.0


def scale_bbox(bbox: list[float], sx: float, sy: float) -> list[float]:
    x, y, width, height = bbox
    return [x * sx, y * sy, width * sx, height * sy]


def xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, width, height = bbox
    return x, y, x + width, y + height


def clean_files(directory: Path, pattern: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()


def convert_source(database: Path, layout_path: Path, layout: str) -> list[dict]:
    source = json.loads(layout_path.read_text(encoding="utf-8"))
    source_id = source["source_id"]
    image_root = database / "output" / "images" / layout / source_id
    label_group = "tab_only" if layout == "tab_only" else "score_tab_symbols"
    label_root = database / "labels" / "pages" / label_group / source_id
    overlay_root = database / "output" / "annotation_overlays" / label_group / source_id
    clean_files(label_root, "page_*.json")
    clean_files(overlay_root, "page_*.png")

    records: list[dict] = []
    for page in source["pages"]:
        page_index = int(page["page_index"])
        image_path = image_root / f"page_{page_index:03d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing rendered page for annotation: {image_path}")

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        sx, sy = width / PAGE_WIDTH, height / PAGE_HEIGHT
        output_measures: list[dict] = []
        symbol_counts: Counter[str] = Counter()
        invalid_boxes: list[dict] = []

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for measure in page["measures"]:
            measure_bbox = scale_bbox(measure["bbox"], sx, sy)
            staff_bbox = scale_bbox(measure["tab_staff"]["bbox"], sx, sy)
            string_y = [value * sy for value in measure["tab_staff"]["string_y"]]
            symbols: list[dict] = []
            events: list[dict] = []

            draw.rectangle(xyxy(measure_bbox), outline=(0, 170, 255), width=2)
            draw.rectangle(xyxy(staff_bbox), outline=(0, 180, 70), width=2)
            draw.text(
                (measure_bbox[0] + 3, measure_bbox[1] + 2),
                f"m{measure['measure_number']}",
                fill=(0, 100, 220),
            )

            for symbol in measure["symbols"]:
                bbox = scale_bbox(symbol["bbox"], sx, sy)
                center = [symbol["center"][0] * sx, symbol["center"][1] * sy]
                x, y, box_width, box_height = bbox
                if box_width <= 0 or box_height <= 0 or x + box_width < 0 or y + box_height < 0 \
                        or x >= width or y >= height:
                    invalid_boxes.append({"class": symbol["class"], "bbox": bbox})
                output_symbol = dict(symbol)
                output_symbol["bbox"] = bbox
                output_symbol["center"] = center
                symbols.append(output_symbol)
                symbol_counts[symbol["class"]] += 1
                color = (230, 40, 40) if symbol["class"].startswith("digit_") else (220, 0, 200)
                draw.rectangle(xyxy(bbox), outline=color, width=2)

            for event in measure.get("events", []):
                copied = dict(event)
                copied["x"] = float(event["x"]) * sx
                events.append(copied)
                draw.line(
                    (copied["x"], string_y[0] - 5 * (string_y[1] - string_y[0]),
                     copied["x"], string_y[-1] + 5 * (string_y[1] - string_y[0])),
                    fill=(255, 140, 0), width=1,
                )

            output_measures.append(
                {
                    "measure_index": measure["measure_index"],
                    "measure_number": measure["measure_number"],
                    "bbox": measure_bbox,
                    "tab_staff": {"bbox": staff_bbox, "string_y": string_y},
                    "symbols": symbols,
                    "events": events,
                }
            )

        if invalid_boxes:
            raise ValueError(f"Found invalid boxes in {image_path}: {invalid_boxes[:5]}")

        page_label = {
            "schema_version": "1.0",
            "source_id": source_id,
            "layout": layout,
            "page_index": page_index,
            "image": image_path.relative_to(database).as_posix(),
            "image_size": {"width": width, "height": height},
            "coordinate_space": {"name": "png_pixels_top_left", "scale_x": sx, "scale_y": sy},
            "measures": output_measures,
            "summary": {
                "measure_count": len(output_measures),
                "symbol_count": sum(symbol_counts.values()),
                "classes": dict(sorted(symbol_counts.items())),
            },
        }
        label_path = label_root / f"page_{page_index:03d}.json"
        overlay_path = overlay_root / f"page_{page_index:03d}.png"
        label_path.write_text(json.dumps(page_label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        overlay.save(overlay_path, format="PNG", compress_level=1)
        records.append(
            {
                "sample_id": f"{source_id}_{layout}_p{page_index:03d}",
                "source_id": source_id,
                "layout": layout,
                "page_index": page_index,
                "image": image_path.relative_to(database).as_posix(),
                "label": label_path.relative_to(database).as_posix(),
                "overlay": overlay_path.relative_to(database).as_posix(),
                "measure_count": len(output_measures),
                "symbol_count": sum(symbol_counts.values()),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TuxGuitar logical layout coordinates to PNG pixels and draw QA overlays."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--layout", choices=("tab_only", "score_tab"), default="tab_only")
    args = parser.parse_args()

    database = args.database.resolve()
    layout_group = "tab_only" if args.layout == "tab_only" else "score_tab_symbols"
    layout_root = database / "labels" / "layout" / layout_group
    layout_paths = sorted(layout_root.glob("*.json"))
    if args.limit > 0:
        layout_paths = layout_paths[: args.limit]
    if not layout_paths:
        raise FileNotFoundError(f"No TuxGuitar layout labels found in {layout_root}")

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(convert_source, database, path, args.layout) for path in layout_paths]
        for index, future in enumerate(as_completed(futures), start=1):
            records.extend(future.result())
            if index % 20 == 0 or index == len(futures):
                print(f"BUILT_TAB_PAGE_SOURCES={index}/{len(futures)}", flush=True)
    records.sort(key=lambda record: record["sample_id"])

    manifest_prefix = "tab_symbol" if args.layout == "tab_only" else "score_tab_symbol"
    manifest_path = database / "manifests" / f"{manifest_prefix}_annotations.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "1.0",
        "source_count": len({record["source_id"] for record in records}),
        "page_count": len(records),
        "measure_count": sum(record["measure_count"] for record in records),
        "symbol_count": sum(record["symbol_count"] for record in records),
        "layout": args.layout,
        "scope": f"TuxGuitar 2.0.1 {args.layout} digit/X detection ground truth",
    }
    summary_path = database / "manifests" / f"{manifest_prefix}_annotation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
