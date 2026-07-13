from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from guitarocr.data.build_tab_detector_dataset import tile_measure
from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import DATABASE_ROOT, WEIGHTS_ROOT
from guitarocr.training.train_tab_detector import box_iou, decode_detections


def group_runs(values: np.ndarray, maximum_gap: int = 1) -> list[list[int]]:
    groups: list[list[int]] = []
    for raw_value in values:
        value = int(raw_value)
        if not groups or value > groups[-1][-1] + maximum_gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def detect_tab_geometry(image: Image.Image) -> list[dict]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    black = gray < 160
    height, width = black.shape

    minimum_line_run = max(60, round(width * 0.055))
    longest_runs = np.zeros(height, dtype=np.int32)
    for row_index, row in enumerate(black):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        if edges.size:
            longest_runs[row_index] = int((edges[1::2] - edges[::2]).max(initial=0))
    row_groups = group_runs(np.where(longest_runs >= minimum_line_run)[0])
    candidate_rows = [max(group, key=lambda y: int(longest_runs[y])) for group in row_groups]
    chains: list[list[int]] = []
    for row_index, first_y in enumerate(candidate_rows):
        for second_y in candidate_rows[row_index + 1 :]:
            spacing = second_y - first_y
            if spacing < 8:
                continue
            if spacing > 40:
                break
            tolerance = max(1.0, spacing * 0.1)
            chain = [first_y]
            current = first_y
            while len(chain) < 8:
                expected = current + spacing
                options = [
                    value for value in candidate_rows
                    if value > current and abs(value - expected) <= tolerance
                ]
                if not options:
                    break
                current = min(options, key=lambda value: abs(value - expected))
                chain.append(current)
            if len(chain) >= 4:
                chains.append(chain)

    chains.sort(key=lambda chain: (-len(chain), chain[0]))
    selected: list[list[int]] = []
    for chain in chains:
        if any(set(chain) & set(existing) for existing in selected):
            continue
        selected.append(chain)
    selected.sort(key=lambda chain: chain[0])

    staffs: list[dict] = []
    measure_number = 1
    for staff_index, string_y in enumerate(selected):
        spacing = float(np.median(np.diff(string_y)))
        top_y, bottom_y = string_y[0], string_y[-1]
        vertical_counts = black[top_y : bottom_y + 1].sum(axis=0)
        bar_groups = group_runs(
            np.where(vertical_counts > (bottom_y - top_y + 1) * 0.90)[0]
        )
        merged: list[list[int]] = []
        for group in bar_groups:
            if merged and group[0] - merged[-1][-1] <= max(3.0, spacing * 0.75):
                merged[-1].extend(group)
            else:
                merged.append(group)
        if len(merged) < 2:
            continue

        raw_boundaries = [merged[0][0], *(group[0] for group in merged[1:-1]), merged[-1][-1]]
        minimum_measure_width = max(60.0, spacing * 4.0)
        boundaries = [raw_boundaries[0]]
        for boundary in raw_boundaries[1:-1]:
            if boundary - boundaries[-1] >= minimum_measure_width:
                boundaries.append(boundary)
        final_boundary = raw_boundaries[-1]
        if final_boundary - boundaries[-1] < minimum_measure_width and len(boundaries) > 1:
            boundaries.pop()
        boundaries.append(final_boundary)

        measures: list[dict] = []
        staff_top = top_y - spacing / 2.0
        staff_height = (bottom_y - top_y) + spacing
        for local_index in range(len(boundaries) - 1):
            left, right = boundaries[local_index], boundaries[local_index + 1]
            measures.append(
                {
                    "measure_number": measure_number,
                    "system_measure_index": local_index,
                    "bbox": [float(left), staff_top, float(right - left), staff_height],
                    "symbols": [],
                    "events": [],
                }
            )
            measure_number += 1
        staffs.append(
            {
                "staff_index": staff_index,
                "string_count": len(string_y),
                "string_y": [float(value) for value in string_y],
                "spacing": spacing,
                "boundaries": [float(value) for value in boundaries],
                "measures": measures,
            }
        )
    return staffs


def map_detection_to_page(detection: dict, transform: dict) -> dict:
    left, top, _, _ = transform["page_crop_xyxy"]
    scale = transform["scale"]
    offset_x, offset_y = transform["offset"]
    tile_start = transform["tile_start_x"]
    x, y, width, height = detection["bbox"]
    page_bbox = [
        (x + tile_start - offset_x) / scale + left,
        (y - offset_y) / scale + top,
        width / scale,
        height / scale,
    ]
    return {**detection, "bbox": page_bbox}


def nms_same_class(detections: list[dict], threshold: float = 0.4) -> list[dict]:
    kept: list[dict] = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if any(
            existing["class_index"] == detection["class_index"]
            and box_iou(existing["bbox"], detection["bbox"]) >= threshold
            for existing in kept
        ):
            continue
        kept.append(detection)
    return sorted(kept, key=lambda item: item["bbox"][0])


def group_symbols(symbols: list[dict], string_y: list[float], spacing: float) -> list[dict]:
    for symbol in symbols:
        x, y, width, height = symbol["bbox"]
        center_y = y + height / 2.0
        symbol["string"] = min(range(len(string_y)), key=lambda index: abs(string_y[index] - center_y)) + 1

    tokens: list[dict] = []
    for string_number in range(1, len(string_y) + 1):
        row = sorted((item for item in symbols if item["string"] == string_number), key=lambda item: item["bbox"][0])
        current: list[dict] = []
        for symbol in row:
            if not current:
                current = [symbol]
                continue
            previous = current[-1]
            previous_right = previous["bbox"][0] + previous["bbox"][2]
            gap = symbol["bbox"][0] - previous_right
            current_digits = "".join(
                item["class"].removeprefix("digit_")
                for item in current
                if item["class"].startswith("digit_")
            )
            next_digit = symbol["class"].removeprefix("digit_")
            candidate = current_digits + next_digit
            can_join = (
                len(current) == 1
                and previous["class"].startswith("digit_")
                and symbol["class"].startswith("digit_")
                and current_digits in {"1", "2", "3"}
                and 10 <= int(candidate) <= 36
                and gap < spacing * 0.18
            )
            if can_join:
                current.append(symbol)
            else:
                tokens.append(make_token(current, string_number))
                current = [symbol]
        if current:
            tokens.append(make_token(current, string_number))

    events: list[dict] = []
    for token in sorted(tokens, key=lambda item: item["center_x"]):
        target = next(
            (event for event in events if abs(event["x"] - token["center_x"]) <= spacing * 0.20),
            None,
        )
        if target is None:
            target = {"x": token["center_x"], "notes": []}
            events.append(target)
        target["notes"].append({"string": token["string"], "fret": token["value"]})
        target["x"] = sum(note["center_x"] for note in target.setdefault("_tokens", []) + [token]) / (
            len(target["_tokens"]) + 1
        )
        target["_tokens"].append(token)
    for event in events:
        event.pop("_tokens", None)
        event["notes"].sort(key=lambda note: note["string"])
    return events


def make_token(symbols: list[dict], string_number: int) -> dict:
    left = min(symbol["bbox"][0] for symbol in symbols)
    right = max(symbol["bbox"][0] + symbol["bbox"][2] for symbol in symbols)
    if any(symbol["class"] == "dead_x" for symbol in symbols):
        value: int | str = "X"
    else:
        value = int("".join(symbol["class"].removeprefix("digit_") for symbol in symbols))
    return {"string": string_number, "value": value, "center_x": (left + right) / 2.0}


def draw_overlay(image: Image.Image, staffs: list[dict], output: Path) -> None:
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for staff in staffs:
        for y in staff["string_y"]:
            draw.line((staff["boundaries"][0], y, staff["boundaries"][-1], y), fill=(0, 180, 70), width=1)
        for boundary in staff["boundaries"]:
            draw.line((boundary, staff["string_y"][0], boundary, staff["string_y"][-1]), fill=(0, 150, 255), width=2)
        for measure in staff["measures"]:
            for symbol in measure["symbols"]:
                x, y, width, height = symbol["bbox"]
                color = (220, 0, 200) if symbol["class"] == "dead_x" else (230, 40, 40)
                draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a TuxGuitar tab_only page using only its pixels.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=WEIGHTS_ROOT / "tab_symbol_detector.pt",
    )
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATABASE_ROOT / "tab_detector" / "page_inference",
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    with Image.open(image_path) as opened:
        page_image = opened.convert("L")
    staffs = detect_tab_geometry(page_image)
    if not staffs:
        raise RuntimeError("No TuxGuitar TAB staff was detected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = TabSymbolDetector(len(classes)).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])

    tile_records: list[tuple[Image.Image, dict, dict, dict]] = []
    for staff in staffs:
        for measure in staff["measures"]:
            for tile_image, _, transform in tile_measure(page_image, measure["bbox"], []):
                tile_records.append((tile_image, transform, staff, measure))

    for start in range(0, len(tile_records), 16):
        batch_records = tile_records[start : start + 16]
        arrays = [np.asarray(record[0], dtype=np.float32) / 255.0 for record in batch_records]
        tensor = torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(device)
        tensor = (tensor - 0.5) / 0.5
        with torch.inference_mode():
            decoded = decode_detections(model(tensor), threshold=args.threshold)
        for detections, (_, transform, _, measure) in zip(decoded, batch_records):
            measure["symbols"].extend(map_detection_to_page(item, transform) for item in detections)

    for staff in staffs:
        for measure in staff["measures"]:
            measure["symbols"] = nms_same_class(measure["symbols"])
            for symbol in measure["symbols"]:
                symbol["class"] = classes[symbol.pop("class_index")]
            measure["events"] = group_symbols(measure["symbols"], staff["string_y"], staff["spacing"])

    source_key = f"{image_path.parent.name}_{image_path.stem}"
    output_dir = args.output_dir.resolve()
    json_path = output_dir / f"{source_key}.json"
    overlay_path = output_dir / f"{source_key}_overlay.png"
    output = {
        "schema_version": "1.0",
        "image": str(image_path),
        "image_size": [page_image.width, page_image.height],
        "scope": "TuxGuitar tab_only pixel-only geometry plus TAB digit/X detection",
        "staffs": staffs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    draw_overlay(page_image, staffs, overlay_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "overlay": str(overlay_path),
                "staffs": len(staffs),
                "measures": sum(len(staff["measures"]) for staff in staffs),
                "symbols": sum(len(measure["symbols"]) for staff in staffs for measure in staff["measures"]),
                "events": sum(len(measure["events"]) for staff in staffs for measure in staff["measures"]),
            }
        )
    )


if __name__ == "__main__":
    main()
