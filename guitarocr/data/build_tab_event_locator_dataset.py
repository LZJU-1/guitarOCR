from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from PIL import Image

from guitarocr.data.build_score_event_locator_dataset import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    TILE_OVERLAP,
    crop_with_padding,
    event_kind,
)
from guitarocr.data.build_score_rhythm_dataset import read_jsonl
from guitarocr.paths import DATABASE_ROOT


def tile_tab_measure(
    page: Image.Image,
    measure_bbox: list[float],
    string_y: list[float],
    events: list[dict],
) -> list[tuple[Image.Image, list[dict], dict]]:
    if len(string_y) < 4:
        raise ValueError(f"Expected at least four TAB strings, got {string_y}")
    spacing = sum(
        string_y[index + 1] - string_y[index]
        for index in range(len(string_y) - 1)
    ) / (len(string_y) - 1)
    if spacing <= 0:
        raise ValueError(f"Invalid TAB string spacing: {string_y}")

    measure_x, _, measure_width, _ = measure_bbox
    left = math.floor(measure_x - 0.75 * spacing)
    right = math.ceil(measure_x + measure_width + 0.75 * spacing)
    top = math.floor(string_y[0] - 6.0 * spacing)
    bottom = math.ceil(string_y[-1] + 6.0 * spacing)
    crop = crop_with_padding(page, (left, top, right, bottom))

    scale = INPUT_HEIGHT / crop.height
    resized_width = max(1, round(crop.width * scale))
    resized = crop.resize((resized_width, INPUT_HEIGHT), Image.Resampling.LANCZOS)
    if resized_width <= INPUT_WIDTH:
        tile_starts = [0]
        canvas_offset_x = (INPUT_WIDTH - resized_width) // 2
    else:
        stride = INPUT_WIDTH - TILE_OVERLAP
        tile_starts = list(range(0, resized_width - INPUT_WIDTH + 1, stride))
        last_start = resized_width - INPUT_WIDTH
        if not tile_starts or tile_starts[-1] != last_start:
            tile_starts.append(last_start)
        canvas_offset_x = 0

    results: list[tuple[Image.Image, list[dict], dict]] = []
    for tile_start in tile_starts:
        output = Image.new("L", (INPUT_WIDTH, INPUT_HEIGHT), 255)
        if resized_width <= INPUT_WIDTH:
            output.paste(resized, (canvas_offset_x, 0))
        else:
            output.paste(
                resized.crop((tile_start, 0, tile_start + INPUT_WIDTH, INPUT_HEIGHT)),
                (0, 0),
            )
        output_events = []
        for event in events:
            center_x = (float(event["x"]) - left) * scale + canvas_offset_x - tile_start
            if 0.0 <= center_x < INPUT_WIDTH:
                output_events.append({
                    "event_id": event["event_id"],
                    "beat_index": int(event["beat_index"]),
                    "kind": event_kind(event),
                    "x": center_x,
                    "x_page": float(event["x"]),
                })
        transform = {
            "page_crop_xyxy": [left, top, right, bottom],
            "scale": scale,
            "offset_x": canvas_offset_x,
            "tile_start_x": tile_start,
            "resized_size": [resized_width, INPUT_HEIGHT],
            "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
            "staff_spacing_input": spacing * scale,
            "tab_string_y_input": [(value - top) * scale for value in string_y],
        }
        results.append((output, output_events, transform))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pure-TAB measure tiles for x-axis event localisation."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    args = parser.parse_args()
    database = args.database.resolve()
    root = database / "tab_event_locator"
    dataset = root / "dataset"
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        for leaf, pattern in (("images", "*.png"), ("labels", "*.json")):
            directory = dataset / split / leaf
            directory.mkdir(parents=True, exist_ok=True)
            for path in directory.glob(pattern):
                path.unlink()

    source_records = read_jsonl(database / "manifests" / "sources.jsonl")
    split_by_source = {record["id"]: record["split"] for record in source_records}
    page_labels = sorted((database / "labels" / "pages" / "tab_only").glob("*/*.json"))
    if not page_labels:
        raise FileNotFoundError("Build tab_only page annotations first")

    rows: dict[str, list[dict]] = defaultdict(list)
    sources: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    for page_path in page_labels:
        page_label = json.loads(page_path.read_text(encoding="utf-8"))
        source_id = page_label["source_id"]
        split = split_by_source[source_id]
        sources[split].add(source_id)
        with Image.open(database / page_label["image"]) as opened:
            page = opened.convert("L")
        for measure in page_label["measures"]:
            tiles = tile_tab_measure(
                page,
                measure["bbox"],
                [float(value) for value in measure["tab_staff"]["string_y"]],
                measure.get("events", []),
            )
            assigned = {event["event_id"]: 0 for event in measure.get("events", [])}
            base = (
                f"{source_id}_p{int(page_label['page_index']):03d}"
                f"_m{int(measure['measure_number']):03d}"
            )
            counts["measures"] += 1
            counts["unique_events"] += len(assigned)
            for tile_index, (image, events, transform) in enumerate(tiles):
                sample_id = f"{base}_w{tile_index:02d}"
                image_path = dataset / split / "images" / f"{sample_id}.png"
                label_path = dataset / split / "labels" / f"{sample_id}.json"
                image.save(image_path, format="PNG", compress_level=1)
                for event in events:
                    assigned[event["event_id"]] += 1
                    counts[f"event_kind:{event['kind']}"] += 1
                label = {
                    "schema_version": "1.0",
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "split": split,
                    "page_index": int(page_label["page_index"]),
                    "measure_number": int(measure["measure_number"]),
                    "tile_index": tile_index,
                    "source_page_image": page_label["image"],
                    "image_size": [INPUT_WIDTH, INPUT_HEIGHT],
                    "transform": transform,
                    "events": events,
                }
                label_path.write_text(
                    json.dumps(label, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                rows[split].append({
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "split": split,
                    "image": image_path.relative_to(database).as_posix(),
                    "label": label_path.relative_to(database).as_posix(),
                    "event_count": len(events),
                })
                counts["tiles"] += 1
                counts["event_assignments"] += len(events)
            missing = [key for key, value in assigned.items() if value == 0]
            if missing:
                raise ValueError(f"Events were not assigned to a tile in {base}: {missing}")

    for split in ("train", "validation", "test"):
        (manifests / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows[split]),
            encoding="utf-8",
        )
    summary = {
        "schema_version": "1.0",
        "task": "TuxGuitar tab_only x-axis event localisation",
        "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
        "tile_overlap": TILE_OVERLAP,
        "counts": dict(sorted(counts.items())),
        "splits": {
            split: {
                "sources": len(sources[split]),
                "tiles": len(rows[split]),
                "event_assignments": sum(row["event_count"] for row in rows[split]),
            }
            for split in ("train", "validation", "test")
        },
        "scope": "Ground-truth TAB geometry creates training tiles; inference recovers it from pixels.",
    }
    (manifests / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
