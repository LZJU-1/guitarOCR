from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from guitarocr.data.build_score_rhythm_dataset import SPLIT_OVERRIDES, read_jsonl
from guitarocr.paths import DATABASE_ROOT


INPUT_WIDTH = 512
INPUT_HEIGHT = 192
TILE_OVERLAP = 96


def clean_dataset(root: Path) -> None:
    expected = (root.parent / "dataset").resolve()
    if root.resolve() != expected:
        raise ValueError(f"Refusing to clean unexpected event-locator path: {root}")
    for split in ("train", "validation", "test"):
        for leaf, pattern in (("images", "*.png"), ("labels", "*.json")):
            directory = root / split / leaf
            directory.mkdir(parents=True, exist_ok=True)
            for path in directory.glob(pattern):
                if path.is_file():
                    path.unlink()


def crop_with_padding(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    output = Image.new("L", (right - left, bottom - top), 255)
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
        output.paste(image.crop(source_box).convert("L"), (source_box[0] - left, source_box[1] - top))
    return output


def event_kind(event: dict) -> str:
    visible = [voice for voice in event["voices"] if voice["visible"]]
    if any(not voice["rest"] for voice in visible):
        return "note"
    return "rest"


def tile_score_measure(
    page: Image.Image,
    measure_bbox: list[float],
    score_line_y: list[float],
    events: list[dict],
) -> list[tuple[Image.Image, list[dict], dict]]:
    """Create scale-preserving score tiles and transform event x positions.

    The vertical window deliberately matches build_event_crop: eight staff
    spacings above and below the five lines. This keeps locator and rhythm-CNN
    glyph scale compatible.
    """
    if len(score_line_y) != 5:
        raise ValueError(f"Expected five score lines, got {score_line_y}")
    spacing = sum(score_line_y[index + 1] - score_line_y[index] for index in range(4)) / 4.0
    if spacing <= 0:
        raise ValueError(f"Invalid score line spacing: {score_line_y}")

    measure_x, _, measure_width, _ = measure_bbox
    left = math.floor(measure_x - 0.75 * spacing)
    right = math.ceil(measure_x + measure_width + 0.75 * spacing)
    top = math.floor(score_line_y[0] - 8.0 * spacing)
    bottom = math.ceil(score_line_y[-1] + 8.0 * spacing)
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
            output.paste(resized.crop((tile_start, 0, tile_start + INPUT_WIDTH, INPUT_HEIGHT)), (0, 0))

        output_events: list[dict] = []
        for event in events:
            center_x = (float(event["x"]) - left) * scale + canvas_offset_x - tile_start
            if not (0.0 <= center_x < INPUT_WIDTH):
                continue
            output_events.append(
                {
                    "event_id": event["event_id"],
                    "beat_index": int(event["beat_index"]),
                    "kind": event_kind(event),
                    "x": center_x,
                    "x_page": float(event["x"]),
                }
            )

        transform = {
            "page_crop_xyxy": [left, top, right, bottom],
            "scale": scale,
            "offset_x": canvas_offset_x,
            "tile_start_x": tile_start,
            "resized_size": [resized_width, INPUT_HEIGHT],
            "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
            "staff_spacing_input": spacing * scale,
            "score_line_y_input": [(value - top) * scale for value in score_line_y],
        }
        results.append((output, output_events, transform))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build score-measure tiles for x-axis event localisation.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    args = parser.parse_args()

    database = args.database.resolve()
    root = database / "score_event_locator"
    dataset_root = root / "dataset"
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    clean_dataset(dataset_root)

    source_records = read_jsonl(database / "manifests" / "sources.jsonl")
    source_splits = {
        record["id"]: SPLIT_OVERRIDES.get(record["id"], record["split"])
        for record in source_records
    }
    page_labels = sorted((database / "labels" / "pages" / "score_tab_rhythm").glob("*/*.json"))
    if not page_labels:
        raise FileNotFoundError("No score_tab rhythm page labels found. Build the rhythm dataset first.")

    records_by_split: dict[str, list[dict]] = defaultdict(list)
    sources_by_split: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for page_label_path in page_labels:
        page_label = json.loads(page_label_path.read_text(encoding="utf-8"))
        source_id = page_label["source_id"]
        split = source_splits[source_id]
        sources_by_split[split].add(source_id)
        page_path = database / page_label["image"]
        with Image.open(page_path) as opened:
            page = opened.convert("L")

        for measure in page_label["measures"]:
            base_id = f"{source_id}_p{int(page_label['page_index']):03d}_m{int(measure['measure_number']):03d}"
            tiles = tile_score_measure(page, measure["bbox"], measure["score_staff"]["line_y"], measure["events"])
            counts["measures"] += 1
            counts["unique_events"] += len(measure["events"])
            assigned = {event["event_id"]: 0 for event in measure["events"]}
            for tile_index, (image, events, transform) in enumerate(tiles):
                sample_id = f"{base_id}_w{tile_index:02d}"
                image_path = dataset_root / split / "images" / f"{sample_id}.png"
                label_path = dataset_root / split / "labels" / f"{sample_id}.json"
                image.save(image_path, format="PNG", optimize=True)
                for event in events:
                    assigned[event["event_id"]] += 1
                    counts[f"event_kind:{event['kind']}"] += 1
                label = {
                    "schema_version": "1.0",
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "split": split,
                    "page_index": int(page_label["page_index"]),
                    "measure_index": int(measure["measure_index"]),
                    "measure_number": int(measure["measure_number"]),
                    "tile_index": tile_index,
                    "source_page_image": page_label["image"],
                    "image_size": [INPUT_WIDTH, INPUT_HEIGHT],
                    "transform": transform,
                    "events": events,
                }
                label_path.write_text(json.dumps(label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                records_by_split[split].append(
                    {
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "split": split,
                        "image": image_path.relative_to(database).as_posix(),
                        "label": label_path.relative_to(database).as_posix(),
                        "event_count": len(events),
                    }
                )
                counts["tiles"] += 1
                counts["event_assignments"] += len(events)
            missing = [event_id for event_id, assignment_count in assigned.items() if assignment_count == 0]
            if missing:
                raise ValueError(f"Events were not assigned to a tile in {base_id}: {missing}")

    for split in ("train", "validation", "test"):
        (manifest_root / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for record in records_by_split[split]),
            encoding="utf-8",
        )

    summary = {
        "schema_version": "1.0",
        "task": "TuxGuitar score_tab x-axis event localisation",
        "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
        "tile_overlap": TILE_OVERLAP,
        "counts": dict(sorted(counts.items())),
        "splits": {
            split: {
                "sources": len(sources_by_split[split]),
                "tiles": len(records_by_split[split]),
                "event_assignments": sum(record["event_count"] for record in records_by_split[split]),
            }
            for split in ("train", "validation", "test")
        },
        "scope": "Ground-truth score-staff geometry creates training tiles; inference must recover geometry from pixels.",
    }
    (manifest_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
