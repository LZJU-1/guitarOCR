from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from guitarocr.paths import DATABASE_ROOT


PAGE_WIDTH = 550.0
PAGE_HEIGHT = 800.0
INPUT_WIDTH = 256
INPUT_HEIGHT = 192
# The original song split leaves tuplets and 32nd notes out of validation and
# puts almost all rhythm variety in training. These source-level swaps preserve
# 25/3/3 songs, keep every source isolated, and expose rare rhythm classes in
# held-out evaluation. Only two songs contain visible voice 1, so one remains
# in training and the other in test; a source-disjoint voice-1 validation score
# is impossible with the current corpus and is reported explicitly.
SPLIT_OVERRIDES = {
    "0d3a153380b36c1d": "train",
    "b02f9bfc2bfcc33c": "test",
    "2f48f92446035eba": "train",
    "520429005af0d7e6": "validation",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def clean_dataset(dataset_root: Path) -> None:
    expected = (dataset_root.parent / "dataset").resolve()
    if dataset_root.resolve() != expected:
        raise ValueError(f"Refusing to clean unexpected rhythm dataset path: {dataset_root}")
    for split in ("train", "validation", "test"):
        clean_files(dataset_root / split / "images", "*.png")
        clean_files(dataset_root / split / "labels", "*.json")


def crop_with_padding(
    image: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> Image.Image:
    width = max(1, right - left)
    height = max(1, bottom - top)
    output = Image.new("L", (width, height), 255)
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
        crop = image.crop(source_box).convert("L")
        output.paste(crop, (source_box[0] - left, source_box[1] - top))
    return output


def voice_target(voice: dict | None) -> dict:
    if voice is None or not voice["visible"]:
        return {
            "state": "empty",
            "duration_value": 0,
            "dot": "none",
            "division": "1:1",
            "direction": 0,
            "beam_count": 0,
            "note_count": 0,
            "tied_note_count": 0,
        }
    if voice["double_dotted"]:
        dot = "double"
    elif voice["dotted"]:
        dot = "single"
    else:
        dot = "none"
    notes = voice["notes"]
    return {
        "state": "rest" if voice["rest"] else "note",
        "duration_value": int(voice["duration_value"]),
        "dot": dot,
        "division": f"{voice['division_enters']}:{voice['division_times']}",
        "direction": int(voice["direction"]),
        "beam_count": int(voice["beam_count"]),
        "note_count": len(notes),
        "tied_note_count": sum(bool(note["tied"]) for note in notes),
    }


def short_voice_label(voice: dict | None) -> str:
    target = voice_target(voice)
    if target["state"] == "empty":
        return "-"
    prefix = "R" if target["state"] == "rest" else "N"
    dot = "." if target["dot"] == "single" else ".." if target["dot"] == "double" else ""
    division = "" if target["division"] == "1:1" else f"/{target['division']}"
    return f"{prefix}{target['duration_value']}{dot}{division}"


def build_event_crop(
    page_image: Image.Image,
    event_x: float,
    score_line_y: list[float],
) -> tuple[Image.Image, dict]:
    if len(score_line_y) != 5:
        raise ValueError(f"Expected five score lines, got {score_line_y}")
    spacing = sum(score_line_y[i + 1] - score_line_y[i] for i in range(4)) / 4.0
    if spacing <= 0:
        raise ValueError(f"Invalid score line spacing: {score_line_y}")

    # The vertical window includes stems in both voices; the horizontal window
    # includes neighbouring notes so that beams and ties remain interpretable.
    left = math.floor(event_x - 9.0 * spacing)
    right = math.ceil(event_x + 9.0 * spacing)
    top = math.floor(score_line_y[0] - 8.0 * spacing)
    bottom = math.ceil(score_line_y[-1] + 8.0 * spacing)
    crop = crop_with_padding(page_image, left, top, right, bottom)
    resized = crop.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.LANCZOS)
    transform = {
        "page_crop_xyxy": [left, top, right, bottom],
        "event_x_in_crop": event_x - left,
        "score_line_y_in_crop": [value - top for value in score_line_y],
        "original_crop_size": [right - left, bottom - top],
        "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
    }
    return resized, transform


def convert_source(
    database: Path,
    layout_path: Path,
    split: str,
    dataset_root: Path,
) -> tuple[list[dict], Counter[str]]:
    source = json.loads(layout_path.read_text(encoding="utf-8"))
    source_id = source["source_id"]
    image_root = database / "output" / "images" / "score_tab" / source_id
    label_root = database / "labels" / "pages" / "score_tab_rhythm" / source_id
    overlay_root = database / "output" / "annotation_overlays" / "score_tab_rhythm" / source_id
    clean_files(label_root, "page_*.json")
    clean_files(overlay_root, "page_*.png")

    image_output = dataset_root / split / "images"
    label_output = dataset_root / split / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    counts: Counter[str] = Counter()
    for page in source["pages"]:
        page_index = int(page["page_index"])
        image_path = image_root / f"page_{page_index:03d}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing score_tab page: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        sx, sy = width / PAGE_WIDTH, height / PAGE_HEIGHT
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        output_measures: list[dict] = []

        for measure in page["measures"]:
            measure_bbox = scale_bbox(measure["bbox"], sx, sy)
            score_bbox = scale_bbox(measure["score_staff"]["bbox"], sx, sy)
            tab_bbox = scale_bbox(measure["tab_staff"]["bbox"], sx, sy)
            score_line_y = [float(value) * sy for value in measure["score_staff"]["line_y"]]
            string_y = [float(value) * sy for value in measure["tab_staff"]["string_y"]]
            draw.rectangle(xyxy(measure_bbox), outline=(0, 140, 230), width=2)
            draw.rectangle(xyxy(score_bbox), outline=(0, 175, 70), width=2)
            draw.rectangle(xyxy(tab_bbox), outline=(120, 120, 120), width=1)
            draw.text((measure_bbox[0] + 3, measure_bbox[1] + 2),
                      f"m{measure['measure_number']}", fill=(0, 100, 210))

            output_events: list[dict] = []
            for event in measure["events"]:
                event_x = float(event["x"]) * sx
                voices_by_index = {int(voice["voice_index"]): voice for voice in event["voices"]}
                targets = [voice_target(voices_by_index.get(index)) for index in range(2)]
                sample_id = (
                    f"{source_id}_p{page_index:03d}_m{int(measure['measure_number']):03d}"
                    f"_b{int(event['beat_index']):03d}"
                )
                crop, transform = build_event_crop(image, event_x, score_line_y)
                crop_path = image_output / f"{sample_id}.png"
                crop.save(crop_path, format="PNG", optimize=True)

                label = {
                    "schema_version": "1.0",
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "split": split,
                    "page_index": page_index,
                    "measure_index": int(measure["measure_index"]),
                    "measure_number": int(measure["measure_number"]),
                    "time_signature": measure["time_signature"],
                    "event_id": event["event_id"],
                    "beat_index": int(event["beat_index"]),
                    "beat_start": int(event["beat_start"]),
                    "precise_start": int(event["precise_start"]),
                    "event_x_page": event_x,
                    "voices": targets,
                    "transform": transform,
                }
                label_path = label_output / f"{sample_id}.json"
                label_path.write_text(json.dumps(label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

                for voice_index, target in enumerate(targets):
                    counts[f"voice_{voice_index}_state:{target['state']}"] += 1
                    if target["state"] != "empty":
                        counts[f"duration:{target['duration_value']}"] += 1
                        counts[f"dot:{target['dot']}"] += 1
                        counts[f"division:{target['division']}"] += 1
                counts["events"] += 1

                spacing = (score_line_y[-1] - score_line_y[0]) / 4.0
                draw.line((event_x, score_line_y[0] - 7 * spacing,
                           event_x, score_line_y[-1] + 7 * spacing), fill=(255, 135, 0), width=1)
                label_text = f"{short_voice_label(voices_by_index.get(0))}|{short_voice_label(voices_by_index.get(1))}"
                draw.text((event_x + 2, score_line_y[0] - 7 * spacing), label_text, fill=(190, 70, 0))

                page_voices: list[dict] = []
                for voice in event["voices"]:
                    copied_voice = dict(voice)
                    copied_notes: list[dict] = []
                    for note in voice["notes"]:
                        copied_note = dict(note)
                        copied_note["center_y"] = float(note["center_y"]) * sy
                        copied_note["bbox"] = scale_bbox(note["bbox"], sx, sy)
                        copied_notes.append(copied_note)
                        color = (230, 30, 50) if int(voice["voice_index"]) == 0 else (190, 0, 210)
                        draw.rectangle(xyxy(copied_note["bbox"]), outline=color, width=2)
                    copied_voice["notes"] = copied_notes
                    page_voices.append(copied_voice)

                output_events.append({
                    "event_id": event["event_id"],
                    "beat_start": event["beat_start"],
                    "precise_start": event["precise_start"],
                    "beat_index": event["beat_index"],
                    "x": event_x,
                    "voices": page_voices,
                })
                records.append({
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "split": split,
                    "image": crop_path.relative_to(database).as_posix(),
                    "label": label_path.relative_to(database).as_posix(),
                    "page_index": page_index,
                    "measure_number": int(measure["measure_number"]),
                    "beat_index": int(event["beat_index"]),
                    "voice_states": [target["state"] for target in targets],
                })

            output_measures.append({
                "measure_index": measure["measure_index"],
                "measure_number": measure["measure_number"],
                "time_signature": measure["time_signature"],
                "bbox": measure_bbox,
                "score_staff": {"bbox": score_bbox, "line_y": score_line_y},
                "tab_staff": {"bbox": tab_bbox, "string_y": string_y},
                "events": output_events,
            })

        page_label = {
            "schema_version": "1.0",
            "source_id": source_id,
            "layout": "score_tab",
            "task": "rhythm_events",
            "page_index": page_index,
            "image": image_path.relative_to(database).as_posix(),
            "image_size": {"width": width, "height": height},
            "coordinate_space": {"name": "png_pixels_top_left", "scale_x": sx, "scale_y": sy},
            "measures": output_measures,
        }
        page_label_path = label_root / f"page_{page_index:03d}.json"
        page_label_path.write_text(json.dumps(page_label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        overlay.save(overlay_root / f"page_{page_index:03d}.png", format="PNG", optimize=True)

    return records, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build score_tab rhythm-event crops and page-coordinate QA labels."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    database = args.database.resolve()
    layout_root = database / "labels" / "layout" / "score_tab_rhythm"
    layout_paths = sorted(layout_root.glob("*.json"))
    if args.limit > 0:
        layout_paths = layout_paths[: args.limit]
    if not layout_paths:
        raise FileNotFoundError(f"No rhythm layout labels in {layout_root}")

    source_records = read_jsonl(database / "manifests" / "sources.jsonl")
    split_by_source = {
        record["id"]: SPLIT_OVERRIDES.get(record["id"], record["split"])
        for record in source_records
    }
    dataset_root = database / "rhythm_events" / "dataset"
    clean_dataset(dataset_root)

    records: list[dict] = []
    total_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for layout_path in layout_paths:
        source_id = layout_path.stem
        if source_id not in split_by_source:
            raise KeyError(f"No split for source {source_id}")
        split = split_by_source[source_id]
        source_events, counts = convert_source(database, layout_path, split, dataset_root)
        records.extend(source_events)
        total_counts.update(counts)
        split_counts[split].update(counts)
        split_counts[split]["sources"] += 1

    manifest_root = database / "rhythm_events" / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        split_records = [record for record in records if record["split"] == split]
        (manifest_root / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for record in split_records),
            encoding="utf-8",
        )

    summary = {
        "schema_version": "1.0",
        "task": "TuxGuitar score_tab event-level rhythm recognition",
        "source_count": len({record["source_id"] for record in records}),
        "event_crop_count": len(records),
        "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
        "counts": dict(sorted(total_counts.items())),
        "splits": {split: dict(sorted(counts.items())) for split, counts in split_counts.items()},
        "scope": "Ground-truth event centers; two voice targets; note/rest duration, dots and tuplet division.",
    }
    (manifest_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
