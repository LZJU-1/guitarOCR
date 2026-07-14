from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from PIL import Image

from guitarocr.data.build_score_rhythm_dataset import (
    TECHNIQUE_CLASSES,
    read_jsonl,
    voice_target,
)
from guitarocr.paths import DATABASE_ROOT


INPUT_WIDTH = 256
INPUT_HEIGHT = 192


def crop_with_padding(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = box
    output = Image.new("L", (right - left, bottom - top), 255)
    source = max(0, left), max(0, top), min(image.width, right), min(image.height, bottom)
    if source[2] > source[0] and source[3] > source[1]:
        output.paste(image.crop(source), (source[0] - left, source[1] - top))
    return output


def target(voice: dict | None) -> dict:
    if voice is None or not voice.get("visible", False):
        return {
            "state": "empty",
            "duration_value": 0,
            "dot": "none",
            "division": "1:1",
            "beam_count": 0,
            "note_count": 0,
            "tied_note_count": 0,
            "effects": {name: False for name in TECHNIQUE_CLASSES},
        }
    return {
        "state": "rest" if voice.get("rest", False) else "note",
        "duration_value": int(voice["duration_value"]),
        "dot": "double" if voice.get("double_dotted") else "single" if voice.get("dotted") else "none",
        "division": f"{voice['division_enters']}:{voice['division_times']}",
        "beam_count": 0,
        "note_count": 0,
        "tied_note_count": 0,
        "effects": {name: False for name in TECHNIQUE_CLASSES},
    }


def build_tab_event_crop(
    image: Image.Image,
    event_x: float,
    string_y: list[float],
) -> tuple[Image.Image, dict]:
    if len(string_y) < 4:
        raise ValueError(f"Expected at least four TAB strings, got {string_y}")
    spacing = sum(
        string_y[index + 1] - string_y[index]
        for index in range(len(string_y) - 1)
    ) / (len(string_y) - 1)
    if spacing <= 0:
        raise ValueError(f"Invalid TAB string spacing: {string_y}")
    box = (
        math.floor(event_x - 9.0 * spacing),
        math.floor(string_y[0] - 6.0 * spacing),
        math.ceil(event_x + 9.0 * spacing),
        math.ceil(string_y[-1] + 6.0 * spacing),
    )
    original = crop_with_padding(image, box)
    crop = original.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.LANCZOS)
    return crop, {
        "page_crop_xyxy": list(box),
        "original_crop_size": list(original.size),
        "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
        "tab_string_y": list(string_y),
        "tab_spacing": spacing,
    }


def load_score_semantics(database: Path, source_id: str) -> dict[tuple[int, int], dict]:
    """Use the GP-backed score_tab annotations as layout-independent semantics."""
    result: dict[tuple[int, int], dict] = {}
    root = database / "labels" / "pages" / "score_tab_rhythm" / source_id
    for path in sorted(root.glob("*.json")):
        page = json.loads(path.read_text(encoding="utf-8"))
        for measure in page.get("measures", []):
            measure_number = int(measure["measure_number"])
            for event in measure.get("events", []):
                result[(measure_number, int(event["beat_index"]))] = event
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-context rhythm crops from TuxGuitar tab_only pages")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    args = parser.parse_args()
    database = args.database.resolve()
    source_records = read_jsonl(database / "manifests" / "sources.jsonl")
    split_by_source = {record["id"]: record["split"] for record in source_records}
    page_labels = sorted((database / "labels" / "pages" / "tab_only").glob("*/*.json"))
    if not page_labels:
        raise FileNotFoundError("Build tab-only page annotations first")
    root = database / "tab_rhythm_events"; dataset = root / "dataset"; manifests = root / "manifests"
    for split in ("train", "validation", "test"):
        for leaf in ("images", "labels"):
            directory = dataset / split / leaf; directory.mkdir(parents=True, exist_ok=True)
            for path in directory.glob("*.png" if leaf == "images" else "*.json"): path.unlink()
    manifests.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict]] = defaultdict(list); counts = Counter()
    semantic_cache: dict[str, dict[tuple[int, int], dict]] = {}
    for page_label_path in page_labels:
        page = json.loads(page_label_path.read_text(encoding="utf-8"))
        source_id = page["source_id"]; split = split_by_source[source_id]
        if source_id not in semantic_cache:
            semantic_cache[source_id] = load_score_semantics(database, source_id)
        semantics = semantic_cache[source_id]
        with Image.open(database / page["image"]) as opened: image = opened.convert("L")
        for measure in page["measures"]:
            string_y = [float(value) for value in measure["tab_staff"]["string_y"]]
            spacing = sum(string_y[index + 1] - string_y[index] for index in range(len(string_y) - 1)) / (len(string_y) - 1)
            for event in measure.get("events", []):
                x = float(event["x"])
                crop, transform = build_tab_event_crop(image, x, string_y)
                sample_id = f"{source_id}_p{int(page['page_index']):03d}_m{int(measure['measure_number']):03d}_b{int(event['beat_index']):03d}"
                image_path = dataset / split / "images" / f"{sample_id}.png"
                label_path = dataset / split / "labels" / f"{sample_id}.json"
                semantic = semantics.get(
                    (int(measure["measure_number"]), int(event["beat_index"]))
                )
                if semantic is not None:
                    voices = {
                        int(value["voice_index"]): value
                        for value in semantic.get("voices", [])
                    }
                    targets = [voice_target(voices.get(index)) for index in range(2)]
                    pick_stroke = int(semantic.get("pick_stroke", 0))
                    targets[0]["effects"]["pick_up"] = pick_stroke == 1
                    targets[0]["effects"]["pick_down"] = pick_stroke == -1
                    counts["semantic_matches"] += 1
                else:
                    voices = {
                        int(value["voice_index"]): value
                        for value in event.get("voices", [])
                    }
                    targets = [target(voices.get(index)) for index in range(2)]
                    pick_stroke = 0
                    counts["semantic_fallbacks"] += 1
                crop.save(image_path, format="PNG", compress_level=1)
                label = {"schema_version": "1.0", "sample_id": sample_id, "source_id": source_id,
                         "split": split, "page_index": page["page_index"],
                         "measure_number": measure["measure_number"], "beat_index": event["beat_index"],
                         "event_x_page": x, "pick_stroke": pick_stroke,
                         "voices": targets, "transform": transform}
                label_path.write_text(json.dumps(label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                rows[split].append({"sample_id": sample_id, "source_id": source_id, "split": split,
                                    "image": image_path.relative_to(database).as_posix(),
                                    "label": label_path.relative_to(database).as_posix()})
                counts["events"] += 1
                for index, value in enumerate(targets):
                    counts[f"voice_{index}:{value['state']}"] += 1
                    if value["state"] != "empty":
                        counts[f"duration:{value['duration_value']}"] += 1
                        counts[f"dot:{value['dot']}"] += 1
                        counts[f"division:{value['division']}"] += 1
                    for name, enabled in value["effects"].items():
                        if enabled:
                            counts[f"technique:{name}"] += 1
    for split in ("train", "validation", "test"):
        (manifests / f"{split}.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows[split]), encoding="utf-8")
    summary = {"schema_version": "1.0", "task": "tab_only event rhythm and technique context", "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
               "counts": dict(sorted(counts.items())), "splits": {key: len(value) for key, value in rows.items()},
               "scope": "GP-backed event centers and semantics rendered in TuxGuitar tab_only layout."}
    (manifests / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
