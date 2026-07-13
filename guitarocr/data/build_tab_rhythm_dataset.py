from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from PIL import Image

from guitarocr.data.build_score_rhythm_dataset import read_jsonl
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
        return {"state": "empty", "duration_value": 0, "dot": "none", "division": "1:1"}
    return {
        "state": "rest" if voice.get("rest", False) else "note",
        "duration_value": int(voice["duration_value"]),
        "dot": "double" if voice.get("double_dotted") else "single" if voice.get("dotted") else "none",
        "division": f"{voice['division_enters']}:{voice['division_times']}",
    }


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
    for page_label_path in page_labels:
        page = json.loads(page_label_path.read_text(encoding="utf-8"))
        source_id = page["source_id"]; split = split_by_source[source_id]
        with Image.open(database / page["image"]) as opened: image = opened.convert("L")
        for measure in page["measures"]:
            string_y = [float(value) for value in measure["tab_staff"]["string_y"]]
            spacing = sum(string_y[index + 1] - string_y[index] for index in range(len(string_y) - 1)) / (len(string_y) - 1)
            for event in measure.get("events", []):
                x = float(event["x"])
                box = (math.floor(x - 9 * spacing), math.floor(string_y[0] - 6 * spacing),
                       math.ceil(x + 9 * spacing), math.ceil(string_y[-1] + 6 * spacing))
                crop = crop_with_padding(image, box).resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.LANCZOS)
                sample_id = f"{source_id}_p{int(page['page_index']):03d}_m{int(measure['measure_number']):03d}_b{int(event['beat_index']):03d}"
                image_path = dataset / split / "images" / f"{sample_id}.png"
                label_path = dataset / split / "labels" / f"{sample_id}.json"
                voices = {int(value["voice_index"]): value for value in event.get("voices", [])}
                targets = [target(voices.get(index)) for index in range(2)]
                crop.save(image_path, format="PNG", compress_level=1)
                label = {"schema_version": "1.0", "sample_id": sample_id, "source_id": source_id,
                         "split": split, "page_index": page["page_index"],
                         "measure_number": measure["measure_number"], "beat_index": event["beat_index"],
                         "event_x": x, "voices": targets, "crop_box": list(box)}
                label_path.write_text(json.dumps(label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                rows[split].append({"sample_id": sample_id, "source_id": source_id, "split": split,
                                    "image": image_path.relative_to(database).as_posix(),
                                    "label": label_path.relative_to(database).as_posix()})
                counts["events"] += 1
                for index, value in enumerate(targets): counts[f"voice_{index}:{value['state']}"] += 1
    for split in ("train", "validation", "test"):
        (manifests / f"{split}.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows[split]), encoding="utf-8")
    summary = {"schema_version": "1.0", "task": "tab_only event rhythm", "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
               "counts": dict(counts), "splits": {key: len(value) for key, value in rows.items()}}
    (manifests / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
