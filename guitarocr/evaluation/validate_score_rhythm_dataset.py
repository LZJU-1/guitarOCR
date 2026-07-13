from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from guitarocr.paths import DATABASE_ROOT


SPLIT_OVERRIDES = {
    "0d3a153380b36c1d": "train",
    "b02f9bfc2bfcc33c": "test",
    "2f48f92446035eba": "train",
    "520429005af0d7e6": "validation",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_voice(source_id: str, measure_number: int, beat_index: int, expected: dict, actual: dict) -> None:
    prefix = f"{source_id} m{measure_number} b{beat_index} v{expected['index']}"
    require(not actual["empty"], f"{prefix}: semantic voice unexpectedly empty")
    require(bool(actual["rest"]) == bool(expected["rest"]), f"{prefix}: rest mismatch")
    duration = expected["duration"]
    require(int(actual["duration_value"]) == int(duration["value"]), f"{prefix}: duration mismatch")
    require(bool(actual["dotted"]) == bool(duration["dotted"]), f"{prefix}: dotted mismatch")
    require(bool(actual["double_dotted"]) == bool(duration["double_dotted"]),
            f"{prefix}: double-dotted mismatch")
    require(int(actual["division_enters"]) == int(duration["division_enters"]),
            f"{prefix}: division enters mismatch")
    require(int(actual["division_times"]) == int(duration["division_times"]),
            f"{prefix}: division times mismatch")
    require(int(actual["precise_duration"]) == int(duration["precise_time"]),
            f"{prefix}: precise duration mismatch")
    expected_notes = expected["notes"]
    actual_notes = actual["notes"]
    require(len(actual_notes) == len(expected_notes), f"{prefix}: note count mismatch")
    for note_index, (expected_note, actual_note) in enumerate(zip(expected_notes, actual_notes)):
        note_prefix = f"{prefix} n{note_index}"
        for field in ("string", "fret", "tied"):
            require(actual_note[field] == expected_note[field], f"{note_prefix}: {field} mismatch")
        bbox = actual_note["bbox"]
        require(len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0, f"{note_prefix}: invalid bbox")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate score_tab rhythm semantics, coordinates and event crops.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    database = args.database.resolve()
    layout_paths = sorted((database / "labels" / "layout" / "score_tab_rhythm").glob("*.json"))
    if args.limit > 0:
        layout_paths = layout_paths[: args.limit]
    require(bool(layout_paths), "No score_tab rhythm layout labels")

    manifests: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        path = database / "rhythm_events" / "manifests" / f"{split}.jsonl"
        if path.is_file():
            for record in read_jsonl(path):
                require(record["sample_id"] not in manifests, f"Duplicate sample {record['sample_id']}")
                manifests[record["sample_id"]] = record

    source_splits = {
        row["id"]: SPLIT_OVERRIDES.get(row["id"], row["split"])
        for row in read_jsonl(database / "manifests" / "sources.jsonl")
    }
    counts: Counter[str] = Counter()
    seen_measures: dict[str, set[int]] = {}
    for layout_path in layout_paths:
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        source_id = layout["source_id"]
        song = json.loads((database / "labels" / "songs" / f"{source_id}.json").read_text(encoding="utf-8"))
        song_measures = {int(measure["index"]): measure for measure in song["measures"]}
        source_measure_indices: set[int] = set()

        image_pages = sorted((database / "output" / "images" / "score_tab" / source_id).glob("page_*.png"))
        require(len(layout["pages"]) == len(image_pages),
                f"{source_id}: layout pages {len(layout['pages'])} != image pages {len(image_pages)}")

        for page in layout["pages"]:
            page_index = int(page["page_index"])
            page_image_path = database / "output" / "images" / "score_tab" / source_id / f"page_{page_index:03d}.png"
            pixel_label_path = database / "labels" / "pages" / "score_tab_rhythm" / source_id / f"page_{page_index:03d}.json"
            overlay_path = database / "output" / "annotation_overlays" / "score_tab_rhythm" / source_id / f"page_{page_index:03d}.png"
            require(page_image_path.is_file(), f"Missing page {page_image_path}")
            require(pixel_label_path.is_file(), f"Missing pixel label {pixel_label_path}")
            require(overlay_path.is_file(), f"Missing overlay {overlay_path}")
            pixel_page = json.loads(pixel_label_path.read_text(encoding="utf-8"))
            require(len(pixel_page["measures"]) == len(page["measures"]),
                    f"{source_id} p{page_index}: pixel measure count mismatch")
            counts["pages"] += 1

            for measure in page["measures"]:
                measure_index = int(measure["measure_index"])
                require(measure_index not in source_measure_indices,
                        f"{source_id}: duplicate measure {measure_index}")
                source_measure_indices.add(measure_index)
                expected_measure = song_measures[measure_index]
                require(int(measure["measure_number"]) == int(expected_measure["number"]),
                        f"{source_id} m{measure_index}: measure number mismatch")
                expected_signature = expected_measure["time_signature"]
                require(measure["time_signature"] == [expected_signature["numerator"], expected_signature["denominator"]],
                        f"{source_id} m{measure_index}: time signature mismatch")
                require(len(measure["score_staff"]["line_y"]) == 5,
                        f"{source_id} m{measure_index}: expected five score lines")

                expected_beats = expected_measure["beats"]
                require(len(measure["events"]) == len(expected_beats),
                        f"{source_id} m{measure_index}: event count mismatch")
                for beat_index, (event, expected_beat) in enumerate(zip(measure["events"], expected_beats)):
                    prefix = f"{source_id} m{measure_index} b{beat_index}"
                    require(int(event["beat_index"]) == beat_index, f"{prefix}: beat index mismatch")
                    require(int(event["beat_start"]) == int(expected_beat["start"]), f"{prefix}: start mismatch")
                    require(int(event["precise_start"]) == int(expected_beat["precise_start"]),
                            f"{prefix}: precise start mismatch")
                    measure_bbox = measure["bbox"]
                    require(measure_bbox[0] <= event["x"] <= measure_bbox[0] + measure_bbox[2],
                            f"{prefix}: event x outside measure")
                    actual_voices = {int(voice["voice_index"]): voice for voice in event["voices"]}
                    for expected_voice in expected_beat["voices"]:
                        voice_index = int(expected_voice["index"])
                        require(voice_index in actual_voices, f"{prefix}: missing voice {voice_index}")
                        validate_voice(source_id, int(measure["measure_number"]), beat_index,
                                       expected_voice, actual_voices[voice_index])

                    sample_id = (
                        f"{source_id}_p{page_index:03d}_m{int(measure['measure_number']):03d}_b{beat_index:03d}"
                    )
                    require(sample_id in manifests, f"Missing event manifest sample {sample_id}")
                    record = manifests[sample_id]
                    require(record["source_id"] == source_id, f"{sample_id}: source mismatch")
                    require(record["split"] == source_splits[source_id], f"{sample_id}: split mismatch")
                    crop_path = database / record["image"]
                    crop_label_path = database / record["label"]
                    require(crop_path.is_file(), f"Missing crop {crop_path}")
                    require(crop_label_path.is_file(), f"Missing crop label {crop_label_path}")
                    with Image.open(crop_path) as crop:
                        require(crop.size == (256, 192), f"{sample_id}: crop size {crop.size}")
                    crop_label = json.loads(crop_label_path.read_text(encoding="utf-8"))
                    require(crop_label["sample_id"] == sample_id, f"{sample_id}: crop label id mismatch")
                    require(len(crop_label["voices"]) == 2, f"{sample_id}: expected two target voices")
                    counts["events"] += 1
                    counts["visible_voices"] += sum(bool(voice["visible"]) for voice in event["voices"])

                counts["measures"] += 1

        require(source_measure_indices == set(song_measures),
                f"{source_id}: layout measure coverage mismatch")
        seen_measures[source_id] = source_measure_indices
        counts["sources"] += 1

    result = {"status": "passed", **dict(counts)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
