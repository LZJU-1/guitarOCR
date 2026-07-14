from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from guitarocr.models.rhythm_context_model import INPUT_HEIGHT
from guitarocr.paths import DATABASE_ROOT


Y_BINS = 48
DEFAULT_POSITIVE_VALIDATION_SOURCE = "43d4d168dd9a5afb"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def load_rhythm_records(database: Path, task_root: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for split in ("train", "validation", "test"):
        manifest = database / task_root / "manifests" / f"{split}.jsonl"
        for record in read_jsonl(manifest):
            sample_id = record["sample_id"]
            if sample_id in records:
                raise ValueError(f"Duplicate rhythm sample: {sample_id}")
            records[sample_id] = record
    return records


def choose_split(
    source_id: str,
    global_validation: set[str],
    global_test: set[str],
    positive_validation_source: str,
) -> str:
    if source_id in global_test:
        return "test"
    if source_id in global_validation or source_id == positive_validation_source:
        return "validation"
    return "train"


def target_y_in_crop(center_y: float, transform: dict) -> float:
    top = float(transform["page_crop_xyxy"][1])
    original_height = float(transform["original_crop_size"][1])
    return (float(center_y) - top) * INPUT_HEIGHT / original_height


def y_bin(value: float) -> int:
    return max(0, min(Y_BINS - 1, int(value / INPUT_HEIGHT * Y_BINS)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build event-centred tie-in labels from real TuxGuitar score_tab PDF renders."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument(
        "--layout",
        choices=("score_tab", "tab_only"),
        default="score_tab",
        help="Build score-staff or pure-TAB tie context labels.",
    )
    parser.add_argument(
        "--positive-validation-source",
        default=DEFAULT_POSITIVE_VALIDATION_SOURCE,
        help="A global-train source held out for positive tie validation because the global validation songs contain none.",
    )
    args = parser.parse_args()
    database = args.database.resolve()
    rhythm_task_root = "rhythm_events" if args.layout == "score_tab" else "tab_rhythm_events"
    rhythm_records = load_rhythm_records(database, rhythm_task_root)
    global_validation = {
        record["source_id"] for record in rhythm_records.values() if record["split"] == "validation"
    }
    global_test = {
        record["source_id"] for record in rhythm_records.values() if record["split"] == "test"
    }
    output_root = database / ("tie_events" if args.layout == "score_tab" else "tab_tie_events")
    manifest_root = output_root / "manifests"
    overlay_root = output_root / "overlays"
    manifests: dict[str, list[dict]] = {split: [] for split in ("train", "validation", "test")}
    counts: dict[str, Counter] = {split: Counter() for split in manifests}

    page_task_root = "score_tab_rhythm" if args.layout == "score_tab" else "tab_only"
    source_roots = sorted((database / "labels" / "pages" / page_task_root).iterdir())
    for source_root in source_roots:
        if not source_root.is_dir():
            continue
        source_id = source_root.name
        semantic_events: dict[tuple[int, int], dict] = {}
        if args.layout == "tab_only":
            semantic_root = database / "labels" / "pages" / "score_tab_rhythm" / source_id
            for semantic_path in sorted(semantic_root.glob("page_*.json")):
                semantic_page = json.loads(semantic_path.read_text(encoding="utf-8"))
                for semantic_measure in semantic_page["measures"]:
                    for semantic_event in semantic_measure["events"]:
                        semantic_events[(
                            int(semantic_measure["measure_number"]),
                            int(semantic_event["beat_index"]),
                        )] = semantic_event
        split = choose_split(
            source_id, global_validation, global_test, args.positive_validation_source
        )
        for page_label_path in sorted(source_root.glob("page_*.json")):
            page_label = json.loads(page_label_path.read_text(encoding="utf-8"))
            page_has_tie = any(
                note["tied"]
                for measure in page_label["measures"]
                for event in measure["events"]
                for voice in (
                    semantic_events.get(
                        (int(measure["measure_number"]), int(event["beat_index"])),
                        event,
                    ).get("voices", [])
                )
                for note in voice.get("notes", [])
            )
            overlay = None
            draw = None
            if page_has_tie:
                with Image.open(database / page_label["image"]) as opened:
                    overlay = opened.convert("RGB")
                draw = ImageDraw.Draw(overlay)

            for measure in page_label["measures"]:
                reference_y = (
                    measure["score_staff"]["line_y"]
                    if args.layout == "score_tab" else measure["tab_staff"]["string_y"]
                )
                spacing = (
                    float(reference_y[-1]) - float(reference_y[0])
                ) / max(1, len(reference_y) - 1)
                for event in measure["events"]:
                    semantic_event = semantic_events.get(
                        (int(measure["measure_number"]), int(event["beat_index"])),
                        event,
                    )
                    sample_id = (
                        f"{source_id}_p{int(page_label['page_index']):03d}"
                        f"_m{int(measure['measure_number']):03d}_b{int(event['beat_index']):03d}"
                    )
                    rhythm_record = rhythm_records.get(sample_id)
                    if rhythm_record is None:
                        raise KeyError(f"Missing rhythm crop for {sample_id}")
                    rhythm_label = json.loads(
                        (database / rhythm_record["label"]).read_text(encoding="utf-8")
                    )
                    tied_notes: list[dict] = []
                    score_note_count = sum(
                        len(voice.get("notes", []))
                        for voice in semantic_event.get("voices", [])
                    )
                    attacked_note_count = sum(
                        not note["tied"]
                        for voice in semantic_event.get("voices", [])
                        for note in voice.get("notes", [])
                    )
                    for voice in semantic_event.get("voices", []):
                        for note in voice.get("notes", []):
                            if not note["tied"]:
                                continue
                            center_y_page = (
                                float(note["center_y"])
                                if args.layout == "score_tab"
                                else float(measure["tab_staff"]["string_y"][int(note["string"]) - 1])
                            )
                            crop_y = target_y_in_crop(center_y_page, rhythm_label["transform"])
                            tied_notes.append(
                                {
                                    "voice": int(voice["voice_index"]),
                                    "string": int(note["string"]),
                                    "fret": int(note["fret"]),
                                    "center_y_page": center_y_page,
                                    "center_y_crop": crop_y,
                                    "y_bin": y_bin(crop_y),
                                }
                            )
                            if draw is not None:
                                if args.layout == "score_tab":
                                    x, y, width, height = (float(value) for value in note["bbox"])
                                else:
                                    x = float(event["x"]) - spacing * 0.4
                                    y = center_y_page - spacing * 0.4
                                    width = height = spacing * 0.8
                                draw.rectangle((x, y, x + width, y + height), outline=(220, 0, 190), width=3)
                    unique_bins = sorted({note["y_bin"] for note in tied_notes})
                    record = {
                        "schema_version": "1.0",
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "split": split,
                        "global_split": rhythm_record["split"],
                        "image": rhythm_record["image"],
                        "page_label": page_label_path.relative_to(database).as_posix(),
                        "page_index": int(page_label["page_index"]),
                        "measure_number": int(measure["measure_number"]),
                        "beat_index": int(event["beat_index"]),
                        "event_x_page": float(event["x"]),
                        "layout": args.layout,
                        "tie_present": bool(tied_notes),
                        "score_note_count": score_note_count,
                        "attacked_note_count": attacked_note_count,
                        "tied_note_count": len(tied_notes),
                        "tied_unique_y_count": len(unique_bins),
                        "tied_y_bins": unique_bins,
                        "tied_notes": tied_notes,
                    }
                    manifests[split].append(record)
                    counts[split]["events"] += 1
                    counts[split]["tie_events"] += int(bool(tied_notes))
                    counts[split]["tie_notes"] += len(tied_notes)
                    counts[split]["tie_unique_y"] += len(unique_bins)
                    if tied_notes and draw is not None:
                        event_x = float(event["x"])
                        top = float(reference_y[0]) - 6 * spacing
                        bottom = float(reference_y[-1]) + 6 * spacing
                        draw.line((event_x, top, event_x, bottom), fill=(255, 120, 0), width=2)
                        draw.text((event_x + 3, top), f"tie:{len(tied_notes)}", fill=(185, 0, 145))

            if overlay is not None:
                output_dir = overlay_root / source_id
                output_dir.mkdir(parents=True, exist_ok=True)
                overlay.save(output_dir / f"page_{int(page_label['page_index']):03d}.png", format="PNG", optimize=True)

    for split, records in manifests.items():
        records.sort(key=lambda record: record["sample_id"])
        write_jsonl(manifest_root / f"{split}.jsonl", records)
    if counts["validation"]["tie_events"] == 0:
        raise ValueError("Tie validation split contains no positive events")
    if counts["test"]["tie_events"] == 0:
        raise ValueError("Tie test split contains no positive events")
    summary = {
        "schema_version": "1.0",
        "y_bins": Y_BINS,
        "input_height": INPUT_HEIGHT,
        "positive_validation_source": args.positive_validation_source,
        "split_policy": (
            "The pretrained rhythm CNN test songs remain tie test, so neither pretraining nor tie training sees "
            "them. The named rhythm-train song is held out with rhythm-validation songs to provide enough positive "
            "validation events. All splits remain source-disjoint."
        ),
        "splits": {split: dict(value) for split, value in counts.items()},
        "scope": (
            f"Event-centred real {args.layout} PDF crops; tie-in presence, tied note count and target vertical bins. "
            "String/fret labels are evaluation-only and are not CNN inputs."
        ),
    }
    (manifest_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
