from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from guitarocr.data.fret_token_crop import crop_fret_token
from guitarocr.models.fret_token_model import CLASSES, INPUT_HEIGHT, INPUT_WIDTH
from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.paths import DATABASE_ROOT, PROJECT_ROOT, WEIGHTS_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import locate_page_events
from guitarocr.pipeline.infer_tuxguitar_tab_document import locate_tab_page_events


PAGE_RE = re.compile(r"_page(\d+)\.png$")
DURATION = {
    "w": Fraction(1, 1),
    "h": Fraction(1, 2),
    "q": Fraction(1, 4),
    "e": Fraction(1, 8),
    "s": Fraction(1, 16),
    "t": Fraction(1, 32),
    "sf": Fraction(1, 64),
}

TECHNIQUE_CLASSES = (
    "dead", "vibrato", "bend", "hammer", "slide", "ghost", "accent",
    "harmonic", "grace", "palm_mute", "staccato", "let_ring", "tapping",
    "pick_up", "pick_down",
)

TNL_EFFECTS = {
    "vibrato": {"v", "wv"},
    "hammer": {"h", "p"},
    "slide": {"sl", "ss", "si", "so"},
    "ghost": {"ghost"},
    "accent": {"acc", "hacc"},
    "harmonic": {"harm", "aharm", "pharm", "tharm"},
    "grace": {"grace"},
    "palm_mute": {"pm"},
    "staccato": {"stac"},
    "let_ring": {"let"},
    "tapping": {"tap"},
}


def empty_effects() -> dict[str, bool]:
    return {name: False for name in TECHNIQUE_CLASSES}


def tnl_beat_effects(beat: dict) -> dict[str, bool]:
    notes = beat.get("notes") or []
    note_fx = {str(value) for note in notes for value in (note.get("fx") or [])}
    effects = empty_effects()
    effects["dead"] = any(bool(note.get("dead")) for note in notes)
    effects["bend"] = any(bool(note.get("bend")) for note in notes)
    for name, values in TNL_EFFECTS.items():
        effects[name] = bool(note_fx & values)
    stroke = str(beat.get("stroke") or "").lower()
    effects["pick_up"] = stroke == "up"
    effects["pick_down"] = stroke == "down"
    return effects


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def tuplet_factor(value: int | None) -> Fraction:
    if not value or value <= 1:
        return Fraction(1, 1)
    normal = 2 if value == 3 else 4 if value <= 7 else 8
    return Fraction(normal, value)


def tnl_events(path: Path) -> tuple[int, list[list[dict]]]:
    datagen_root = PROJECT_ROOT / "guitar-hero-main"
    if str(datagen_root) not in sys.path:
        sys.path.insert(0, str(datagen_root))
    from datagen.tnl import parse_measure_body, read_tnl  # type: ignore

    document = read_tnl(path)
    string_count = int(document.header.get("strings", "6"))
    measures: list[list[dict]] = []
    for line in document.body_lines:
        onset = Fraction(0, 1)
        events = []
        for beat in parse_measure_body(line):
            notes = {}
            for note in beat.get("notes") or []:
                value: int | str = "X" if note.get("dead") else int(note["fret"])
                notes[int(note["str"])] = value
            duration = DURATION[str(beat.get("dur", "q"))]
            if beat.get("dot"):
                duration *= Fraction(3, 2)
            duration *= tuplet_factor(beat.get("tuplet"))
            duration_value = {
                "w": 1, "h": 2, "q": 4, "e": 8,
                "s": 16, "t": 32, "sf": 64,
            }[str(beat.get("dur", "q"))]
            if beat.get("tuplet"):
                tuplet = int(beat["tuplet"])
                normal = 2 if tuplet == 3 else 4 if tuplet <= 7 else 8
                division = f"{tuplet}:{normal}"
            else:
                division = "1:1"
            voice_zero = {
                "state": "rest" if beat.get("rest") else "note",
                "duration_value": duration_value,
                "dot": "single" if beat.get("dot") else "none",
                "division": division,
                "effects": tnl_beat_effects(beat),
            }
            voice_one = {
                "state": "empty", "duration_value": 0,
                "dot": "none", "division": "1:1",
                "effects": empty_effects(),
            }
            events.append({
                "onset": onset,
                "notes": notes,
                "printed_notes": dict(notes),
                "tied_notes": [],
                "score_note_count": len(notes),
                "attacked_note_count": len(notes),
                "voices": [voice_zero, voice_one],
            })
            onset += duration
        measures.append(events)
    return string_count, measures


def label_events(path: Path) -> tuple[int, list[list[dict]]]:
    label = json.loads(path.read_text(encoding="utf-8"))
    string_count = int(label["target_track"]["string_count"])
    measures = []
    for measure in label["measures"]:
        beats = []
        for beat in measure.get("beats", []):
            notes = {}
            printed_notes = {}
            tied_notes = []
            voices_by_index = {int(voice["index"]): voice for voice in beat.get("voices", [])}
            rhythm_voices = []
            for voice_index in range(2):
                voice = voices_by_index.get(voice_index)
                if voice is None:
                    rhythm_voices.append({
                        "state": "empty", "duration_value": 0,
                        "dot": "none", "division": "1:1",
                        "effects": empty_effects(),
                    })
                    continue
                duration = voice["duration"]
                notes_in_voice = voice.get("notes", [])
                effects = {
                    name: any(
                        bool((note.get("effects") or {}).get(name, False))
                        for note in notes_in_voice
                    )
                    for name in TECHNIQUE_CLASSES
                }
                rhythm_voices.append({
                    "state": "rest" if voice.get("rest") else "note",
                    "duration_value": int(duration["value"]),
                    "dot": (
                        "double" if duration.get("double_dotted") else
                        "single" if duration.get("dotted") else "none"
                    ),
                    "division": (
                        f"{int(duration.get('division_enters', 1))}:"
                        f"{int(duration.get('division_times', 1))}"
                    ),
                    "effects": effects,
                })
            for voice in beat.get("voices", []):
                for note in voice.get("notes", []):
                    effects = note.get("effects") or {}
                    string_index = int(note["string"])
                    value = "X" if effects.get("dead") else int(note["fret"])
                    notes[string_index] = value
                    if bool(note.get("tied", False)):
                        tied_notes.append({"string": string_index, "fret": value})
                    else:
                        printed_notes[string_index] = value
            beats.append({
                "onset": Fraction(int(beat["precise_start"]), 1),
                "notes": notes,
                "printed_notes": printed_notes,
                "tied_notes": tied_notes,
                "score_note_count": len(notes),
                "attacked_note_count": len(printed_notes),
                "voices": rhythm_voices,
            })
        if beats:
            origin = beats[0]["onset"]
            for beat in beats:
                beat["onset"] -= origin
        measures.append(beats)
    return string_count, measures


def normalized(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.5] * len(values)
    low, high = values[0], values[-1]
    if high <= low:
        return [index / (len(values) - 1) for index in range(len(values))]
    return [(value - low) / (high - low) for value in values]


def align_events(expected: list[dict], detected: list[dict]) -> list[tuple[int, int, float]]:
    if not expected or not detected:
        return []
    if len(expected) == len(detected):
        return [(index, index, 0.0) for index in range(len(expected))]
    expected_position = normalized([float(event["onset"]) for event in expected])
    detected_position = normalized([float(event["x"]) for event in detected])
    rows, columns = len(expected), len(detected)
    skip_expected = 0.16
    skip_detected = 0.11
    cost = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    action = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    cost[0, 0] = 0.0
    for row in range(rows + 1):
        for column in range(columns + 1):
            current = cost[row, column]
            if not np.isfinite(current):
                continue
            if row < rows and column < columns:
                value = current + abs(expected_position[row] - detected_position[column])
                if value < cost[row + 1, column + 1]:
                    cost[row + 1, column + 1] = value
                    action[row + 1, column + 1] = 1
            if row < rows and current + skip_expected < cost[row + 1, column]:
                cost[row + 1, column] = current + skip_expected
                action[row + 1, column] = 2
            if column < columns and current + skip_detected < cost[row, column + 1]:
                cost[row, column + 1] = current + skip_detected
                action[row, column + 1] = 3
    pairs = []
    row, column = rows, columns
    while row or column:
        selected = int(action[row, column])
        if selected == 1:
            delta = abs(expected_position[row - 1] - detected_position[column - 1])
            if delta <= 0.22:
                pairs.append((row - 1, column - 1, delta))
            row -= 1
            column -= 1
        elif selected == 2:
            row -= 1
        elif selected == 3:
            column -= 1
        else:
            break
    return list(reversed(pairs))


def _note_key(notes: dict) -> tuple[tuple[int, str], ...]:
    return tuple(sorted((int(string), str(fret).upper()) for string, fret in notes.items()))


def align_events_with_printed_tab_anchors(
    expected: list[dict],
    detected: list[dict],
    vector_events: list[dict],
) -> tuple[list[tuple[int, int, float]], dict]:
    """Align semantic beats to detected x positions using exact PDF TAB anchors.

    Attack fret tokens are an ordered fingerprint of a measure.  When the
    source semantics and vector-PDF fingerprints agree, their event indices are
    fixed first; only rests and unprinted tie continuations between those
    anchors need positional alignment. This avoids assigning correct rhythm
    labels to the wrong GP8 crop merely because event counts happen to match.
    """
    expected_anchors = [
        (index, _note_key(event.get("printed_notes", event.get("notes", {}))))
        for index, event in enumerate(expected)
        if event.get("printed_notes", event.get("notes", {}))
    ]
    vector_by_x = {
        round(float(event["x"]), 4): _note_key({
            int(note["string"]): note["fret"]
            for note in event.get("notes", [])
            if not bool(note.get("tie_in", False))
        })
        for event in vector_events
    }
    detected_anchors = [
        (index, vector_by_x.get(round(float(event["x"]), 4), ()))
        for index, event in enumerate(detected)
        if vector_by_x.get(round(float(event["x"]), 4), ())
    ]
    expected_fingerprint = [key for _index, key in expected_anchors]
    detected_fingerprint = [key for _index, key in detected_anchors]
    if not expected_anchors and len(expected) == len(detected):
        pairs = [(index, index, 0.0) for index in range(len(expected))]
        return pairs, {
            "method": "event_count_exact_no_attacks",
            "expected_attack_anchors": 0,
            "detected_attack_anchors": len(detected_anchors),
            "matched_attack_anchors": 0,
            "aligned_events": len(pairs),
        }
    if expected_fingerprint != detected_fingerprint:
        pairs = align_events(expected, detected)
        return pairs, {
            "method": "normalized_position_fallback",
            "expected_attack_anchors": len(expected_anchors),
            "detected_attack_anchors": len(detected_anchors),
            "matched_attack_anchors": 0,
            "aligned_events": len(pairs),
        }

    anchors = [
        (expected_index, detected_index)
        for (expected_index, _), (detected_index, _) in zip(
            expected_anchors, detected_anchors
        )
    ]
    pairs: list[tuple[int, int, float]] = [
        (expected_index, detected_index, 0.0)
        for expected_index, detected_index in anchors
    ]
    boundaries = [(-1, -1), *anchors, (len(expected), len(detected))]
    for (expected_left, detected_left), (expected_right, detected_right) in zip(
        boundaries, boundaries[1:]
    ):
        expected_indices = list(range(expected_left + 1, expected_right))
        detected_indices = list(range(detected_left + 1, detected_right))
        if not expected_indices or not detected_indices:
            continue
        if len(expected_indices) == len(detected_indices):
            pairs.extend(
                (expected_index, detected_index, 0.0)
                for expected_index, detected_index in zip(expected_indices, detected_indices)
            )
            continue
        local_pairs = align_events(
            [expected[index] for index in expected_indices],
            [detected[index] for index in detected_indices],
        )
        pairs.extend(
            (
                expected_indices[expected_index],
                detected_indices[detected_index],
                delta,
            )
            for expected_index, detected_index, delta in local_pairs
        )
    pairs.sort()
    return pairs, {
        "method": "pdf_vector_tab_anchors",
        "expected_attack_anchors": len(expected_anchors),
        "detected_attack_anchors": len(detected_anchors),
        "matched_attack_anchors": len(anchors),
        "aligned_events": len(pairs),
    }


def native_measure_numbers(layout_path: Path, page_index: int) -> list[int]:
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    rows = []
    for record in layout["records"]:
        if (
            record.get("kind") != "bar"
            or int(record.get("staff_type", -1)) != 2
            or int(record.get("page", 0)) != page_index + 1
        ):
            continue
        bbox = record.get("page_bbox_mm")
        if not bbox:
            continue
        rows.append((float(bbox[1]), float(bbox[0]), int(record["first_bar_index"]) + 1))
    rows.sort()
    return [measure for _y, _x, measure in rows]


def save_event_samples(
    page: Image.Image,
    detected: list[dict],
    expected: list[dict],
    string_y: list[float],
    spacing: float,
    split: str,
    sample_prefix: str,
    dataset_root: Path,
    records: dict[str, list[dict]],
    counts: Counter,
) -> None:
    class_to_index = {name: index for index, name in enumerate(CLASSES)}
    for expected_index, detected_index, delta in align_events(expected, detected):
        event = expected[expected_index]
        detected_event = detected[detected_index]
        if float(detected_event.get("locator_confidence", 1.0)) < 0.18:
            continue
        event_x = float(detected_event["x"])
        # Blank positions dominate six-string crops and made a first full
        # build exceed 250k tiny files. Keep every printed token plus one
        # rotating blank string per event; class weighting handles the rest.
        blank_strings = [
            string_index
            for string_index in range(1, len(string_y) + 1)
            if string_index not in event["notes"]
        ]
        kept_blank = (
            blank_strings[expected_index % len(blank_strings)] if blank_strings else None
        )
        selected_strings = set(event["notes"])
        if kept_blank is not None:
            selected_strings.add(kept_blank)
        for string_index, y in enumerate(string_y, start=1):
            if string_index not in selected_strings:
                continue
            raw_value = event["notes"].get(string_index, "blank")
            class_name = str(raw_value)
            if class_name not in class_to_index:
                counts["out_of_vocabulary"] += 1
                continue
            sample_id = f"{sample_prefix}_e{expected_index:03d}_s{string_index}"
            image_path = dataset_root / split / "images" / f"{sample_id}.png"
            crop_fret_token(page, event_x, float(y), spacing).save(
                image_path, format="PNG", compress_level=0
            )
            record = {
                "sample_id": sample_id,
                "split": split,
                "image": image_path.relative_to(dataset_root.parent.parent).as_posix(),
                "class": class_name,
                "class_index": class_to_index[class_name],
                "alignment_delta": delta,
            }
            records[split].append(record)
            counts[f"class:{class_name}"] += 1
            counts["samples"] += 1


def clean_dataset(dataset_root: Path) -> None:
    for split in ("train", "validation"):
        image_root = dataset_root / split / "images"
        image_root.mkdir(parents=True, exist_ok=True)
        for path in image_root.glob("*.png"):
            path.unlink()


def add_gp8_samples(
    database: Path,
    gp8_root: Path,
    dataset_root: Path,
    records: dict[str, list[dict]],
    counts: Counter,
    device: torch.device,
    modes: set[str],
    max_sources: int,
) -> None:
    source_splits = {
        row["id"]: row["split"] for row in read_jsonl(database / "manifests" / "sources.jsonl")
    }
    tab_checkpoint = torch.load(WEIGHTS_ROOT / "tab_event_locator.pt", map_location=device, weights_only=False)
    tab_locator = ScoreEventLocator().to(device).eval()
    tab_locator.load_state_dict(tab_checkpoint["model_state"])
    score_checkpoint = torch.load(WEIGHTS_ROOT / "score_event_locator.pt", map_location=device, weights_only=False)
    score_locator = ScoreEventLocator().to(device).eval()
    score_locator.load_state_dict(score_checkpoint["model_state"])
    semantic_cache: dict[str, tuple[int, list[list[dict]]]] = {}
    layout_cache: dict[tuple[str, str], Path] = {}
    source_seen: set[str] = set()

    for coco_split, output_split in (("train", "train"), ("val", "validation")):
        coco_path = gp8_root / "layout_coco_source_disjoint" / "annotations" / f"instance_{coco_split}.json"
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        for image_record in coco["images"]:
            file_name = image_record["file_name"]
            mode = file_name.split("_", 1)[0]
            if mode not in modes:
                continue
            source_group = str(image_record["source_group"])
            if max_sources and source_group not in source_seen and len(source_seen) >= max_sources:
                continue
            source_seen.add(source_group)
            if source_group.startswith("real_"):
                source_id = source_group.rsplit("_", 1)[-1]
                original_split = source_splits.get(source_id)
                if original_split == "test" or original_split is None:
                    counts["excluded_test_or_unknown_sources"] += 1
                    continue
                split = "validation" if original_split == "validation" else output_split
                semantic_path = database / "v2" / "labels" / "songs" / f"{source_id}.json"
                if not semantic_path.is_file():
                    counts["missing_semantics"] += 1
                    continue
                semantic_key = f"label:{source_id}"
                semantic_cache.setdefault(semantic_key, label_events(semantic_path))
            else:
                split = output_split
                semantic_path = gp8_root / mode / "synth" / "tnl" / f"{source_group}.tnl"
                semantic_key = f"tnl:{source_group}"
                semantic_cache.setdefault(semantic_key, tnl_events(semantic_path))
            string_count, semantic_measures = semantic_cache[semantic_key]
            page_match = PAGE_RE.search(file_name)
            if not page_match:
                counts["bad_page_name"] += 1
                continue
            page_index = int(page_match.group(1))
            layout_path = layout_cache.setdefault(
                (mode, source_group), gp8_root / mode / "layout" / f"{source_group}.layout.json"
            )
            measure_numbers = native_measure_numbers(layout_path, page_index)
            image_path = gp8_root / "layout_coco_source_disjoint" / "images" / file_name
            with Image.open(image_path) as opened:
                page = opened.convert("L")
            systems = (
                locate_tab_page_events(
                    page,
                    tab_locator,
                    device,
                    float(tab_checkpoint.get("detection_threshold", 0.3)),
                )
                if mode == "tab"
                else locate_page_events(
                    page,
                    score_locator,
                    device,
                    float(score_checkpoint.get("detection_threshold", 0.3)),
                )
            )
            detected_measures = [measure for system in systems for measure in system["measures"]]
            if len(detected_measures) != len(measure_numbers):
                counts["geometry_mismatch_pages"] += 1
                continue
            for measure, measure_number in zip(detected_measures, measure_numbers):
                if not (1 <= measure_number <= len(semantic_measures)):
                    counts["missing_measure_semantics"] += 1
                    continue
                system = next(item for item in systems if measure in item["measures"])
                save_event_samples(
                    page,
                    measure["events"],
                    semantic_measures[measure_number - 1],
                    [float(value) for value in system["tab_string_y"]],
                    float(system["tab_spacing"]),
                    split,
                    f"gp8_{mode}_{source_group}_p{page_index:03d}_m{measure_number:03d}",
                    dataset_root,
                    records,
                    counts,
                )
            counts["gp8_pages"] += 1


def add_tux_samples(
    database: Path,
    dataset_root: Path,
    records: dict[str, list[dict]],
    counts: Counter,
) -> None:
    source_splits = {
        row["id"]: row["split"] for row in read_jsonl(database / "manifests" / "sources.jsonl")
    }
    for task_root in ("tab_only", "score_tab_symbols"):
        for label_path in sorted((database / "labels" / "pages" / task_root).glob("*/*.json")):
            page_label = json.loads(label_path.read_text(encoding="utf-8"))
            source_id = page_label["source_id"]
            source_split = source_splits[source_id]
            if source_split == "test":
                continue
            split = "validation" if source_split == "validation" else "train"
            with Image.open(database / page_label["image"]) as opened:
                page = opened.convert("L")
            for measure in page_label["measures"]:
                staff = measure["tab_staff"]
                string_y = [float(value) for value in staff["string_y"]]
                spacing = float(np.median(np.diff(string_y)))
                grouped: dict[str, list[dict]] = defaultdict(list)
                for symbol in measure.get("symbols", []):
                    grouped[str(symbol["event_id"])].append(symbol)
                for event_index, symbols in enumerate(grouped.values()):
                    notes = {}
                    for symbol in symbols:
                        notes[int(symbol["string"])] = (
                            "X" if symbol["class"] == "dead_x" else int(symbol["fret"])
                        )
                    centers = [float(symbol["center"][0]) for symbol in symbols]
                    expected = [{"onset": Fraction(0, 1), "notes": notes}]
                    detected = [{"x": float(np.median(centers)), "locator_confidence": 1.0}]
                    save_event_samples(
                        page,
                        detected,
                        expected,
                        string_y,
                        spacing,
                        split,
                        (
                            f"tux_{task_root}_{source_id}_p{int(page_label['page_index']):03d}"
                            f"_m{int(measure['measure_number']):03d}_g{event_index:03d}"
                        ),
                        dataset_root,
                        records,
                        counts,
                    )
            counts["tux_pages"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mixed GP8/TuxGuitar event-conditioned fret-token crops."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--gp8-root", type=Path)
    parser.add_argument("--modes", default="tab,both")
    parser.add_argument("--max-gp8-sources", type=int, default=0)
    parser.add_argument("--skip-tux", action="store_true")
    args = parser.parse_args()
    database = args.database.resolve()
    gp8_root = (args.gp8_root or database / "guitarpro8_multimode_v1").resolve()
    root = database / "fret_token"
    dataset_root = root / "dataset"
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    clean_dataset(dataset_root)
    records: dict[str, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes = {value.strip() for value in args.modes.split(",") if value.strip()}
    add_gp8_samples(
        database,
        gp8_root,
        dataset_root,
        records,
        counts,
        device,
        modes,
        args.max_gp8_sources,
    )
    if not args.skip_tux:
        add_tux_samples(database, dataset_root, records, counts)
    for split in ("train", "validation"):
        (manifest_root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records[split]),
            encoding="utf-8",
        )
    summary = {
        "schema_version": "1.0",
        "classes": CLASSES,
        "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
        "device": str(device),
        "splits": {split: len(records[split]) for split in ("train", "validation")},
        "counts": dict(sorted(counts.items())),
        "leakage_policy": "All v2 test sources are excluded, including GP8 renderings.",
    }
    (manifest_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
