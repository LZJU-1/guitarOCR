from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import re
from pathlib import Path

from PIL import Image
import torch

from guitarocr.data.build_fret_token_dataset import (
    align_events_with_printed_tab_anchors,
    label_events,
    native_measure_numbers,
    read_jsonl,
    tnl_events,
)
from guitarocr.data.build_score_rhythm_dataset import build_event_crop
from guitarocr.data.build_tab_rhythm_dataset import build_tab_event_crop
from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.paths import DATABASE_ROOT, WEIGHTS_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import locate_page_events
from guitarocr.pipeline.infer_tuxguitar_tab_document import locate_tab_page_events
from guitarocr.pipeline.pdf_vector_tab import (
    apply_pdf_vector_tab_glyphs,
    extract_pdf_tab_glyphs,
)


PAGE_RE = re.compile(r"_page(\d+)\.png$")
MODE_TASK = {
    "tab": ("gp8_tab_rhythm_events", "tab_rhythm_events"),
    "both": ("gp8_score_rhythm_events", "rhythm_events"),
}


def clean_outputs(database: Path, modes: set[str]) -> None:
    for mode in modes:
        task_root, _ = MODE_TASK[mode]
        root = database / task_root
        for split in ("train", "validation"):
            image_root = root / "dataset" / split / "images"
            image_root.mkdir(parents=True, exist_ok=True)
            for path in image_root.glob("*.png"):
                path.unlink()
        (root / "manifests").mkdir(parents=True, exist_ok=True)


def append_tux_records(database: Path, mode: str, rows: dict[str, list[dict]]) -> None:
    _task_root, tux_root = MODE_TASK[mode]
    for split in ("train", "validation", "test"):
        path = database / tux_root / "manifests" / f"{split}.jsonl"
        if path.is_file():
            for record in read_jsonl(path):
                copied = dict(record)
                copied["renderer_domain"] = "tuxguitar"
                rows[split].append(copied)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-disjoint GP8 rhythm crops and mix existing TuxGuitar records."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--gp8-root", type=Path)
    parser.add_argument("--modes", default="tab,both")
    parser.add_argument(
        "--include-unanchored",
        action="store_true",
        help="also keep normalized-position fallback alignments (disabled by default)",
    )
    args = parser.parse_args()
    database = args.database.resolve()
    gp8_root = (args.gp8_root or database / "guitarpro8_multimode_v1").resolve()
    modes = {value.strip() for value in args.modes.split(",") if value.strip()}
    if not modes <= set(MODE_TASK):
        raise ValueError(f"Unsupported modes: {sorted(modes - set(MODE_TASK))}")
    clean_outputs(database, modes)
    rows: dict[str, dict[str, list[dict]]] = {
        mode: defaultdict(list) for mode in modes
    }
    counts: dict[str, Counter] = {mode: Counter() for mode in modes}
    for mode in modes:
        append_tux_records(database, mode, rows[mode])

    source_splits = {
        row["id"]: row["split"] for row in read_jsonl(database / "manifests" / "sources.jsonl")
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tab_checkpoint = torch.load(WEIGHTS_ROOT / "tab_event_locator.pt", map_location=device, weights_only=False)
    tab_locator = ScoreEventLocator().to(device).eval()
    tab_locator.load_state_dict(tab_checkpoint["model_state"])
    score_checkpoint = torch.load(WEIGHTS_ROOT / "score_event_locator.pt", map_location=device, weights_only=False)
    score_locator = ScoreEventLocator().to(device).eval()
    score_locator.load_state_dict(score_checkpoint["model_state"])
    semantic_cache = {}

    for coco_split, default_split in (("train", "train"), ("val", "validation")):
        coco = json.loads(
            (
                gp8_root / "layout_coco_source_disjoint" / "annotations"
                / f"instance_{coco_split}.json"
            ).read_text(encoding="utf-8")
        )
        for image_record in coco["images"]:
            file_name = image_record["file_name"]
            mode = file_name.split("_", 1)[0]
            if mode not in modes:
                continue
            source_group = str(image_record["source_group"])
            if source_group.startswith("real_"):
                source_id = source_group.rsplit("_", 1)[-1]
                original_split = source_splits.get(source_id)
                if original_split in {None, "test"}:
                    counts[mode]["excluded_test_or_unknown_pages"] += 1
                    continue
                split = "validation" if original_split == "validation" else default_split
                semantic_path = database / "v2" / "labels" / "songs" / f"{source_id}.json"
                key = f"label:{source_id}"
                semantic_cache.setdefault(key, label_events(semantic_path))
            else:
                split = default_split
                semantic_path = gp8_root / mode / "synth" / "tnl" / f"{source_group}.tnl"
                key = f"tnl:{source_group}"
                semantic_cache.setdefault(key, tnl_events(semantic_path))
            _string_count, semantic_measures = semantic_cache[key]
            match = PAGE_RE.search(file_name)
            if not match:
                counts[mode]["bad_page_names"] += 1
                continue
            page_index = int(match.group(1))
            measure_numbers = native_measure_numbers(
                gp8_root / mode / "layout" / f"{source_group}.layout.json",
                page_index,
            )
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
            source_pdf = gp8_root / mode / "pdf" / f"{source_group}.pdf"
            vector_summary = None
            if source_pdf.is_file():
                vector_summary = apply_pdf_vector_tab_glyphs(
                    systems,
                    extract_pdf_tab_glyphs(source_pdf, page_index + 1, page.size),
                )
                counts[mode]["vector_pdf_pages"] += 1
                counts[mode]["vector_pdf_tokens"] += int(vector_summary["tokens"])
            detected_measures = [measure for system in systems for measure in system["measures"]]
            if len(detected_measures) != len(measure_numbers):
                counts[mode]["geometry_mismatch_pages"] += 1
                continue
            task_root, _ = MODE_TASK[mode]
            image_output = database / task_root / "dataset" / split / "images"
            for measure, measure_number in zip(detected_measures, measure_numbers):
                if not (1 <= measure_number <= len(semantic_measures)):
                    counts[mode]["missing_measure_semantics"] += 1
                    continue
                system = next(item for item in systems if measure in item["measures"])
                semantic_events = semantic_measures[measure_number - 1]
                aligned, alignment_summary = align_events_with_printed_tab_anchors(
                    semantic_events,
                    measure["events"],
                    measure.get("vector_tab_events", []),
                )
                counts[mode][f"alignment:{alignment_summary['method']}"] += 1
                counts[mode]["matched_attack_anchors"] += int(
                    alignment_summary["matched_attack_anchors"]
                )
                if (
                    alignment_summary["method"] == "normalized_position_fallback"
                    and not args.include_unanchored
                ):
                    counts[mode]["excluded_unanchored_measures"] += 1
                    counts[mode]["excluded_unanchored_events"] += len(semantic_events)
                    continue
                for expected_index, detected_index, delta in aligned:
                    expected = semantic_events[expected_index]
                    event = measure["events"][detected_index]
                    if float(event.get("locator_confidence", 1.0)) < 0.18:
                        continue
                    if mode == "tab":
                        crop, transform = build_tab_event_crop(
                            page,
                            float(event["x"]),
                            [float(value) for value in system["tab_string_y"]],
                        )
                    else:
                        crop, transform = build_event_crop(
                            page,
                            float(event["x"]),
                            [float(value) for value in system["score_line_y"]],
                        )
                    sample_id = (
                        f"gp8_{mode}_{source_group}_p{page_index:03d}"
                        f"_m{measure_number:03d}_e{expected_index:03d}"
                    )
                    output_path = image_output / f"{sample_id}.png"
                    crop.save(output_path, format="PNG", compress_level=1)
                    voices = expected["voices"]
                    row = {
                        "sample_id": sample_id,
                        "source_id": source_group,
                        "split": split,
                        "renderer_domain": "guitarpro8",
                        "image": output_path.relative_to(database).as_posix(),
                        "voices": voices,
                        "tie_present": bool(expected.get("tied_notes")),
                        "tied_note_count": len(expected.get("tied_notes", [])),
                        "score_note_count": int(
                            expected.get("score_note_count", len(expected.get("notes", {})))
                        ),
                        "attacked_note_count": int(
                            expected.get("attacked_note_count", len(expected.get("printed_notes", {})))
                        ),
                        "tied_notes": expected.get("tied_notes", []),
                        "alignment_delta": delta,
                        "alignment_method": alignment_summary["method"],
                        "alignment_attack_anchors": int(
                            alignment_summary["matched_attack_anchors"]
                        ),
                        "transform": transform,
                    }
                    rows[mode][split].append(row)
                    counts[mode]["gp8_events"] += 1
                    for voice_index, voice in enumerate(voices):
                        counts[mode][f"voice_{voice_index}:{voice['state']}"] += 1
                        if voice["state"] != "empty":
                            counts[mode][f"duration:{voice['duration_value']}"] += 1
                            counts[mode][f"dot:{voice['dot']}"] += 1
                            counts[mode][f"division:{voice['division']}"] += 1
            counts[mode]["gp8_pages"] += 1

    summary = {"schema_version": "1.0", "device": str(device), "modes": {}}
    for mode in sorted(modes):
        task_root, _ = MODE_TASK[mode]
        manifest_root = database / task_root / "manifests"
        for split in ("train", "validation", "test"):
            (manifest_root / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows[mode][split]),
                encoding="utf-8",
            )
        summary["modes"][mode] = {
            "task_root": task_root,
            "splits": {split: len(rows[mode][split]) for split in ("train", "validation", "test")},
            "counts": dict(sorted(counts[mode].items())),
            "leakage_policy": "v2 test sources are excluded from GP8 train/validation crops",
        }
        (manifest_root / "summary.json").write_text(
            json.dumps(summary["modes"][mode], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
