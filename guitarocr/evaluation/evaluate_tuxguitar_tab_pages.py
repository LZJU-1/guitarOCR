from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from guitarocr.data.build_tab_detector_dataset import SPLIT_OVERRIDES, tile_measure
from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import DATABASE_ROOT
from guitarocr.pipeline.infer_tuxguitar_tab_page import (
    detect_tab_geometry,
    group_symbols,
    map_detection_to_page,
    nms_same_class,
)
from guitarocr.training.train_tab_detector import box_iou, decode_detections


def metrics(counter: Counter) -> dict[str, float | int]:
    tp, fp, fn = counter["tp"], counter["fp"], counter["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def match_symbols(predictions: list[dict], truths: list[dict]) -> tuple[int, int, int, int]:
    used: set[int] = set()
    true_positive = 0
    correct_string = 0
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        best_index = -1
        best_iou = 0.0
        for truth_index, truth in enumerate(truths):
            if truth_index in used or truth["class"] != prediction["class"]:
                continue
            iou = box_iou(prediction["bbox"], truth["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_index = truth_index
        if best_index >= 0 and best_iou >= 0.3:
            used.add(best_index)
            true_positive += 1
            correct_string += int(prediction["string"] == truths[best_index]["string"])
    return true_positive, len(predictions) - true_positive, len(truths) - true_positive, correct_string


def truth_events(measure: dict) -> list[dict]:
    # A Level-A TAB onset merges the two GP voices at the same beat position.
    # Voice separation is a later rhythm task and is not visible from fret
    # digits alone.
    grouped: dict[int, dict[tuple, list[dict]]] = {}
    for symbol in measure["symbols"]:
        note_key = (symbol["voice_index"], symbol["note_index"], symbol["string"], symbol["fret"])
        grouped.setdefault(symbol["beat_index"], {}).setdefault(note_key, []).append(symbol)
    events: list[dict] = []
    for notes in grouped.values():
        output_notes: list[tuple[int, int | str]] = []
        centers: list[float] = []
        for (_, _, string_number, fret), symbols in notes.items():
            value: int | str = "X" if any(item["class"] == "dead_x" for item in symbols) else fret
            output_notes.append((string_number, value))
            centers.extend(item["center"][0] for item in symbols)
        events.append({"x": sum(centers) / len(centers), "notes": tuple(sorted(output_notes))})
    return sorted(events, key=lambda event: event["x"])


def match_events(predictions: list[dict], truths: list[dict], tolerance: float) -> tuple[int, int, int]:
    used: set[int] = set()
    true_positive = 0
    for prediction in predictions:
        signature = tuple(sorted(
            ((note["string"], note["fret"]) for note in prediction["notes"]),
            key=lambda value: int(value[0]),
        ))
        options = [
            (abs(prediction["x"] - truth["x"]), truth_index)
            for truth_index, truth in enumerate(truths)
            if truth_index not in used and truth["notes"] == signature
            and abs(prediction["x"] - truth["x"]) <= tolerance
        ]
        if options:
            _, truth_index = min(options)
            used.add(truth_index)
            true_positive += 1
    return true_positive, len(predictions) - true_positive, len(truths) - true_positive


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pixel-only evaluation on TuxGuitar tab_only pages.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    database = args.database.resolve()
    source_records = [
        json.loads(line)
        for line in (database / "manifests" / "sources.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    detector_splits = {
        record["id"]: SPLIT_OVERRIDES.get(record["id"], record["split"])
        for record in source_records
    }
    source_ids = {
        source_id for source_id, split in detector_splits.items()
        if args.split == "all" or split == args.split
    }

    checkpoint = torch.load(
        database / "tab_detector" / "models" / "tab_symbol_detector.pt",
        map_location="cuda",
        weights_only=False,
    )
    classes = checkpoint["classes"]
    model = TabSymbolDetector(len(classes)).cuda().eval()
    model.load_state_dict(checkpoint["model_state"])

    symbol_counter = Counter()
    event_counter = Counter()
    correct_string = 0
    page_count = measure_count = 0
    for label_path in sorted((database / "labels" / "pages" / "tab_only").glob("*/*.json")):
        label = json.loads(label_path.read_text(encoding="utf-8"))
        if label["source_id"] not in source_ids:
            continue
        with Image.open(database / label["image"]) as opened:
            page_image = opened.convert("L")
        staffs = detect_tab_geometry(page_image)
        detected_measures = [measure for staff in staffs for measure in staff["measures"]]
        if len(detected_measures) != len(label["measures"]):
            raise RuntimeError(f"Geometry mismatch for {label_path}")

        tile_records: list[tuple[Image.Image, dict, dict, dict]] = []
        for staff in staffs:
            for measure in staff["measures"]:
                for tile_image, _, transform in tile_measure(page_image, measure["bbox"], []):
                    tile_records.append((tile_image, transform, staff, measure))
        for start in range(0, len(tile_records), 32):
            batch = tile_records[start : start + 32]
            arrays = [np.asarray(record[0], dtype=np.float32) / 255.0 for record in batch]
            tensor = torch.from_numpy(np.stack(arrays)).unsqueeze(1).cuda()
            tensor = (tensor - 0.5) / 0.5
            with torch.inference_mode():
                decoded = decode_detections(model(tensor), threshold=args.threshold)
            for detections, (_, transform, _, measure) in zip(decoded, batch):
                measure["symbols"].extend(map_detection_to_page(item, transform) for item in detections)

        flat_staffs = [staff for staff in staffs for _ in staff["measures"]]
        for measure_index, (measure, truth, staff) in enumerate(
            zip(detected_measures, label["measures"], flat_staffs)
        ):
            measure["symbols"] = nms_same_class(measure["symbols"])
            for symbol in measure["symbols"]:
                symbol["class"] = classes[symbol.pop("class_index")]
                x, y, width, height = symbol["bbox"]
                center_y = y + height / 2
                symbol["string"] = min(
                    range(len(staff["string_y"])),
                    key=lambda index: abs(staff["string_y"][index] - center_y),
                ) + 1
            symbol_counts = match_symbols(measure["symbols"], truth["symbols"])
            symbol_counter.update({"tp": symbol_counts[0], "fp": symbol_counts[1], "fn": symbol_counts[2]})
            correct_string += symbol_counts[3]

            predicted_events = group_symbols(measure["symbols"], staff["string_y"], staff["spacing"])
            expected_events = truth_events(truth)
            event_counts = match_events(predicted_events, expected_events, staff["spacing"] * 0.45)
            event_counter.update({"tp": event_counts[0], "fp": event_counts[1], "fn": event_counts[2]})
            measure_count += 1
        page_count += 1

    report = {
        "split": args.split,
        "threshold": args.threshold,
        "pages": page_count,
        "measures": measure_count,
        "symbol_detection_iou30": metrics(symbol_counter),
        "matched_symbol_string_accuracy": correct_string / max(1, symbol_counter["tp"]),
        "exact_event_f1": metrics(event_counter),
        "scope": "Only the input page pixels are used for staff, measure, symbol, string, fret, and event inference.",
    }
    output = database / "tab_detector" / "models" / f"page_end_to_end_{args.split}_metrics.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
