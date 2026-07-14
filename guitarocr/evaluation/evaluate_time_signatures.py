from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from guitarocr.data.build_score_rhythm_dataset import build_event_crop
from guitarocr.data.build_tab_rhythm_dataset import build_tab_event_crop
from guitarocr.models.rhythm_context_model import RhythmContextCNN
from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.paths import DATABASE_ROOT, WEIGHTS_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import classify_rhythm, locate_page_events
from guitarocr.pipeline.infer_tuxguitar_tab_document import locate_tab_page_events
from guitarocr.pipeline.measure_rhythm_constraints import refine_time_signatures_from_rhythm
from guitarocr.pipeline.time_signature_recognizer import load_atomic_model, propagate_time_signatures


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate printed time-signature OCR and propagation.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--layout", choices=("score_tab", "tab_only"), default="score_tab")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--locator-model", type=Path)
    args = parser.parse_args()
    database = args.database.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_root = "tab_event_locator" if args.layout == "tab_only" else "score_event_locator"
    label_layout = "tab_only" if args.layout == "tab_only" else "score_tab_rhythm"
    locator_path = args.locator_model
    if locator_path is None:
        public_name = "tab_event_locator.pt" if args.layout == "tab_only" else "score_event_locator.pt"
        public_path = WEIGHTS_ROOT / public_name
        trained_path = database / task_root / "models" / public_name
        locator_path = public_path if public_path.exists() else trained_path
    threshold = args.threshold if args.threshold is not None else (0.20 if args.layout == "tab_only" else 0.45)
    locator_checkpoint = torch.load(locator_path, map_location=device, weights_only=False)
    locator = ScoreEventLocator().to(device)
    locator.load_state_dict(locator_checkpoint["model_state"])
    locator.eval()
    trained_atomic = database / "symbol_cnn" / "models" / "atomic_symbol_cnn.pt"
    public_atomic = WEIGHTS_ROOT / "atomic_symbol_cnn.pt"
    atomic_path = public_atomic if public_atomic.exists() else trained_atomic
    atomic, classes = load_atomic_model(atomic_path, device)
    rhythm_task = "tab_rhythm_events" if args.layout == "tab_only" else "rhythm_events"
    rhythm_public_name = (
        "tab_rhythm_context_cnn.pt" if args.layout == "tab_only" else "rhythm_context_cnn.pt"
    )
    trained_rhythm = database / rhythm_task / "models" / "rhythm_context_cnn.pt"
    public_rhythm = WEIGHTS_ROOT / rhythm_public_name
    rhythm_path = public_rhythm if public_rhythm.exists() else trained_rhythm
    rhythm_checkpoint = torch.load(rhythm_path, map_location=device, weights_only=False)
    rhythm = RhythmContextCNN().to(device)
    rhythm.load_state_dict(rhythm_checkpoint["model_state"])
    rhythm.eval()

    if args.split == "all":
        source_ids = {path.stem for path in (database / "labels" / "songs").glob("*.json")}
    else:
        source_ids = {record["source_id"] for record in read_jsonl(
            database / task_root / "manifests" / f"{args.split}.jsonl"
        )}
    printed_tp = printed_fp = printed_fn = printed_value_correct = 0
    propagated_correct = propagated_total = 0
    refined_correct = refined_total = 0
    refinement_totals = {
        "capacity_supported": 0,
        "visually_confirmed_changes": 0,
        "sequence_confirmed_changes": 0,
    }
    errors: list[dict] = []
    for source_id in sorted(source_ids):
        song = json.loads((database / "labels" / "songs" / f"{source_id}.json").read_text(encoding="utf-8"))
        song_measures = {int(measure["number"]): measure for measure in song["measures"]}
        # A new document always starts with a printed signature and must not
        # inherit the previous document's recognition state.
        previous_truth: tuple[int, int] | None = None
        carried: tuple[int, int] | None = None
        refinement_measures: list[dict] = []
        refinement_truth: list[tuple[int, int]] = []
        for page_path in sorted((database / "labels" / "pages" / label_layout / source_id).glob("page_*.json")):
            page_label = json.loads(page_path.read_text(encoding="utf-8"))
            with Image.open(database / page_label["image"]) as opened:
                image = opened.convert("L")
            if args.layout == "tab_only":
                systems = locate_tab_page_events(
                    image, locator, device,
                    threshold=float(locator_checkpoint.get("detection_threshold", 0.25)),
                )
            else:
                systems = locate_page_events(image, locator, device, threshold=0.3)
            carried = propagate_time_signatures(
                image, systems, atomic, classes, device, initial=carried, threshold=threshold
            )
            classify_rhythm(
                image, systems, rhythm, device, crop_root=None,
                crop_builder=build_tab_event_crop if args.layout == "tab_only" else build_event_crop,
                reference_line_key="tab_string_y" if args.layout == "tab_only" else "score_line_y",
            )
            predicted_measures = [measure for system in systems for measure in system["measures"]]
            if len(predicted_measures) != len(page_label["measures"]):
                raise ValueError(f"Measure mismatch on {page_path}")
            for predicted, truth_measure in zip(predicted_measures, page_label["measures"]):
                number = int(truth_measure["measure_number"])
                semantic = song_measures[number]["time_signature"]
                truth = (int(semantic["numerator"]), int(semantic["denominator"]))
                should_print = previous_truth is None or truth != previous_truth
                detection = predicted["printed_time_signature"]
                if detection is not None:
                    if should_print:
                        printed_tp += 1
                        detected_value = (int(detection["numerator"]), int(detection["denominator"]))
                        printed_value_correct += int(detected_value == truth)
                        if detected_value != truth:
                            errors.append({"source": source_id, "measure": number, "type": "wrong_value", "truth": truth, "prediction": detected_value})
                    else:
                        printed_fp += 1
                        errors.append({"source": source_id, "measure": number, "type": "false_presence", "truth": truth})
                elif should_print:
                    printed_fn += 1
                    errors.append({"source": source_id, "measure": number, "type": "missed_print", "truth": truth})
                propagated_total += 1
                predicted_value = tuple(predicted["time_signature"]) if predicted["time_signature"] is not None else None
                propagated_correct += int(predicted_value == truth)
                if predicted_value != truth:
                    errors.append({"source": source_id, "measure": number, "type": "propagation", "truth": truth, "prediction": predicted_value})
                refinement_measures.append({
                    "number": number,
                    "bbox": predicted.get("bbox"),
                    "tab_spacing": next(
                        float(system.get("tab_spacing", system.get("score_spacing", 0.0)))
                        for system in systems
                        if any(predicted is item for item in system["measures"])
                    ),
                    "time_signature": list(predicted_value) if predicted_value is not None else None,
                    "time_signature_source": predicted.get("time_signature_source"),
                    "printed_time_signature": detection,
                    "printed_time_signature_candidate": predicted.get("printed_time_signature_candidate"),
                    "printed_time_signature_shape_hint": predicted.get(
                        "printed_time_signature_shape_hint"
                    ),
                    "events": [
                        {
                            "voices": [
                                {
                                    "voice": voice_index,
                                    "state": voice["state"]["value"],
                                    "duration_value": int(voice["duration"]["value"]),
                                    "dot": voice["dot"]["value"],
                                    "division": voice["division"]["value"],
                                }
                                for voice_index, voice in enumerate(event["voices"])
                            ]
                        }
                        for event in predicted.get("events", [])
                    ],
                })
                refinement_truth.append(truth)
                previous_truth = truth
        refinement_ir = {"tracks": [{"measures": refinement_measures}]}
        refinement = refine_time_signatures_from_rhythm(refinement_ir)
        for key in refinement_totals:
            refinement_totals[key] += int(refinement[key])
        for measure, truth in zip(refinement_measures, refinement_truth):
            predicted = tuple(measure["time_signature"]) if measure.get("time_signature") else None
            refined_total += 1
            refined_correct += int(predicted == truth)
            if predicted != truth:
                errors.append({
                    "source": source_id,
                    "measure": int(measure["number"]),
                    "type": "rhythm_refined",
                    "truth": truth,
                    "prediction": predicted,
                })

    precision = printed_tp / max(1, printed_tp + printed_fp)
    recall = printed_tp / max(1, printed_tp + printed_fn)
    report = {
        "split": args.split,
        "layout": args.layout,
        "threshold": threshold,
        "printed_presence": {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "true_positive": printed_tp,
            "false_positive": printed_fp,
            "false_negative": printed_fn,
            "value_accuracy_on_detected_truth_prints": printed_value_correct / max(1, printed_tp),
        },
        "propagated_measure_accuracy": propagated_correct / max(1, propagated_total),
        "rhythm_refined_measure_accuracy": refined_correct / max(1, refined_total),
        "rhythm_refinement": refinement_totals,
        "measure_support": propagated_total,
        "errors": errors,
        "scope": "Pixel-only printed signature OCR with per-document carry-forward; labels are evaluation-only.",
    }
    output = database / task_root / "models" / f"time_signature_{args.split}_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
