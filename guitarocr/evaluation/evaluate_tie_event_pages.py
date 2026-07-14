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
from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import DATABASE_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import classify_rhythm, locate_page_events
from guitarocr.pipeline.infer_tuxguitar_tab_document import locate_tab_page_events
from guitarocr.pipeline.score_tab_fingering import build_score_ir, detect_tab_fingering, resolve_unambiguous_ties
from guitarocr.pipeline.tie_inference import classify_ties, load_tie_model


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(tp: int, fp: int, fn: int, negatives: int) -> dict:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": negatives,
        "positive_support": tp + fn,
    }


def truth_tied_notes(event: dict) -> list[dict]:
    return [
        note
        for voice in event.get("voices", [])
        for note in voice.get("notes", [])
        if note.get("tied")
    ]


def match_y(predicted: list[float], truth: list[float], tolerance: float) -> int:
    unused = set(range(len(predicted)))
    matched = 0
    for target in truth:
        candidates = [index for index in unused if abs(predicted[index] - target) <= tolerance]
        if candidates:
            selected = min(candidates, key=lambda index: abs(predicted[index] - target))
            unused.remove(selected)
            matched += 1
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pixel-only tie candidates after event location and Score/TAB consistency."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--event-threshold", type=float, default=0.3)
    parser.add_argument("--tab-threshold", type=float, default=0.3)
    parser.add_argument("--tie-threshold", type=float)
    parser.add_argument(
        "--layout", choices=("score_tab", "tab_only"), default="score_tab"
    )
    args = parser.parse_args()
    database = args.database.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    locator_root = "score_event_locator" if args.layout == "score_tab" else "tab_event_locator"
    locator_name = "score_event_locator.pt" if args.layout == "score_tab" else "tab_event_locator.pt"
    rhythm_root = "rhythm_events" if args.layout == "score_tab" else "tab_rhythm_events"
    tie_root = "tie_events" if args.layout == "score_tab" else "tab_tie_events"
    page_label_root = "score_tab_rhythm" if args.layout == "score_tab" else "tab_only"
    locator_checkpoint = torch.load(
        database / locator_root / "models" / locator_name,
        map_location=device, weights_only=False,
    )
    locator = ScoreEventLocator().to(device)
    locator.load_state_dict(locator_checkpoint["model_state"])
    locator.eval()
    rhythm_checkpoint = torch.load(
        database / rhythm_root / "models" / "rhythm_context_cnn.pt",
        map_location=device, weights_only=False,
    )
    rhythm = RhythmContextCNN().to(device)
    rhythm.load_state_dict(rhythm_checkpoint["model_state"])
    rhythm.eval()
    tab_checkpoint = torch.load(
        database / "tab_detector" / "models" / "tab_symbol_detector.pt",
        map_location=device, weights_only=False,
    )
    tab_classes = tab_checkpoint["classes"]
    tab = TabSymbolDetector(len(tab_classes)).to(device)
    tab.load_state_dict(tab_checkpoint["model_state"])
    tab.eval()
    tie, tie_threshold = load_tie_model(
        database / tie_root / "models" / (
            "tie_context_cnn.pt" if args.layout == "score_tab" else "tab_tie_context_cnn.pt"
        ), device
    )
    if args.tie_threshold is not None:
        tie_threshold = args.tie_threshold

    source_ids = sorted({
        record["source_id"]
        for record in read_jsonl(database / tie_root / "manifests" / f"{args.split}.jsonl")
    })
    visual_tp = visual_fp = visual_fn = visual_tn = 0
    candidate_tp = candidate_fp = candidate_fn = candidate_tn = 0
    count_exact = count_support = 0
    y_matched = y_predicted = y_truth = 0
    truth_tie_events = truth_tie_notes_count = 0
    resolved_events = resolved_event_exact = resolved_notes = resolved_notes_correct = 0
    page_count = measure_count = truth_event_count = matched_event_count = 0
    errors: list[dict] = []

    for source_id in source_ids:
        document_measures: list[dict] = []
        truth_by_predicted_key: dict[tuple[int, int], dict] = {}
        semantic_events: dict[tuple[int, int], dict] = {}
        if args.layout == "tab_only":
            for semantic_path in sorted(
                (database / "labels" / "pages" / "score_tab_rhythm" / source_id).glob("page_*.json")
            ):
                semantic_page = json.loads(semantic_path.read_text(encoding="utf-8"))
                for measure in semantic_page["measures"]:
                    for event in measure["events"]:
                        semantic_events[(int(measure["measure_number"]), int(event["beat_index"]))] = event
        for page_path in sorted(
            (database / "labels" / "pages" / page_label_root / source_id).glob("page_*.json")
        ):
            page_count += 1
            page_label = json.loads(page_path.read_text(encoding="utf-8"))
            with Image.open(database / page_label["image"]) as opened:
                page = opened.convert("L")
            systems = (
                locate_page_events(page, locator, device, args.event_threshold)
                if args.layout == "score_tab"
                else locate_tab_page_events(page, locator, device, args.event_threshold)
            )
            classify_rhythm(
                page, systems, rhythm, device, crop_root=None,
                crop_builder=build_tab_event_crop if args.layout == "tab_only" else build_event_crop,
                reference_line_key="tab_string_y" if args.layout == "tab_only" else "score_line_y",
            )
            detect_tab_fingering(page, systems, tab, tab_classes, device, threshold=args.tab_threshold)
            classify_ties(
                page, systems, tie, device, tie_threshold,
                crop_builder=build_tab_event_crop if args.layout == "tab_only" else build_event_crop,
                reference_line_key="tab_string_y" if args.layout == "tab_only" else "score_line_y",
            )
            truth_measures = page_label["measures"]
            offset = int(truth_measures[0]["measure_number"])
            score_ir = build_score_ir(systems, measure_number_offset=offset)
            predicted_measures = score_ir["tracks"][0]["measures"]
            if len(predicted_measures) != len(truth_measures):
                raise ValueError(f"Measure mismatch on {page_path}")
            document_measures.extend(predicted_measures)
            measure_count += len(truth_measures)

            for predicted_measure, truth_measure in zip(predicted_measures, truth_measures):
                predictions = predicted_measure["events"]
                truths = truth_measure["events"]
                truth_event_count += len(truths)
                reference_y = (
                    truth_measure["score_staff"]["line_y"]
                    if args.layout == "score_tab"
                    else truth_measure["tab_staff"]["string_y"]
                )
                spacing = (reference_y[-1] - reference_y[0]) / (len(reference_y) - 1)
                unmatched = set(range(len(predictions)))
                matched_truth: set[int] = set()
                for truth_index, truth_event in enumerate(truths):
                    semantic_event = (
                        truth_event
                        if args.layout == "score_tab"
                        else semantic_events.get(
                            (int(truth_measure["measure_number"]), int(truth_event["beat_index"])),
                            {},
                        )
                    )
                    tied_notes = truth_tied_notes(semantic_event)
                    truth_positive = bool(tied_notes)
                    truth_tie_events += int(truth_positive)
                    truth_tie_notes_count += len(tied_notes)
                    if not unmatched:
                        if truth_positive:
                            visual_fn += 1
                            candidate_fn += 1
                        else:
                            visual_tn += 1
                            candidate_tn += 1
                        continue
                    prediction_index = min(
                        unmatched,
                        key=lambda index: abs(predictions[index]["x"] - float(truth_event["x"])),
                    )
                    if abs(predictions[prediction_index]["x"] - float(truth_event["x"])) > spacing * 0.5:
                        if truth_positive:
                            visual_fn += 1
                            candidate_fn += 1
                        else:
                            visual_tn += 1
                            candidate_tn += 1
                        continue
                    unmatched.remove(prediction_index)
                    matched_truth.add(truth_index)
                    matched_event_count += 1
                    prediction = predictions[prediction_index]
                    truth_by_predicted_key[(int(predicted_measure["number"]), int(prediction["order"]))] = semantic_event
                    relation = prediction["tie_relation"]
                    visual_positive = bool(relation["visual_positive"])
                    candidate = bool(relation["candidate"])
                    visual_tp += int(truth_positive and visual_positive)
                    visual_fp += int(not truth_positive and visual_positive)
                    visual_fn += int(truth_positive and not visual_positive)
                    visual_tn += int(not truth_positive and not visual_positive)
                    candidate_tp += int(truth_positive and candidate)
                    candidate_fp += int(not truth_positive and candidate)
                    candidate_fn += int(truth_positive and not candidate)
                    candidate_tn += int(not truth_positive and not candidate)
                    if truth_positive and candidate:
                        count_support += 1
                        count_exact += int(int(relation["candidate_tie_count"]) == len(tied_notes))
                    if candidate:
                        predicted_y = relation["target_y_page"]
                        truth_y = (
                            [float(note["center_y"]) for note in tied_notes]
                            if args.layout == "score_tab"
                            else [float(reference_y[int(note["string"]) - 1]) for note in tied_notes]
                        )
                        y_matched += match_y(predicted_y, truth_y, tolerance=spacing * 0.5)
                        y_predicted += len(predicted_y)
                        y_truth += len(truth_y)
                    if visual_positive != truth_positive or candidate != truth_positive:
                        errors.append({
                            "source": source_id,
                            "measure": truth_measure["measure_number"],
                            "beat_index": truth_event["beat_index"],
                            "truth": truth_positive,
                            "visual": visual_positive,
                            "candidate": candidate,
                            "relation": relation,
                        })

                for prediction_index in unmatched:
                    relation = predictions[prediction_index]["tie_relation"]
                    visual_fp += int(relation["visual_positive"])
                    visual_tn += int(not relation["visual_positive"])
                    candidate_fp += int(relation["candidate"])
                    candidate_tn += int(not relation["candidate"])

        document_ir = {
            "tracks": [{"track": 1, "measures": document_measures}],
        }
        resolve_unambiguous_ties(document_ir)
        for measure in document_measures:
            for event in measure["events"]:
                continued = [note for note in event["notes"] if note.get("source") == "tie_continuation"]
                if not continued:
                    continue
                resolved_events += 1
                resolved_notes += len(continued)
                truth_event = truth_by_predicted_key.get((int(measure["number"]), int(event["order"])))
                truth_set = set() if truth_event is None else {
                    (int(note["string"]), int(note["fret"])) for note in truth_tied_notes(truth_event)
                }
                predicted_set = {(int(note["string"]), note["fret"]) for note in continued}
                resolved_event_exact += int(predicted_set == truth_set and bool(truth_set))
                resolved_notes_correct += len(predicted_set & truth_set)

    y_precision = y_matched / max(1, y_predicted)
    y_recall = y_matched / max(1, y_truth)
    report = {
        "split": args.split,
        "layout": args.layout,
        "tie_threshold": tie_threshold,
        "sources": source_ids,
        "pages": page_count,
        "measures": measure_count,
        "truth_events": truth_event_count,
        "matched_events": matched_event_count,
        "truth_tie_events": truth_tie_events,
        "truth_tie_notes": truth_tie_notes_count,
        "visual_presence": metrics(visual_tp, visual_fp, visual_fn, visual_tn),
        "semantic_consistent_candidate": metrics(candidate_tp, candidate_fp, candidate_fn, candidate_tn),
        "candidate_tie_count_accuracy": count_exact / max(1, count_support),
        "candidate_tie_count_exact": count_exact,
        "candidate_tie_count_support": count_support,
        "target_y": {
            "precision": y_precision,
            "recall": y_recall,
            "f1": 2 * y_precision * y_recall / max(1e-12, y_precision + y_recall),
            "matched": y_matched,
            "predicted": y_predicted,
            "truth_on_candidate_events": y_truth,
        },
        "conservative_auto_resolution": {
            "events": resolved_events,
            "event_exact": resolved_event_exact,
            "event_precision": resolved_event_exact / max(1, resolved_events),
            "notes": resolved_notes,
            "notes_correct": resolved_notes_correct,
            "note_precision": resolved_notes_correct / max(1, resolved_notes),
            "truth_note_coverage": resolved_notes_correct / max(1, truth_tie_notes_count),
        },
        "scope_warning": (
            f"This split contains only {truth_tie_events} positive tie events. Auto-resolution intentionally handles only "
            "string-local continuations with an unambiguous visual target; ambiguous candidates remain unresolved."
        ),
        "errors": errors,
    }
    output = database / tie_root / "models" / f"page_{args.split}_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
