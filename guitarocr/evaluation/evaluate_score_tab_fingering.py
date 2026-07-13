from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import DATABASE_ROOT
from guitarocr.pipeline.score_tab_fingering import detect_tab_fingering
from guitarocr.pipeline.score_tab_geometry import detect_score_tab_geometry


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_events(page_measure: dict, song_measure: dict) -> list[dict]:
    semantic_beats = {int(beat["precise_start"]): beat for beat in song_measure["beats"]}
    expected: list[dict] = []
    for event in page_measure["events"]:
        semantic = semantic_beats[int(event["precise_start"])]
        semantic_voices = {int(voice["index"]): voice for voice in semantic["voices"]}
        notes: list[tuple[int, int | str]] = []
        for voice in event["voices"]:
            semantic_voice = semantic_voices.get(int(voice["voice_index"]), {"notes": []})
            semantic_notes = {
                (int(note["string"]), int(note["fret"])): note
                for note in semantic_voice["notes"]
            }
            for note in voice["notes"]:
                if note["tied"]:
                    continue
                key = (int(note["string"]), int(note["fret"]))
                semantic_note = semantic_notes[key]
                value: int | str = "X" if semantic_note["effects"]["dead"] else key[1]
                notes.append((key[0], value))
        if notes:
            expected.append({"x": float(event["x"]), "notes": sorted(notes, key=lambda item: item[0])})
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the tab_only detector on score_tab TAB regions.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--split", choices=("validation", "test", "all"), default="test")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    database = args.database.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        database / "tab_detector" / "models" / "tab_symbol_detector.pt",
        map_location=device, weights_only=False,
    )
    classes = checkpoint["classes"]
    model = TabSymbolDetector(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if args.split == "all":
        source_ids = {path.parent.name for path in (database / "labels" / "pages" / "score_tab_rhythm").glob("*/*.json")}
    else:
        source_ids = {record["source_id"] for record in read_jsonl(
            database / "score_event_locator" / "manifests" / f"{args.split}.jsonl"
        )}
    page_paths = sorted(
        path for source_id in source_ids
        for path in (database / "labels" / "pages" / "score_tab_rhythm" / source_id).glob("page_*.json")
    )

    event_tp = event_fp = event_fn = exact_events = 0
    note_tp = note_fp = note_fn = 0
    x_errors: list[float] = []
    expected_event_total = 0
    page_reports: list[dict] = []
    song_cache: dict[str, dict] = {}
    for page_path in page_paths:
        page_label = json.loads(page_path.read_text(encoding="utf-8"))
        source_id = page_label["source_id"]
        song = song_cache.setdefault(
            source_id,
            json.loads((database / "labels" / "songs" / f"{source_id}.json").read_text(encoding="utf-8")),
        )
        song_measures = {int(measure["number"]): measure for measure in song["measures"]}
        with Image.open(database / page_label["image"]) as opened:
            image = opened.convert("L")
        systems = detect_score_tab_geometry(image)
        detect_tab_fingering(image, systems, model, classes, device, threshold=args.threshold)
        predicted_measures = [measure for system in systems for measure in system["measures"]]
        if len(predicted_measures) != len(page_label["measures"]):
            raise ValueError(f"Measure mismatch on {page_path}")
        page_exact = page_expected = page_predicted = 0
        for predicted_measure, page_measure in zip(predicted_measures, page_label["measures"]):
            truths = expected_events(page_measure, song_measures[int(page_measure["measure_number"])])
            predictions = predicted_measure["tab_events"]
            expected_event_total += len(truths)
            page_expected += len(truths)
            page_predicted += len(predictions)
            unmatched = set(range(len(truths)))
            for prediction in predictions:
                if not unmatched:
                    event_fp += 1
                    note_fp += len(prediction["notes"])
                    continue
                truth_index = min(unmatched, key=lambda index: abs(truths[index]["x"] - prediction["x"]))
                error = abs(truths[truth_index]["x"] - prediction["x"])
                spacing = float(systems[0]["tab_spacing"]) if systems else 20.0
                if error > spacing * 0.5:
                    event_fp += 1
                    note_fp += len(prediction["notes"])
                    continue
                unmatched.remove(truth_index)
                event_tp += 1
                x_errors.append(error)
                truth_notes = set(truths[truth_index]["notes"])
                predicted_notes = {(int(note["string"]), note["fret"]) for note in prediction["notes"]}
                exact = truth_notes == predicted_notes
                exact_events += int(exact)
                page_exact += int(exact)
                note_tp += len(truth_notes & predicted_notes)
                note_fp += len(predicted_notes - truth_notes)
                note_fn += len(truth_notes - predicted_notes)
            event_fn += len(unmatched)
            for truth_index in unmatched:
                note_fn += len(truths[truth_index]["notes"])
        page_reports.append({
            "source_id": source_id, "page_index": page_label["page_index"],
            "expected_events": page_expected, "predicted_events": page_predicted,
            "exact_fingering_events": page_exact,
        })

    event_precision = event_tp / max(1, event_tp + event_fp)
    event_recall = event_tp / max(1, event_tp + event_fn)
    note_precision = note_tp / max(1, note_tp + note_fp)
    note_recall = note_tp / max(1, note_tp + note_fn)
    report = {
        "split": args.split,
        "pages": len(page_paths),
        "threshold": args.threshold,
        "visible_tab_events": {
            "precision": event_precision,
            "recall": event_recall,
            "f1": 2 * event_precision * event_recall / max(1e-12, event_precision + event_recall),
            "true_positive": event_tp, "false_positive": event_fp, "false_negative": event_fn,
            "exact_fingering_accuracy_on_matched": exact_events / max(1, event_tp),
            "exact_fingering_recall": exact_events / max(1, expected_event_total),
            "mean_x_error_px": sum(x_errors) / max(1, len(x_errors)),
        },
        "notes": {
            "precision": note_precision,
            "recall": note_recall,
            "f1": 2 * note_precision * note_recall / max(1e-12, note_precision + note_recall),
            "true_positive": note_tp, "false_positive": note_fp, "false_negative": note_fn,
        },
        "scope": "Existing tab_only detector applied unchanged to score_tab TAB regions; tied notes are not expected as visible glyphs.",
        "page_reports": page_reports,
    }
    output = database / "score_event_locator" / "models" / f"score_tab_fingering_{args.split}_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "page_reports"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
