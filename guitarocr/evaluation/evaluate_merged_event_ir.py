from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from guitarocr.data.build_score_rhythm_dataset import voice_target
from guitarocr.models.rhythm_context_model import RhythmContextCNN
from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import DATABASE_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import classify_rhythm, locate_page_events
from guitarocr.pipeline.score_tab_fingering import build_score_ir, detect_tab_fingering


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_truth(page: dict) -> list[dict]:
    systems: list[dict] = []
    for measure in page["measures"]:
        line_y = measure["score_staff"]["line_y"]
        if not systems or abs(systems[-1]["score_line_y"][0] - line_y[0]) > 1.0:
            systems.append({"score_line_y": line_y, "measures": []})
        systems[-1]["measures"].append(measure)
    return systems


def expected_visible_notes(event: dict, semantic_measure: dict) -> set[tuple[int, int | str]]:
    semantic_beat = next(
        beat for beat in semantic_measure["beats"]
        if int(beat["precise_start"]) == int(event["precise_start"])
    )
    semantic_voices = {int(voice["index"]): voice for voice in semantic_beat["voices"]}
    result: set[tuple[int, int | str]] = set()
    for voice in event["voices"]:
        semantic_voice = semantic_voices.get(int(voice["voice_index"]), {"notes": []})
        semantic_notes = {
            (int(note["string"]), int(note["fret"])): note for note in semantic_voice["notes"]
        }
        for note in voice["notes"]:
            if note["tied"]:
                continue
            key = (int(note["string"]), int(note["fret"]))
            source_note = semantic_notes[key]
            result.add((key[0], "X" if source_note["effects"]["dead"] else key[1]))
    return result


def rhythm_voice_exact(prediction: dict, target: dict) -> bool:
    if prediction["state"] != target["state"]:
        return False
    if target["state"] == "empty":
        return True
    return (
        prediction["duration_value"] == int(target["duration_value"])
        and prediction["dot"] == target["dot"]
        and prediction["division"] == target["division"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate merged pixel-only score/TAB Event IR.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--event-threshold", type=float, default=0.3)
    parser.add_argument("--tab-threshold", type=float, default=0.3)
    args = parser.parse_args()
    database = args.database.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    locator_checkpoint = torch.load(
        database / "score_event_locator" / "models" / "score_event_locator.pt",
        map_location=device, weights_only=False,
    )
    locator = ScoreEventLocator().to(device)
    locator.load_state_dict(locator_checkpoint["model_state"])
    locator.eval()
    rhythm_checkpoint = torch.load(
        database / "rhythm_events" / "models" / "rhythm_context_cnn.pt",
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
    tab_model = TabSymbolDetector(len(tab_classes)).to(device)
    tab_model.load_state_dict(tab_checkpoint["model_state"])
    tab_model.eval()

    source_ids = {record["source_id"] for record in read_jsonl(
        database / "score_event_locator" / "manifests" / f"{args.split}.jsonl"
    )}
    page_paths = sorted(
        path for source_id in source_ids
        for path in (database / "labels" / "pages" / "score_tab_rhythm" / source_id).glob("page_*.json")
    )
    truth_count = matched_count = extra_count = 0
    primary_rhythm_exact = two_voice_rhythm_exact = 0
    fingering_exact = primary_core_exact = two_voice_core_exact = 0
    visible_tab_event_count = visible_tab_event_exact = 0
    song_cache: dict[str, dict] = {}
    for page_path in page_paths:
        page_label = json.loads(page_path.read_text(encoding="utf-8"))
        source_id = page_label["source_id"]
        song = song_cache.setdefault(
            source_id,
            json.loads((database / "labels" / "songs" / f"{source_id}.json").read_text(encoding="utf-8")),
        )
        semantic_measures = {int(measure["number"]): measure for measure in song["measures"]}
        truth_systems = group_truth(page_label)
        with Image.open(database / page_label["image"]) as opened:
            page = opened.convert("L")
        systems = locate_page_events(page, locator, device, args.event_threshold)
        classify_rhythm(page, systems, rhythm, device, crop_root=None)
        detect_tab_fingering(page, systems, tab_model, tab_classes, device, threshold=args.tab_threshold)
        score_ir = build_score_ir(systems)
        predicted_measures = score_ir["tracks"][0]["measures"]
        truth_measures = [measure for system in truth_systems for measure in system["measures"]]
        if len(predicted_measures) != len(truth_measures):
            raise ValueError(f"Measure mismatch on {page_path}")
        for predicted_measure, truth_measure in zip(predicted_measures, truth_measures):
            truths = truth_measure["events"]
            truth_count += len(truths)
            unmatched_predictions = set(range(len(predicted_measure["events"])))
            spacing = (truth_measure["score_staff"]["line_y"][-1] - truth_measure["score_staff"]["line_y"][0]) / 4.0
            semantic_measure = semantic_measures[int(truth_measure["measure_number"])]
            for truth_event in truths:
                if not unmatched_predictions:
                    continue
                prediction_index = min(
                    unmatched_predictions,
                    key=lambda index: abs(predicted_measure["events"][index]["x"] - float(truth_event["x"])),
                )
                prediction = predicted_measure["events"][prediction_index]
                if abs(prediction["x"] - float(truth_event["x"])) > spacing * 0.5:
                    continue
                unmatched_predictions.remove(prediction_index)
                matched_count += 1
                truth_by_voice = {int(voice["voice_index"]): voice for voice in truth_event["voices"]}
                voice_matches = []
                for voice_index in range(2):
                    target = voice_target(truth_by_voice.get(voice_index))
                    voice_matches.append(rhythm_voice_exact(prediction["voices"][voice_index], target))
                primary_rhythm_exact += int(voice_matches[0])
                all_rhythm = all(voice_matches)
                two_voice_rhythm_exact += int(all_rhythm)

                truth_notes = expected_visible_notes(truth_event, semantic_measure)
                predicted_notes = {(int(note["string"]), note["fret"]) for note in prediction["notes"]}
                notes_exact = truth_notes == predicted_notes
                fingering_exact += int(notes_exact)
                primary_core_exact += int(voice_matches[0] and notes_exact)
                two_voice_core_exact += int(all_rhythm and notes_exact)
                if truth_notes:
                    visible_tab_event_count += 1
                    visible_tab_event_exact += int(notes_exact)
            extra_count += len(unmatched_predictions)

    report = {
        "split": args.split,
        "pages": len(page_paths),
        "truth_events": truth_count,
        "matched_events": matched_count,
        "missed_events": truth_count - matched_count,
        "extra_score_events": extra_count,
        "primary_rhythm_exact_recall": primary_rhythm_exact / max(1, truth_count),
        "two_voice_rhythm_exact_recall": two_voice_rhythm_exact / max(1, truth_count),
        "visible_fingering_exact_all_event_recall": fingering_exact / max(1, truth_count),
        "visible_tab_event_exact_recall": visible_tab_event_exact / max(1, visible_tab_event_count),
        "visible_tab_event_support": visible_tab_event_count,
        "primary_core_exact_recall": primary_core_exact / max(1, truth_count),
        "two_voice_core_exact_recall": two_voice_core_exact / max(1, truth_count),
        "core_definition": (
            "Detected event + predicted rhythm + TAB glyphs visibly printed at that event. "
            "Tied continuation notes, onset totals, time signature, tuning and effects are not yet included."
        ),
    }
    output = database / "score_event_locator" / "models" / f"merged_event_ir_{args.split}_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
