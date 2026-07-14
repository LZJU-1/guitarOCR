from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from guitarocr.data.build_score_rhythm_dataset import build_event_crop, voice_target
from guitarocr.data.build_tab_rhythm_dataset import build_tab_event_crop, target as tab_voice_target
from guitarocr.models.rhythm_context_model import RhythmContextCNN
from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.paths import DATABASE_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import classify_rhythm, locate_page_events
from guitarocr.pipeline.infer_tuxguitar_tab_document import locate_tab_page_events


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_truth(page: dict, layout: str) -> list[dict]:
    systems: list[dict] = []
    for measure in page["measures"]:
        line_y = (
            measure["score_staff"]["line_y"]
            if layout == "score_tab" else measure["tab_staff"]["string_y"]
        )
        if not systems or abs(systems[-1]["reference_y"][0] - line_y[0]) > 1.0:
            systems.append({"reference_y": line_y, "measures": []})
        systems[-1]["measures"].append(measure)
    return systems


def predicted_voice_values(voice: dict) -> dict:
    return {
        "state": voice["state"]["value"],
        "duration_value": int(voice["duration"]["value"]),
        "dot": voice["dot"]["value"],
        "division": voice["division"]["value"],
    }


def voice_exact(predicted: dict, truth: dict) -> bool:
    if predicted["state"] != truth["state"]:
        return False
    if truth["state"] == "empty":
        return True
    return all(predicted[key] == truth[key] for key in ("duration_value", "dot", "division"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rhythm after pixel-only page event localisation.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--layout", choices=("score_tab", "tab_only"), default="score_tab"
    )
    args = parser.parse_args()
    database = args.database.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    locator_task = "score_event_locator" if args.layout == "score_tab" else "tab_event_locator"
    locator_name = "score_event_locator.pt" if args.layout == "score_tab" else "tab_event_locator.pt"
    rhythm_task = "rhythm_events" if args.layout == "score_tab" else "tab_rhythm_events"
    locator_checkpoint = torch.load(
        database / locator_task / "models" / locator_name,
        map_location=device, weights_only=False,
    )
    threshold = args.threshold if args.threshold is not None else float(locator_checkpoint.get("detection_threshold", 0.3))
    locator = ScoreEventLocator().to(device)
    locator.load_state_dict(locator_checkpoint["model_state"])
    locator.eval()
    rhythm_checkpoint = torch.load(
        database / rhythm_task / "models" / "rhythm_context_cnn.pt",
        map_location=device, weights_only=False,
    )
    rhythm = RhythmContextCNN().to(device)
    rhythm.load_state_dict(rhythm_checkpoint["model_state"])
    rhythm.eval()

    source_ids = {record["source_id"] for record in read_jsonl(
        database / locator_task / "manifests" / f"{args.split}.jsonl"
    )}
    page_paths = sorted(
        path for source_id in source_ids
        for path in (
            database / "labels" / "pages"
            / ("score_tab_rhythm" if args.layout == "score_tab" else "tab_only")
            / source_id
        ).glob("page_*.json")
    )
    truth_events = matched_events = false_positive_events = 0
    visible_support = [0, 0]
    visible_exact = [0, 0]
    matched_event_exact = 0
    primary_exact_recovered = 0
    for page_path in page_paths:
        truth_page = json.loads(page_path.read_text(encoding="utf-8"))
        truth_systems = group_truth(truth_page, args.layout)
        with Image.open(database / truth_page["image"]) as opened:
            page = opened.convert("L")
        predicted_systems = (
            locate_page_events(page, locator, device, threshold)
            if args.layout == "score_tab"
            else locate_tab_page_events(page, locator, device, threshold)
        )
        classify_rhythm(
            page, predicted_systems, rhythm, device, crop_root=None,
            crop_builder=build_tab_event_crop if args.layout == "tab_only" else build_event_crop,
            reference_line_key="tab_string_y" if args.layout == "tab_only" else "score_line_y",
        )
        if len(predicted_systems) != len(truth_systems):
            raise ValueError(f"System mismatch on {page_path}")
        for predicted_system, truth_system in zip(predicted_systems, truth_systems):
            if len(predicted_system["measures"]) != len(truth_system["measures"]):
                raise ValueError(f"Measure mismatch on {page_path}")
            spacing = (
                truth_system["reference_y"][-1] - truth_system["reference_y"][0]
            ) / max(1, len(truth_system["reference_y"]) - 1)
            for predicted_measure, truth_measure in zip(predicted_system["measures"], truth_system["measures"]):
                truths = truth_measure["events"]
                truth_events += len(truths)
                unmatched = set(range(len(truths)))
                for event in sorted(predicted_measure["events"], key=lambda item: item["locator_confidence"], reverse=True):
                    if not unmatched:
                        false_positive_events += 1
                        continue
                    truth_index = min(unmatched, key=lambda index: abs(float(truths[index]["x"]) - event["x"]))
                    if abs(float(truths[truth_index]["x"]) - event["x"]) > spacing * 0.5:
                        false_positive_events += 1
                        continue
                    unmatched.remove(truth_index)
                    matched_events += 1
                    truth_event = truths[truth_index]
                    truth_by_voice = {int(voice["voice_index"]): voice for voice in truth_event["voices"]}
                    all_exact = True
                    primary_exact = False
                    for voice_index in range(2):
                        target = (
                            voice_target(truth_by_voice.get(voice_index))
                            if args.layout == "score_tab"
                            else tab_voice_target(truth_by_voice.get(voice_index))
                        )
                        prediction = predicted_voice_values(event["voices"][voice_index])
                        exact = voice_exact(prediction, target)
                        all_exact &= exact
                        if voice_index == 0:
                            primary_exact = exact
                        if target["state"] != "empty":
                            visible_support[voice_index] += 1
                            visible_exact[voice_index] += int(exact)
                    matched_event_exact += int(all_exact)
                    primary_exact_recovered += int(primary_exact)

    report = {
        "split": args.split,
        "layout": args.layout,
        "pages": len(page_paths),
        "threshold": threshold,
        "truth_events": truth_events,
        "matched_events": matched_events,
        "missed_events": truth_events - matched_events,
        "false_positive_events": false_positive_events,
        "primary_voice_exact_on_matched": primary_exact_recovered / max(1, matched_events),
        "primary_voice_exact_end_to_end_recall": primary_exact_recovered / max(1, truth_events),
        "two_voice_event_exact_on_matched": matched_event_exact / max(1, matched_events),
        "two_voice_event_exact_end_to_end_recall": matched_event_exact / max(1, truth_events),
        "visible_voice_exact": {
            f"voice_{index}": visible_exact[index] / max(1, visible_support[index]) for index in range(2)
        },
        "visible_voice_support": {f"voice_{index}": visible_support[index] for index in range(2)},
        "scope": "Page pixels -> pixel geometry -> learned event x -> detected-centre rhythm crop -> rhythm CNN.",
    }
    output = database / locator_task / "models" / f"detected_rhythm_{args.split}_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
