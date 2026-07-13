from __future__ import annotations

import argparse
import copy
import json
from fractions import Fraction
from pathlib import Path

from PIL import Image
import torch

from guitarocr.data.build_score_rhythm_dataset import voice_target
from guitarocr.models.rhythm_context_model import RhythmContextCNN
from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import DATABASE_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import classify_rhythm, locate_page_events
from guitarocr.pipeline.measure_rhythm_constraints import (
    PROPOSAL_RELATIVE_PROBABILITY_THRESHOLD,
    audit_measure,
    audit_score_ir,
    duration_fraction,
)
from guitarocr.pipeline.score_tab_fingering import build_score_ir, detect_tab_fingering
from guitarocr.pipeline.time_signature_recognizer import load_atomic_model, propagate_time_signatures


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def match_events(predicted: list[dict], truth: list[dict], spacing: float) -> tuple[dict[int, int], set[int]]:
    unmatched_predictions = set(range(len(predicted)))
    matches: dict[int, int] = {}
    for truth_index, truth_event in enumerate(truth):
        if not unmatched_predictions:
            break
        prediction_index = min(
            unmatched_predictions,
            key=lambda index: abs(predicted[index]["x"] - float(truth_event["x"])),
        )
        if abs(predicted[prediction_index]["x"] - float(truth_event["x"])) <= spacing * 0.5:
            matches[prediction_index] = truth_index
            unmatched_predictions.remove(prediction_index)
    return matches, unmatched_predictions


def primary_target(truth_event: dict) -> dict:
    truth_by_voice = {int(voice["voice_index"]): voice for voice in truth_event["voices"]}
    return voice_target(truth_by_voice.get(0))


def apply_plausible_primary_proposal(measure: dict) -> dict:
    corrected = copy.deepcopy(measure)
    proposal = corrected["rhythm_audit"]["voices"]["voice_0"].get("correction_proposal")
    if proposal is None or not proposal["plausible"]:
        return corrected
    removed: set[int] = set()
    for modification in proposal["modifications"]:
        event_index = int(modification["event_index"])
        if modification["action"] == "remove_low_confidence_event":
            removed.add(event_index)
            continue
        replacement = modification["to"]
        voice = corrected["events"][event_index]["voices"][0]
        voice["duration_value"] = int(replacement["duration_value"])
        voice["dot"] = replacement["dot"]
        voice["division"] = replacement["division"]
    corrected["events"] = [
        event for index, event in enumerate(corrected["events"]) if index not in removed
    ]
    corrected["rhythm_audit"] = audit_measure(corrected)
    return corrected


def proposal_is_correct(measure: dict, truth: list[dict], spacing: float) -> bool | None:
    proposal = measure["rhythm_audit"]["voices"]["voice_0"].get("correction_proposal")
    if proposal is None:
        return None
    matches, unmatched = match_events(measure["events"], truth, spacing)
    for modification in proposal["modifications"]:
        event_index = int(modification["event_index"])
        if modification["action"] == "remove_low_confidence_event":
            if event_index not in unmatched:
                return False
            continue
        if event_index not in matches:
            return False
        target = primary_target(truth[matches[event_index]])
        replacement = modification["to"]
        if target["state"] == "empty":
            return False
        if not (
            int(replacement["duration_value"]) == int(target["duration_value"])
            and replacement["dot"] == target["dot"]
            and replacement["division"] == target["division"]
        ):
            return False
    return True


def truth_primary_onsets(truth: list[dict]) -> dict[int, Fraction]:
    cursor = Fraction(0, 1)
    onsets: dict[int, Fraction] = {}
    for truth_index, truth_event in enumerate(truth):
        target = primary_target(truth_event)
        if target["state"] == "empty":
            continue
        onsets[truth_index] = cursor
        cursor += duration_fraction(
            int(target["duration_value"]), target["dot"], target["division"]
        )
    return onsets


def score_measure(
    measure: dict, truth: list[dict], spacing: float
) -> tuple[int, int, int, bool, int]:
    matches, unmatched = match_events(measure["events"], truth, spacing)
    exact = 0
    onset_exact = 0
    truth_onsets = truth_primary_onsets(truth)
    for prediction_index, truth_index in matches.items():
        exact += int(
            rhythm_voice_exact(
                measure["events"][prediction_index]["voices"][0],
                primary_target(truth[truth_index]),
            )
        )
        if truth_index in truth_onsets:
            prediction_onset = measure["events"][prediction_index]["voices"][0].get("onset")
            onset_exact += int(
                prediction_onset is not None
                and prediction_onset["text"] == (
                    f"{truth_onsets[truth_index].numerator}/{truth_onsets[truth_index].denominator}"
                )
            )
    fully_correct = exact == len(truth) and not unmatched
    return exact, onset_exact, len(truth_onsets), fully_correct, len(measure["events"])


def exact_metrics(exact: int, predicted: int, truth: int) -> dict:
    precision = exact / max(1, predicted)
    recall = exact / max(1, truth)
    return {
        "exact": exact,
        "predicted": predicted,
        "truth": truth,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate exact measure-capacity audits and advisory primary-rhythm corrections."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--event-threshold", type=float, default=0.3)
    parser.add_argument("--tab-threshold", type=float, default=0.3)
    parser.add_argument("--time-signature-threshold", type=float, default=0.20)
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
    tab = TabSymbolDetector(len(tab_classes)).to(device)
    tab.load_state_dict(tab_checkpoint["model_state"])
    tab.eval()
    atomic, atomic_classes = load_atomic_model(
        database / "symbol_cnn" / "models" / "atomic_symbol_cnn.pt", device
    )

    source_ids = {record["source_id"] for record in read_jsonl(
        database / "score_event_locator" / "manifests" / f"{args.split}.jsonl"
    )}
    truth_event_count = baseline_exact = constrained_exact = 0
    primary_onset_support = baseline_onset_exact = constrained_onset_exact = 0
    baseline_prediction_count = constrained_prediction_count = 0
    measure_count = baseline_full_measures = constrained_full_measures = 0
    audit_counts = {"exact": 0, "underfilled": 0, "overfilled": 0}
    proposal_count = plausible_count = proposal_correct = plausible_correct = 0
    errors: list[dict] = []

    for source_id in sorted(source_ids):
        carried_signature: tuple[int, int] | None = None
        page_paths = sorted(
            (database / "labels" / "pages" / "score_tab_rhythm" / source_id).glob("page_*.json")
        )
        for page_path in page_paths:
            page_label = json.loads(page_path.read_text(encoding="utf-8"))
            with Image.open(database / page_label["image"]) as opened:
                page = opened.convert("L")
            systems = locate_page_events(page, locator, device, args.event_threshold)
            carried_signature = propagate_time_signatures(
                page, systems, atomic, atomic_classes, device,
                initial=carried_signature, threshold=args.time_signature_threshold,
            )
            classify_rhythm(page, systems, rhythm, device, crop_root=None)
            detect_tab_fingering(page, systems, tab, tab_classes, device, threshold=args.tab_threshold)
            score_ir = build_score_ir(systems)
            audit_score_ir(score_ir)
            predicted_measures = score_ir["tracks"][0]["measures"]
            truth_measures = page_label["measures"]
            if len(predicted_measures) != len(truth_measures):
                raise ValueError(f"Measure mismatch on {page_path}")

            for prediction, truth_measure in zip(predicted_measures, truth_measures):
                measure_count += 1
                truth_events = truth_measure["events"]
                truth_event_count += len(truth_events)
                spacing = (
                    truth_measure["score_staff"]["line_y"][-1]
                    - truth_measure["score_staff"]["line_y"][0]
                ) / 4.0
                primary_audit = prediction["rhythm_audit"]["voices"]["voice_0"]
                status = primary_audit["status"]
                if status in audit_counts:
                    audit_counts[status] += 1
                proposal = primary_audit.get("correction_proposal")
                correctness = proposal_is_correct(prediction, truth_events, spacing)
                if proposal is not None:
                    proposal_count += 1
                    proposal_correct += int(correctness is True)
                    plausible_count += int(proposal["plausible"])
                    plausible_correct += int(proposal["plausible"] and correctness is True)
                    if correctness is not True:
                        errors.append({
                            "source": source_id,
                            "measure": truth_measure["measure_number"],
                            "plausible": proposal["plausible"],
                            "proposal": proposal,
                        })

                exact, onset_exact, onset_support, full, predicted_count = score_measure(
                    prediction, truth_events, spacing
                )
                baseline_exact += exact
                baseline_onset_exact += onset_exact
                primary_onset_support += onset_support
                baseline_full_measures += int(full)
                baseline_prediction_count += predicted_count
                corrected = apply_plausible_primary_proposal(prediction)
                exact, onset_exact, _, full, predicted_count = score_measure(
                    corrected, truth_events, spacing
                )
                constrained_exact += exact
                constrained_onset_exact += onset_exact
                constrained_full_measures += int(full)
                constrained_prediction_count += predicted_count

    report = {
        "split": args.split,
        "pages": sum(
            len(list((database / "labels" / "pages" / "score_tab_rhythm" / source_id).glob("page_*.json")))
            for source_id in source_ids
        ),
        "measures": measure_count,
        "truth_events": truth_event_count,
        "capacity_audit_primary": {
            **audit_counts,
            "exact_rate": audit_counts["exact"] / max(1, measure_count),
        },
        "correction_proposals": {
            "all": proposal_count,
            "all_correct": proposal_correct,
            "all_precision": proposal_correct / max(1, proposal_count),
            "plausible": plausible_count,
            "plausible_correct": plausible_correct,
            "plausible_precision": plausible_correct / max(1, plausible_count),
        },
        "baseline_primary_rhythm": exact_metrics(
            baseline_exact, baseline_prediction_count, truth_event_count
        ),
        "plausible_corrections_applied_primary_rhythm": exact_metrics(
            constrained_exact, constrained_prediction_count, truth_event_count
        ),
        "primary_onset_exact": {
            "baseline": baseline_onset_exact / max(1, primary_onset_support),
            "after_plausible_corrections": constrained_onset_exact / max(1, primary_onset_support),
            "baseline_exact": baseline_onset_exact,
            "after_exact": constrained_onset_exact,
            "support": primary_onset_support,
        },
        "fully_correct_measures": {
            "baseline": baseline_full_measures,
            "after_plausible_corrections": constrained_full_measures,
            "support": measure_count,
        },
        "policy": (
            f"Only proposals with relative probability >= {PROPOSAL_RELATIVE_PROBABILITY_THRESHOLD:.2f} are "
            "applied for the constrained comparison. "
            "Production IR remains advisory and preserves the original CNN output."
        ),
        "errors": errors,
    }
    output = database / "score_event_locator" / "models" / f"measure_constraints_{args.split}_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
