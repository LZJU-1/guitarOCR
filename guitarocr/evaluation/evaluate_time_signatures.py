from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.paths import DATABASE_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import locate_page_events
from guitarocr.pipeline.time_signature_recognizer import load_atomic_model, propagate_time_signatures


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate printed time-signature OCR and propagation.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--threshold", type=float, default=0.45)
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
    atomic, classes = load_atomic_model(
        database / "symbol_cnn" / "models" / "atomic_symbol_cnn.pt", device
    )

    if args.split == "all":
        source_ids = {path.stem for path in (database / "labels" / "songs").glob("*.json")}
    else:
        source_ids = {record["source_id"] for record in read_jsonl(
            database / "score_event_locator" / "manifests" / f"{args.split}.jsonl"
        )}
    printed_tp = printed_fp = printed_fn = printed_value_correct = 0
    propagated_correct = propagated_total = 0
    errors: list[dict] = []
    for source_id in sorted(source_ids):
        song = json.loads((database / "labels" / "songs" / f"{source_id}.json").read_text(encoding="utf-8"))
        song_measures = {int(measure["number"]): measure for measure in song["measures"]}
        previous_truth: tuple[int, int] | None = None
        carried: tuple[int, int] | None = None
        for page_path in sorted((database / "labels" / "pages" / "score_tab_rhythm" / source_id).glob("page_*.json")):
            page_label = json.loads(page_path.read_text(encoding="utf-8"))
            with Image.open(database / page_label["image"]) as opened:
                image = opened.convert("L")
            systems = locate_page_events(image, locator, device, threshold=0.3)
            carried = propagate_time_signatures(
                image, systems, atomic, classes, device, initial=carried, threshold=args.threshold
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
                previous_truth = truth

    precision = printed_tp / max(1, printed_tp + printed_fp)
    recall = printed_tp / max(1, printed_tp + printed_fn)
    report = {
        "split": args.split,
        "threshold": args.threshold,
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
        "measure_support": propagated_total,
        "errors": errors,
        "scope": "Pixel-only printed signature OCR with document-order carry-forward; labels are evaluation-only.",
    }
    output = database / "score_event_locator" / "models" / f"time_signature_{args.split}_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
