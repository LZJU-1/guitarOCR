from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.paths import DATABASE_ROOT, WEIGHTS_ROOT
from guitarocr.pipeline.infer_tuxguitar_tab_document import locate_tab_page_events


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def match_columns(predictions: list[dict], truths: list[dict], tolerance: float) -> tuple[int, int, int, list[float]]:
    unmatched = set(range(len(truths)))
    true_positive = false_positive = 0
    errors: list[float] = []
    for prediction in sorted(
        predictions, key=lambda value: value["locator_confidence"], reverse=True
    ):
        if not unmatched:
            false_positive += 1
            continue
        truth_index = min(
            unmatched, key=lambda index: abs(float(truths[index]["x"]) - float(prediction["x"]))
        )
        error = abs(float(truths[truth_index]["x"]) - float(prediction["x"]))
        if error <= tolerance:
            unmatched.remove(truth_index)
            true_positive += 1
            errors.append(error)
        else:
            false_positive += 1
    return true_positive, false_positive, len(unmatched), errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pixel-only pure-TAB geometry and event localisation."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument(
        "--model", type=Path, default=WEIGHTS_ROOT / "tab_event_locator.pt"
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    database = args.database.resolve()
    source_ids = {
        row["source_id"]
        for row in read_jsonl(
            database / "tab_event_locator" / "manifests" / f"{args.split}.jsonl"
        )
    }
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScoreEventLocator().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(checkpoint.get("detection_threshold", 0.25))

    pages = exact_measure_pages = 0
    truth_measures = predicted_measures = 0
    true_positive = false_positive = false_negative = 0
    errors: list[float] = []
    reports = []
    for source_id in sorted(source_ids):
        root = database / "labels" / "pages" / "tab_only" / source_id
        for label_path in sorted(root.glob("page_*.json")):
            label = json.loads(label_path.read_text(encoding="utf-8"))
            with Image.open(database / label["image"]) as opened:
                image = opened.convert("L")
            systems = locate_tab_page_events(image, model, device, threshold)
            predicted = [measure for system in systems for measure in system["measures"]]
            truth = label["measures"]
            pages += 1
            truth_measures += len(truth)
            predicted_measures += len(predicted)
            exact_measure_pages += int(len(predicted) == len(truth))
            page_tp = page_fp = page_fn = 0
            if len(predicted) == len(truth):
                for predicted_measure, truth_measure in zip(predicted, truth):
                    string_y = [float(value) for value in truth_measure["tab_staff"]["string_y"]]
                    spacing = (string_y[-1] - string_y[0]) / max(1, len(string_y) - 1)
                    tp, fp, fn, measure_errors = match_columns(
                        predicted_measure["events"], truth_measure["events"], spacing * 0.5
                    )
                    page_tp += tp; page_fp += fp; page_fn += fn
                    errors.extend(measure_errors)
            else:
                page_fp += sum(len(measure["events"]) for measure in predicted)
                page_fn += sum(len(measure["events"]) for measure in truth)
            true_positive += page_tp
            false_positive += page_fp
            false_negative += page_fn
            reports.append({
                "source_id": source_id,
                "page_index": int(label["page_index"]),
                "truth_measures": len(truth),
                "predicted_measures": len(predicted),
                "true_positive": page_tp,
                "false_positive": page_fp,
                "false_negative": page_fn,
            })
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    metrics = {
        "split": args.split,
        "pages": pages,
        "geometry": {
            "exact_measure_pages": exact_measure_pages,
            "truth_measures": truth_measures,
            "predicted_measures": predicted_measures,
        },
        "events": {
            "threshold": threshold,
            "precision_half_tab_spacing": precision,
            "recall_half_tab_spacing": recall,
            "f1_half_tab_spacing": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "mean_x_error_px": sum(errors) / max(1, len(errors)),
        },
        "scope": "Only full-page pixels are used; labels are evaluation-only.",
        "page_reports": reports,
    }
    output = args.output or (
        database / "tab_event_locator" / "models" / f"page_end_to_end_{args.split}_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
