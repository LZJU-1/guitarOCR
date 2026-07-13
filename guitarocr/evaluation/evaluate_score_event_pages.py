from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.paths import DATABASE_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import locate_page_events


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_ground_truth_measures(page: dict) -> list[dict]:
    systems: list[dict] = []
    for measure in page["measures"]:
        lines = measure["score_staff"]["line_y"]
        if not systems or abs(systems[-1]["score_line_y"][0] - lines[0]) > 1.0:
            systems.append({"score_line_y": lines, "measures": []})
        systems[-1]["measures"].append(measure)
    return systems


def interval_iou(first: list[float], second: list[float]) -> float:
    first_left, first_right = first[0], first[0] + first[2]
    second_left, second_right = second[0], second[0] + second[2]
    intersection = max(0.0, min(first_right, second_right) - max(first_left, second_left))
    union = max(first_right, second_right) - min(first_left, second_left)
    return intersection / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pixel-only score_tab event localisation on page images.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    database = args.database.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = database / "score_event_locator" / "models" / "score_event_locator.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ScoreEventLocator().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("detection_threshold", 0.3))

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

    pages_exact_systems = pages_exact_measures = 0
    predicted_systems = truth_systems = predicted_measures = truth_measures = 0
    true_positive = false_positive = false_negative = 0
    x_errors: list[float] = []
    line_errors: list[float] = []
    page_reports: list[dict] = []
    for page_path in page_paths:
        truth_page = json.loads(page_path.read_text(encoding="utf-8"))
        with Image.open(database / truth_page["image"]) as opened:
            image = opened.convert("L")
        predicted = locate_page_events(image, model, device, threshold)
        truth = group_ground_truth_measures(truth_page)
        predicted_systems += len(predicted)
        truth_systems += len(truth)
        predicted_measures += sum(len(system["measures"]) for system in predicted)
        truth_measures += sum(len(system["measures"]) for system in truth)
        exact_systems = len(predicted) == len(truth)
        if exact_systems:
            pages_exact_systems += 1
        exact_measures = exact_systems and all(
            len(predicted[index]["measures"]) == len(truth[index]["measures"])
            for index in range(len(truth))
        )
        if exact_measures:
            pages_exact_measures += 1

        unused_predicted = {
            (system_index, measure_index)
            for system_index, system in enumerate(predicted)
            for measure_index, _ in enumerate(system["measures"])
        }
        page_tp = page_fp = page_fn = 0
        for truth_system in truth:
            spacing = (truth_system["score_line_y"][-1] - truth_system["score_line_y"][0]) / 4.0
            if predicted:
                predicted_system_index = min(
                    range(len(predicted)),
                    key=lambda index: abs(predicted[index]["score_line_y"][0] - truth_system["score_line_y"][0]),
                )
                predicted_system = predicted[predicted_system_index]
                line_errors.extend(
                    abs(float(a) - float(b))
                    for a, b in zip(predicted_system["score_line_y"], truth_system["score_line_y"])
                )
            else:
                predicted_system_index = -1
                predicted_system = {"measures": []}
            for truth_measure in truth_system["measures"]:
                candidates = [
                    (interval_iou(measure["bbox"], truth_measure["bbox"]), measure_index, measure)
                    for measure_index, measure in enumerate(predicted_system["measures"])
                    if (predicted_system_index, measure_index) in unused_predicted
                ]
                if not candidates or max(candidates, key=lambda item: item[0])[0] < 0.5:
                    count = len(truth_measure["events"])
                    false_negative += count
                    page_fn += count
                    continue
                _, predicted_measure_index, predicted_measure = max(candidates, key=lambda item: item[0])
                unused_predicted.remove((predicted_system_index, predicted_measure_index))
                truths = [float(event["x"]) for event in truth_measure["events"]]
                unmatched = set(range(len(truths)))
                for event in sorted(predicted_measure["events"], key=lambda item: item["locator_confidence"], reverse=True):
                    if not unmatched:
                        false_positive += 1
                        page_fp += 1
                        continue
                    truth_index = min(unmatched, key=lambda index: abs(truths[index] - event["x"]))
                    error = abs(truths[truth_index] - event["x"])
                    if error <= spacing * 0.5:
                        unmatched.remove(truth_index)
                        true_positive += 1
                        page_tp += 1
                        x_errors.append(error)
                    else:
                        false_positive += 1
                        page_fp += 1
                false_negative += len(unmatched)
                page_fn += len(unmatched)
        for system_index, measure_index in unused_predicted:
            count = len(predicted[system_index]["measures"][measure_index]["events"])
            false_positive += count
            page_fp += count
        page_reports.append({
            "source_id": truth_page["source_id"], "page_index": truth_page["page_index"],
            "systems_exact": exact_systems, "measures_exact": exact_measures,
            "true_positive": page_tp, "false_positive": page_fp, "false_negative": page_fn,
        })

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    report = {
        "split": args.split,
        "pages": len(page_paths),
        "geometry": {
            "pages_exact_system_count": pages_exact_systems,
            "pages_exact_measure_count": pages_exact_measures,
            "predicted_systems": predicted_systems,
            "truth_systems": truth_systems,
            "predicted_measures": predicted_measures,
            "truth_measures": truth_measures,
            "mean_score_line_error_px": float(np.mean(line_errors)) if line_errors else 0.0,
        },
        "events": {
            "threshold": threshold,
            "precision_half_staff_spacing": precision,
            "recall_half_staff_spacing": recall,
            "f1_half_staff_spacing": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "mean_x_error_px": float(np.mean(x_errors)) if x_errors else 0.0,
            "median_x_error_px": float(np.median(x_errors)) if x_errors else 0.0,
            "p95_x_error_px": float(np.percentile(x_errors, 95)) if x_errors else 0.0,
        },
        "scope": "Only page pixels are used by geometry and event inference; labels are evaluation-only.",
        "page_reports": page_reports,
    }
    output = database / "score_event_locator" / "models" / f"page_end_to_end_{args.split}_metrics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "page_reports"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
