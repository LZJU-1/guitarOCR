from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import DATABASE_ROOT
from guitarocr.training.train_tab_detector import TabDetectorDataset, box_iou, decode_detections


def match_items(predictions: list[dict], truths: list[dict], mode: str) -> tuple[int, int, int]:
    used: set[int] = set()
    true_positive = 0
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        best_index = -1
        best_value = float("-inf")
        px, py, pw, ph = prediction["bbox"]
        prediction_center = (px + pw / 2, py + ph / 2)
        for truth_index, truth in enumerate(truths):
            if truth_index in used or truth["class_index"] != prediction["class_index"]:
                continue
            if mode == "iou30":
                value = box_iou(prediction["bbox"], truth["bbox"])
                valid = value >= 0.3
            else:
                tx, ty, tw, th = truth["bbox"]
                distance = ((prediction_center[0] - (tx + tw / 2)) ** 2 +
                            (prediction_center[1] - (ty + th / 2)) ** 2) ** 0.5
                value = -distance
                valid = distance <= 6.0
            if valid and value > best_value:
                best_value = value
                best_index = truth_index
        if best_index >= 0:
            used.add(best_index)
            true_positive += 1
    return true_positive, len(predictions) - true_positive, len(truths) - true_positive


def metric(counter: Counter) -> dict[str, float | int]:
    tp, fp, fn = counter["tp"], counter["fp"], counter["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def draw_overlay(
    image_path: Path,
    output_path: Path,
    predictions: list[dict],
    truths: list[dict],
    classes: list[str],
) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for truth in truths:
        x, y, width, height = truth["bbox"]
        draw.rectangle((x, y, x + width, y + height), outline=(0, 180, 60), width=1)
    for prediction in predictions:
        x, y, width, height = prediction["bbox"]
        draw.rectangle((x, y, x + width, y + height), outline=(230, 30, 30), width=1)
        draw.text((x, max(0, y - 9)), classes[prediction["class_index"]].replace("digit_", ""), fill=(200, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed evaluation for the compact TAB symbol detector.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--worst", type=int, default=12)
    args = parser.parse_args()

    database = args.database.resolve()
    manifest_root = database / "tab_detector" / "manifests"
    model_root = database / "tab_detector" / "models"
    checkpoint = torch.load(model_root / "tab_symbol_detector.pt", map_location="cuda", weights_only=False)
    classes = checkpoint["classes"]
    dataset = TabDetectorDataset(database, manifest_root / f"{args.split}.jsonl", classes, training=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    model = TabSymbolDetector(len(classes)).cuda().eval()
    model.load_state_dict(checkpoint["model_state"])

    totals: dict[str, Counter] = {"iou30": Counter(), "center6": Counter()}
    per_source: dict[str, Counter] = defaultdict(Counter)
    per_class: dict[str, Counter] = defaultdict(Counter)
    sample_rows: list[dict] = []
    record_by_id = {record["sample_id"]: record for record in dataset.records}

    with torch.inference_mode():
        for raw_batch in loader:
            outputs = model(raw_batch["image"].cuda(non_blocking=True))
            decoded = decode_detections(outputs, threshold=args.threshold)
            for batch_index, predictions in enumerate(decoded):
                sample_id = raw_batch["sample_id"][batch_index]
                object_count = int(raw_batch["masks"][batch_index].sum().item())
                truths = [
                    {
                        "class_index": int(raw_batch["classes"][batch_index, index].item()),
                        "bbox": raw_batch["boxes"][batch_index, index].tolist(),
                    }
                    for index in range(object_count)
                ]
                iou_counts = match_items(predictions, truths, "iou30")
                center_counts = match_items(predictions, truths, "center6")
                for name, counts in (("iou30", iou_counts), ("center6", center_counts)):
                    totals[name].update({"tp": counts[0], "fp": counts[1], "fn": counts[2]})
                source_id = record_by_id[sample_id]["source_id"]
                per_source[source_id].update({"tp": iou_counts[0], "fp": iou_counts[1], "fn": iou_counts[2]})

                for class_index, class_name in enumerate(classes):
                    class_predictions = [item for item in predictions if item["class_index"] == class_index]
                    class_truths = [item for item in truths if item["class_index"] == class_index]
                    counts = match_items(class_predictions, class_truths, "iou30")
                    per_class[class_name].update({"tp": counts[0], "fp": counts[1], "fn": counts[2]})

                sample_counter = Counter(tp=iou_counts[0], fp=iou_counts[1], fn=iou_counts[2])
                sample_rows.append(
                    {
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "metrics": metric(sample_counter),
                        "predictions": predictions,
                        "truths": truths,
                    }
                )

    report = {
        "split": args.split,
        "threshold": args.threshold,
        "iou30": metric(totals["iou30"]),
        "center6": metric(totals["center6"]),
        "per_source_iou30": {key: metric(value) for key, value in sorted(per_source.items())},
        "per_class_iou30": {key: metric(value) for key, value in sorted(per_class.items())},
    }
    report_path = model_root / f"{args.split}_detailed_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    evaluation_root = database / "tab_detector" / "evaluation" / args.split
    evaluation_root.mkdir(parents=True, exist_ok=True)
    for old in evaluation_root.glob("*.png"):
        old.unlink()
    sample_rows.sort(key=lambda row: (row["metrics"]["f1"], -row["metrics"]["fn"], -row["metrics"]["fp"]))
    for rank, row in enumerate(sample_rows[: args.worst], start=1):
        record = record_by_id[row["sample_id"]]
        output_path = evaluation_root / f"{rank:02d}_{row['sample_id']}.png"
        draw_overlay(database / record["image"], output_path, row["predictions"], row["truths"], classes)

    print(json.dumps(report))


if __name__ == "__main__":
    main()
