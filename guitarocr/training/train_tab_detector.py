from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import DATABASE_ROOT


INPUT_WIDTH = 512
INPUT_HEIGHT = 128
OUTPUT_STRIDE = 4
MAX_OBJECTS = 256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def draw_gaussian(heatmap: np.ndarray, center_x: int, center_y: int, radius: int) -> None:
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    coordinates = np.arange(diameter, dtype=np.float32) - radius
    gaussian = np.exp(-(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / (2 * sigma * sigma))
    height, width = heatmap.shape
    left, right = min(center_x, radius), min(width - center_x - 1, radius)
    top, bottom = min(center_y, radius), min(height - center_y - 1, radius)
    if min(left, right, top, bottom) < 0:
        return
    target = heatmap[center_y - top : center_y + bottom + 1, center_x - left : center_x + right + 1]
    kernel = gaussian[radius - top : radius + bottom + 1, radius - left : radius + right + 1]
    np.maximum(target, kernel, out=target)


class TabDetectorDataset(Dataset):
    def __init__(self, database: Path, manifest: Path, classes: list[str], training: bool) -> None:
        self.database = database
        self.records = read_jsonl(manifest)
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.class_count = len(classes)
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        with Image.open(self.database / record["image"]) as opened:
            image = opened.convert("L")
        if self.training:
            image = self.photometric_augmentation(image)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5

        label = json.loads((self.database / record["label"]).read_text(encoding="utf-8"))
        output_height = INPUT_HEIGHT // OUTPUT_STRIDE
        output_width = INPUT_WIDTH // OUTPUT_STRIDE
        heatmap = np.zeros((self.class_count, output_height, output_width), dtype=np.float32)
        sizes = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)
        offsets = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)
        indices = np.zeros(MAX_OBJECTS, dtype=np.int64)
        masks = np.zeros(MAX_OBJECTS, dtype=np.float32)
        classes = np.zeros(MAX_OBJECTS, dtype=np.int64)
        boxes = np.zeros((MAX_OBJECTS, 4), dtype=np.float32)

        symbols = label["symbols"][:MAX_OBJECTS]
        for object_index, symbol in enumerate(symbols):
            x, y, width, height = map(float, symbol["bbox"])
            center_x = (x + width / 2.0) / OUTPUT_STRIDE
            center_y = (y + height / 2.0) / OUTPUT_STRIDE
            center_int_x = int(center_x)
            center_int_y = int(center_y)
            if not (0 <= center_int_x < output_width and 0 <= center_int_y < output_height):
                continue
            class_index = self.class_to_index[symbol["class"]]
            radius = max(1, min(3, round(min(width, height) / OUTPUT_STRIDE / 2.0)))
            draw_gaussian(heatmap[class_index], center_int_x, center_int_y, radius)
            sizes[object_index] = [width / OUTPUT_STRIDE, height / OUTPUT_STRIDE]
            offsets[object_index] = [center_x - center_int_x, center_y - center_int_y]
            indices[object_index] = center_int_y * output_width + center_int_x
            masks[object_index] = 1.0
            classes[object_index] = class_index
            boxes[object_index] = [x, y, width, height]

        return {
            "image": tensor,
            "heatmap": torch.from_numpy(heatmap),
            "sizes": torch.from_numpy(sizes),
            "offsets": torch.from_numpy(offsets),
            "indices": torch.from_numpy(indices),
            "masks": torch.from_numpy(masks),
            "classes": torch.from_numpy(classes),
            "boxes": torch.from_numpy(boxes),
            "sample_id": record["sample_id"],
        }

    @staticmethod
    def photometric_augmentation(image: Image.Image) -> Image.Image:
        if random.random() < 0.8:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.3))
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.08))
        if random.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.65)))
        if random.random() < 0.35:
            pixels = np.asarray(image, dtype=np.float32)
            pixels += np.random.normal(0.0, random.uniform(1.0, 5.0), pixels.shape)
            image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")
        return image


def focal_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    positives = targets.eq(1).float()
    negatives = targets.lt(1).float()
    negative_weights = (1 - targets).pow(4)
    positive_loss = torch.log(predictions) * (1 - predictions).pow(2) * positives
    negative_loss = torch.log(1 - predictions) * predictions.pow(2) * negative_weights * negatives
    positive_count = positives.sum()
    if positive_count.item() == 0:
        return -negative_loss.sum()
    return -(positive_loss.sum() + negative_loss.sum()) / positive_count


def gather_head(head: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, channels, _, _ = head.shape
    flattened = head.view(batch, channels, -1).permute(0, 2, 1)
    expanded = indices.unsqueeze(-1).expand(-1, -1, channels)
    return flattened.gather(1, expanded)


def detector_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    heatmap_logits, size_head, offset_head = outputs
    heatmap_loss = focal_heatmap_loss(heatmap_logits, batch["heatmap"])
    masks = batch["masks"].unsqueeze(-1)
    denominator = masks.sum().clamp_min(1.0)
    predicted_sizes = gather_head(size_head, batch["indices"])
    predicted_offsets = gather_head(offset_head, batch["indices"])
    size_loss = (F.l1_loss(predicted_sizes, batch["sizes"], reduction="none") * masks).sum() / denominator
    offset_loss = (
        F.l1_loss(predicted_offsets, batch["offsets"], reduction="none") * masks
    ).sum() / denominator
    total = heatmap_loss + 0.1 * size_loss + offset_loss
    return total, {
        "heatmap": float(heatmap_loss.detach()),
        "size": float(size_loss.detach()),
        "offset": float(offset_loss.detach()),
    }


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def decode_detections(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], threshold: float = 0.25, top_k: int = 256
) -> list[list[dict]]:
    heatmap_logits, size_head, offset_head = outputs
    heatmap = heatmap_logits.sigmoid()
    heatmap = heatmap * (F.max_pool2d(heatmap, 3, stride=1, padding=1) == heatmap)
    batch_size, class_count, output_height, output_width = heatmap.shape
    results: list[list[dict]] = []
    for batch_index in range(batch_size):
        flat = heatmap[batch_index].reshape(-1)
        count = min(top_k, flat.numel())
        scores, flat_indices = torch.topk(flat, count)
        detections: list[dict] = []
        for score, flat_index in zip(scores.tolist(), flat_indices.tolist()):
            if score < threshold:
                break
            class_index = flat_index // (output_height * output_width)
            spatial = flat_index % (output_height * output_width)
            center_y, center_x = divmod(spatial, output_width)
            width, height = size_head[batch_index, :, center_y, center_x].clamp_min(0.1).tolist()
            offset_x, offset_y = offset_head[batch_index, :, center_y, center_x].tolist()
            center_input_x = (center_x + offset_x) * OUTPUT_STRIDE
            center_input_y = (center_y + offset_y) * OUTPUT_STRIDE
            width *= OUTPUT_STRIDE
            height *= OUTPUT_STRIDE
            detections.append(
                {
                    "class_index": class_index,
                    "score": score,
                    "bbox": [center_input_x - width / 2, center_input_y - height / 2, width, height],
                }
            )
        results.append(detections)
    return results


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.25
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    true_positive = false_positive = false_negative = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            outputs = model(batch["image"])
            loss, _ = detector_loss(outputs, batch)
            total_loss += float(loss) * batch["image"].size(0)
            decoded = decode_detections(outputs, threshold=threshold)
            for batch_index, predictions in enumerate(decoded):
                object_count = int(batch["masks"][batch_index].sum().item())
                truths = [
                    {
                        "class_index": int(batch["classes"][batch_index, index].item()),
                        "bbox": batch["boxes"][batch_index, index].tolist(),
                    }
                    for index in range(object_count)
                ]
                used: set[int] = set()
                for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
                    best_index = -1
                    best_iou = 0.0
                    for truth_index, truth in enumerate(truths):
                        if truth_index in used or truth["class_index"] != prediction["class_index"]:
                            continue
                        iou = box_iou(prediction["bbox"], truth["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_index = truth_index
                    if best_index >= 0 and best_iou >= 0.3:
                        used.add(best_index)
                        true_positive += 1
                    else:
                        false_positive += 1
                false_negative += len(truths) - len(used)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "loss": total_loss / len(loader.dataset),
        "precision_iou30": precision,
        "recall_iou30": recall,
        "f1_iou30": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the compact TuxGuitar TAB symbol detector.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--init-checkpoint", type=Path, help="Optional compatible checkpoint for fine-tuning.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    database = args.database.resolve()
    manifest_root = database / "tab_detector" / "manifests"
    class_data = json.loads((manifest_root / "classes.json").read_text(encoding="utf-8"))
    classes = class_data["classes"]
    train_dataset = TabDetectorDataset(database, manifest_root / "train.jsonl", classes, training=True)
    validation_dataset = TabDetectorDataset(
        database, manifest_root / "validation.jsonl", classes, training=False
    )
    test_dataset = TabDetectorDataset(database, manifest_root / "test.jsonl", classes, training=False)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabSymbolDetector(len(classes)).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if initial.get("classes") != classes:
            raise ValueError("Fine-tuning checkpoint classes do not match the current dataset")
        model.load_state_dict(initial["model_state"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    model_root = database / "tab_detector" / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_root / "tab_symbol_detector.pt"
    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "classes": len(classes),
                "parameters": parameter_count,
                "train_tiles": len(train_dataset),
                "validation_tiles": len(validation_dataset),
                "test_tiles": len(test_dataset),
                "initial_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
            }
        ),
        flush=True,
    )

    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for raw_batch in train_loader:
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in raw_batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(batch["image"])
                loss, _ = detector_loss(outputs, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * batch["image"].size(0)
        scheduler.step()

        metrics = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_dataset),
            "validation_loss": metrics["loss"],
            "validation_precision_iou30": metrics["precision_iou30"],
            "validation_recall_iou30": metrics["recall_iou30"],
            "validation_f1_iou30": metrics["f1_iou30"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["f1_iou30"] > best_f1:
            best_f1 = metrics["f1_iou30"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
                    "output_stride": OUTPUT_STRIDE,
                    "parameter_count": parameter_count,
                    "best_epoch": best_epoch,
                    "best_validation_f1_iou30": best_f1,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    report = {
        "best_epoch": best_epoch,
        "best_validation_f1_iou30": best_f1,
        "test": test_metrics,
        "elapsed_seconds": time.time() - start_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": parameter_count,
        "scope_warning": "These tile metrics use ground-truth TuxGuitar measure crops; see page_end_to_end_*_metrics.json for pixel-only full-page results.",
    }
    (model_root / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (model_root / "test_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    scripted = torch.jit.script(model.eval())
    scripted.save(str(model_root / "tab_symbol_detector.torchscript.pt"))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
