from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from guitarocr.models.score_event_locator_model import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    OUTPUT_STRIDE,
    ScoreEventLocator,
    parameter_count,
)
from guitarocr.paths import DATABASE_ROOT


OUTPUT_WIDTH = INPUT_WIDTH // OUTPUT_STRIDE
MAX_EVENTS = 64


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def draw_gaussian_1d(heatmap: np.ndarray, center: float, sigma: float = 1.25) -> None:
    left = max(0, int(math.floor(center - 4 * sigma)))
    right = min(heatmap.shape[0], int(math.ceil(center + 4 * sigma)) + 1)
    positions = np.arange(left, right, dtype=np.float32)
    gaussian = np.exp(-((positions - center) ** 2) / (2 * sigma * sigma))
    heatmap[left:right] = np.maximum(heatmap[left:right], gaussian)
    peak = min(heatmap.shape[0] - 1, max(0, int(center)))
    heatmap[peak] = 1.0


class ScoreEventDataset(Dataset):
    def __init__(self, database: Path, manifest: Path, augment: bool = False) -> None:
        self.database = database
        self.records = read_jsonl(manifest)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        label = json.loads((self.database / record["label"]).read_text(encoding="utf-8"))
        with Image.open(self.database / record["image"]) as opened:
            image = opened.convert("L")
        if image.size != (INPUT_WIDTH, INPUT_HEIGHT):
            raise ValueError(f"Unexpected image size for {record['sample_id']}: {image.size}")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(1.0 - array).unsqueeze(0)
        if self.augment:
            if random.random() < 0.75:
                tensor = (tensor * random.uniform(0.80, 1.20) + random.uniform(-0.04, 0.04)).clamp(0.0, 1.0)
            if random.random() < 0.45:
                tensor = (tensor + torch.randn_like(tensor) * random.uniform(0.002, 0.018)).clamp(0.0, 1.0)
            if random.random() < 0.35:
                shift = random.randint(-4, 4)
                if shift:
                    tensor = torch.roll(tensor, shifts=shift, dims=1)
                    if shift > 0:
                        tensor[:, :shift, :] = 0.0
                    else:
                        tensor[:, shift:, :] = 0.0

        heatmap = np.zeros(OUTPUT_WIDTH, dtype=np.float32)
        offsets = np.zeros(OUTPUT_WIDTH, dtype=np.float32)
        offset_mask = np.zeros(OUTPUT_WIDTH, dtype=np.float32)
        centers = np.zeros(MAX_EVENTS, dtype=np.float32)
        events = label["events"]
        if len(events) > MAX_EVENTS:
            raise ValueError(f"Too many events in {record['sample_id']}: {len(events)}")
        for event_index, event in enumerate(events):
            center_input = float(event["x"])
            center_output = center_input / OUTPUT_STRIDE
            position = min(OUTPUT_WIDTH - 1, max(0, int(center_output)))
            draw_gaussian_1d(heatmap, center_output)
            offsets[position] = center_output - position
            offset_mask[position] = 1.0
            centers[event_index] = center_input

        return {
            "image": tensor,
            "heatmap": torch.from_numpy(heatmap).unsqueeze(0),
            "offsets": torch.from_numpy(offsets).unsqueeze(0),
            "offset_mask": torch.from_numpy(offset_mask).unsqueeze(0),
            "centers": torch.from_numpy(centers),
            "event_count": torch.tensor(len(events), dtype=torch.long),
            "tolerance": torch.tensor(float(label["transform"]["staff_spacing_input"]) * 0.5),
        }


def focal_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid().clamp(1e-5, 1.0 - 1e-5)
    positive = targets.eq(1.0)
    negative = targets.lt(1.0)
    negative_weights = (1.0 - targets).pow(4)
    positive_loss = -(probabilities.log()) * (1.0 - probabilities).pow(2) * positive
    negative_loss = -((1.0 - probabilities).log()) * probabilities.pow(2) * negative_weights * negative
    positive_count = positive.sum().clamp_min(1)
    return (positive_loss.sum() + negative_loss.sum()) / positive_count


def locator_loss(
    outputs: tuple[torch.Tensor, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    heatmap_logits, offset_predictions = outputs
    heatmap = focal_heatmap_loss(heatmap_logits, batch["heatmap"])
    mask = batch["offset_mask"]
    offset = ((offset_predictions - batch["offsets"]).abs() * mask).sum() / mask.sum().clamp_min(1.0)
    total = heatmap + 0.5 * offset
    return total, {"heatmap": float(heatmap.detach()), "offset": float(offset.detach())}


def decode_events(
    outputs: tuple[torch.Tensor, torch.Tensor], threshold: float, top_k: int = MAX_EVENTS
) -> list[list[dict]]:
    heatmap_logits, offsets = outputs
    heatmap = heatmap_logits.sigmoid()
    local_maximum = nn.functional.max_pool1d(heatmap, kernel_size=3, stride=1, padding=1)
    heatmap = heatmap * heatmap.eq(local_maximum)
    results: list[list[dict]] = []
    for batch_index in range(heatmap.shape[0]):
        scores, indices = torch.topk(heatmap[batch_index, 0], min(top_k, OUTPUT_WIDTH))
        detections: list[dict] = []
        for score, position in zip(scores.tolist(), indices.tolist()):
            if score < threshold:
                break
            offset = float(offsets[batch_index, 0, position].clamp(0.0, 0.999).item())
            detections.append({"x": (position + offset) * OUTPUT_STRIDE, "score": score})
        results.append(detections)
    return results


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict:
    model.eval()
    true_positive = false_positive = false_negative = 0
    errors: list[float] = []
    total_loss = 0.0
    for raw_batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in raw_batch.items()}
        outputs = model(batch["image"])
        loss, _ = locator_loss(outputs, batch)
        total_loss += float(loss) * batch["image"].shape[0]
        predictions = decode_events(outputs, threshold)
        for batch_index, candidates in enumerate(predictions):
            count = int(batch["event_count"][batch_index])
            truths = batch["centers"][batch_index, :count].tolist()
            tolerance = float(batch["tolerance"][batch_index])
            unmatched = set(range(len(truths)))
            for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
                if not unmatched:
                    false_positive += 1
                    continue
                truth_index = min(unmatched, key=lambda item: abs(truths[item] - candidate["x"]))
                error = abs(truths[truth_index] - candidate["x"])
                if error <= tolerance:
                    unmatched.remove(truth_index)
                    true_positive += 1
                    errors.append(error)
                else:
                    false_positive += 1
            false_negative += len(unmatched)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    error_array = np.asarray(errors, dtype=np.float32)
    return {
        "threshold": threshold,
        "loss": total_loss / max(1, len(loader.dataset)),
        "precision_half_staff_spacing": precision,
        "recall_half_staff_spacing": recall,
        "f1_half_staff_spacing": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "mean_x_error_px": float(error_array.mean()) if errors else 0.0,
        "median_x_error_px": float(np.median(error_array)) if errors else 0.0,
        "p95_x_error_px": float(np.percentile(error_array, 95)) if errors else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact one-dimensional score event locator.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--init-checkpoint", type=Path, help="Optional compatible checkpoint for fine-tuning.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    database = args.database.resolve()
    manifest_root = database / "score_event_locator" / "manifests"
    train_set = ScoreEventDataset(database, manifest_root / "train.jsonl", augment=True)
    validation_set = ScoreEventDataset(database, manifest_root / "validation.jsonl")
    test_set = ScoreEventDataset(database, manifest_root / "test.jsonl")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, generator=generator,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScoreEventLocator().to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    model_root = database / "score_event_locator" / "models"
    report_root = database / "score_event_locator" / "reports"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_root / "score_event_locator.pt"

    print(json.dumps({
        "device": str(device), "torch": torch.__version__, "parameters": parameter_count(model),
        "train_tiles": len(train_set), "validation_tiles": len(validation_set), "test_tiles": len(test_set),
        "initial_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
    }))
    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        samples = 0
        for raw_batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in raw_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["image"])
            loss, _ = locator_loss(outputs, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach()) * batch["image"].shape[0]
            samples += batch["image"].shape[0]
        scheduler.step()

        validation = evaluate(model, validation_loader, device, threshold=0.25)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, samples),
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if validation["f1_half_staff_spacing"] > best_f1:
            best_f1 = validation["f1_half_staff_spacing"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_f1_at_025": best_f1,
                    "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
                    "output_stride": OUTPUT_STRIDE,
                    "scope": "Ground-truth measure/score geometry; event x-axis localisation.",
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    threshold_results = [
        evaluate(model, validation_loader, device, threshold=round(value, 2))
        for value in np.arange(0.10, 0.91, 0.05)
    ]
    calibrated = max(threshold_results, key=lambda item: (item["f1_half_staff_spacing"], item["recall_half_staff_spacing"]))
    threshold = float(calibrated["threshold"])
    test = evaluate(model, test_loader, device, threshold=threshold)
    checkpoint["detection_threshold"] = threshold
    torch.save(checkpoint, checkpoint_path)

    final_report = {
        "best_epoch": best_epoch,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": parameter_count(model),
        "calibrated_threshold": threshold,
        "validation": calibrated,
        "test": test,
        "scope_warning": (
            "Tile metrics use ground-truth measure and score-staff geometry. "
            "Pixel-only page geometry and duplicate merging require separate evaluation."
        ),
    }
    (model_root / "metrics.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_root / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    model.eval()
    traced = torch.jit.trace(model, torch.zeros(1, 1, INPUT_HEIGHT, INPUT_WIDTH, device=device))
    traced.save(str(model_root / "score_event_locator.torchscript.pt"))
    print(json.dumps(final_report, ensure_ascii=False))


if __name__ == "__main__":
    main()
