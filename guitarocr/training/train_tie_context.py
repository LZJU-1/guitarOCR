from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from guitarocr.models.rhythm_context_model import RhythmContextCNN
from guitarocr.models.tie_context_model import (
    COUNT_CLASSES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    TieContextCNN,
    Y_BINS,
    parameter_count,
)
from guitarocr.paths import DATABASE_ROOT


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TieEventDataset(Dataset):
    def __init__(self, database: Path, manifest: Path, augment: bool = False):
        self.database = database
        self.records = read_jsonl(manifest)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def augment_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.7:
            tensor = (
                tensor * random.uniform(0.82, 1.18) + random.uniform(-0.05, 0.05)
            ).clamp(0.0, 1.0)
        if random.random() < 0.5:
            tensor = (
                tensor + torch.randn_like(tensor) * random.uniform(0.002, 0.018)
            ).clamp(0.0, 1.0)
        if random.random() < 0.35:
            shift = random.randint(-3, 3)
            if shift:
                tensor = torch.roll(tensor, shifts=shift, dims=2)
                if shift > 0:
                    tensor[:, :, :shift] = 0.0
                else:
                    tensor[:, :, shift:] = 0.0
        return tensor

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(self.database / record["image"]) as opened:
            image = opened.convert("L")
        if image.size != (INPUT_WIDTH, INPUT_HEIGHT):
            raise ValueError(f"Unexpected image size {image.size} for {record['sample_id']}")
        tensor = torch.from_numpy(1.0 - np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
        if self.augment:
            tensor = self.augment_tensor(tensor)
        y_target = torch.zeros(Y_BINS, dtype=torch.float32)
        y_target[record["tied_y_bins"]] = 1.0
        return (
            tensor,
            torch.tensor(int(record["tie_present"]), dtype=torch.long),
            torch.tensor(min(6, int(record["tied_note_count"])), dtype=torch.long),
            torch.tensor(min(6, int(record["score_note_count"])), dtype=torch.long),
            y_target,
        )


def class_weights(values: list[int], classes: int, cap: float = 10.0) -> torch.Tensor:
    counts = Counter(values)
    baseline = max(counts.values())
    return torch.tensor(
        [
            0.0 if counts.get(index, 0) == 0 else min(cap, math.sqrt(baseline / counts[index]))
            for index in range(classes)
        ],
        dtype=torch.float32,
    )


def load_pretrained_rhythm(model: TieContextCNN, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    rhythm = RhythmContextCNN()
    rhythm.load_state_dict(checkpoint["model_state"])
    model.backbone.load_state_dict(rhythm.backbone.state_dict())
    model.context.load_state_dict(rhythm.context.state_dict())


def calculate_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    presence: torch.Tensor,
    counts: torch.Tensor,
    note_counts: torch.Tensor,
    y_targets: torch.Tensor,
    presence_weights: torch.Tensor,
    count_weights: torch.Tensor,
    note_count_weights: torch.Tensor,
    y_positive_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    presence_loss = nn.functional.cross_entropy(outputs[0], presence, weight=presence_weights)
    count_loss = nn.functional.cross_entropy(outputs[1], counts, weight=count_weights)
    note_count_loss = nn.functional.cross_entropy(outputs[2], note_counts, weight=note_count_weights)
    y_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs[3], y_targets, pos_weight=y_positive_weights
    )
    total = presence_loss + 0.45 * count_loss + 0.45 * note_count_loss + 0.65 * y_loss
    return total, {
        "presence": float(presence_loss.detach()),
        "tie_count": float(count_loss.detach()),
        "note_count": float(note_count_loss.detach()),
        "y": float(y_loss.detach()),
    }


def presence_metrics(probabilities: list[float], targets: list[int], threshold: float) -> dict:
    predictions = [int(probability >= threshold) for probability in probabilities]
    true_positive = sum(prediction == 1 and target == 1 for prediction, target in zip(predictions, targets))
    false_positive = sum(prediction == 1 and target == 0 for prediction, target in zip(predictions, targets))
    false_negative = sum(prediction == 0 and target == 1 for prediction, target in zip(predictions, targets))
    true_negative = sum(prediction == 0 and target == 0 for prediction, target in zip(predictions, targets))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "accuracy": (true_positive + true_negative) / max(1, len(targets)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "positive_support": sum(targets),
        "event_support": len(targets),
    }


def tune_threshold(probabilities: list[float], targets: list[int]) -> tuple[float, dict]:
    candidates = [value / 100.0 for value in range(5, 96)]
    scored = [(presence_metrics(probabilities, targets, threshold), threshold) for threshold in candidates]
    metrics, threshold = max(
        scored,
        key=lambda item: (item[0]["f1"], item[0]["precision"], item[0]["recall"], item[1]),
    )
    return threshold, metrics


def decode_y_bins(logits: torch.Tensor, count: int) -> list[int]:
    if count <= 0:
        return []
    probabilities = logits.sigmoid().tolist()
    ranked = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)
    selected: list[int] = []
    for candidate in ranked:
        if all(abs(candidate - existing) > 1 for existing in selected):
            selected.append(candidate)
            if len(selected) >= count:
                break
    return sorted(selected)


def match_y_bins(predicted: list[int], truth: list[int], tolerance: int = 1) -> int:
    unused = set(predicted)
    matched = 0
    for target in truth:
        candidates = [value for value in unused if abs(value - target) <= tolerance]
        if candidates:
            selected = min(candidates, key=lambda value: abs(value - target))
            unused.remove(selected)
            matched += 1
    return matched


@torch.inference_mode()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    result = {
        "probabilities": [], "presence": [], "count_predictions": [], "counts": [],
        "note_count_predictions": [], "note_counts": [], "y_logits": [], "y": [],
    }
    for images, presence, counts, note_counts, y_targets in loader:
        outputs = model(images.to(device, non_blocking=True))
        result["probabilities"].extend(outputs[0].softmax(dim=1)[:, 1].cpu().tolist())
        result["presence"].extend(presence.tolist())
        result["count_predictions"].extend(outputs[1].argmax(dim=1).cpu().tolist())
        result["counts"].extend(counts.tolist())
        result["note_count_predictions"].extend(outputs[2].argmax(dim=1).cpu().tolist())
        result["note_counts"].extend(note_counts.tolist())
        result["y_logits"].extend(outputs[3].cpu())
        result["y"].extend(y_targets.tolist())
    return result


def full_metrics(predictions: dict, threshold: float) -> dict:
    presence = presence_metrics(predictions["probabilities"], predictions["presence"], threshold)
    positive_indices = [index for index, target in enumerate(predictions["presence"]) if target]
    count_correct = sum(
        predictions["count_predictions"][index] == predictions["counts"][index]
        for index in positive_indices
    )
    note_count_correct = sum(
        prediction == truth
        for prediction, truth in zip(
            predictions["note_count_predictions"], predictions["note_counts"]
        )
    )
    positive_note_count_correct = sum(
        predictions["note_count_predictions"][index] == predictions["note_counts"][index]
        for index in positive_indices
    )
    y_true = y_predicted = y_matched = 0
    positive_event_exact = 0
    for index in range(len(predictions["presence"])):
        truth_bins = [bin_index for bin_index, value in enumerate(predictions["y"][index]) if value > 0.5]
        if predictions["probabilities"][index] < threshold:
            decoded: list[int] = []
        else:
            predicted_count = max(1, int(predictions["count_predictions"][index]))
            decoded = decode_y_bins(predictions["y_logits"][index], predicted_count)
        matched = match_y_bins(decoded, truth_bins)
        y_true += len(truth_bins)
        y_predicted += len(decoded)
        y_matched += matched
        if predictions["presence"][index]:
            positive_event_exact += int(
                predictions["probabilities"][index] >= threshold
                and predictions["count_predictions"][index] == predictions["counts"][index]
                and matched == len(truth_bins) == len(decoded)
            )
    y_precision = y_matched / max(1, y_predicted)
    y_recall = y_matched / max(1, y_true)
    return {
        "presence": presence,
        "positive_count_accuracy": count_correct / max(1, len(positive_indices)),
        "positive_count_correct": count_correct,
        "positive_count_support": len(positive_indices),
        "score_note_count_accuracy": note_count_correct / max(1, len(predictions["presence"])),
        "positive_score_note_count_accuracy": positive_note_count_correct / max(1, len(positive_indices)),
        "target_y": {
            "precision": y_precision,
            "recall": y_recall,
            "f1": 2 * y_precision * y_recall / max(1e-12, y_precision + y_recall),
            "matched": y_matched,
            "predicted": y_predicted,
            "truth": y_true,
            "tolerance_bins": 1,
            "bin_pixels": INPUT_HEIGHT / Y_BINS,
        },
        "positive_event_count_and_y_exact": positive_event_exact / max(1, len(positive_indices)),
        "positive_event_count_and_y_exact_count": positive_event_exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the compact event-centred tie relationship CNN.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--init-checkpoint", type=Path, help="Optional compatible tie checkpoint for fine-tuning.")
    parser.add_argument("--fresh", action="store_true", help="Initialize only from the rhythm CNN, not an older tie model.")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    database = args.database.resolve()
    manifests = database / "tie_events" / "manifests"
    train_dataset = TieEventDataset(database, manifests / "train.jsonl", augment=True)
    validation_dataset = TieEventDataset(database, manifests / "validation.jsonl")
    test_dataset = TieEventDataset(database, manifests / "test.jsonl")
    positives = sum(int(record["tie_present"]) for record in train_dataset.records)
    negatives = len(train_dataset) - positives
    positive_weight = min(14.0, negatives / max(1, positives))
    sample_weights = [
        1.0 + (positive_weight if record["tie_present"] else 0.0)
        + 0.5 * max(0, int(record["tied_note_count"]) - 1)
        for record in train_dataset.records
    ]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_args)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TieContextCNN().to(device)
    rhythm_checkpoint = database / "rhythm_events" / "models" / "rhythm_context_cnn.pt"
    load_pretrained_rhythm(model, rhythm_checkpoint, device)
    previous_tie_checkpoint = database / "tie_events" / "models" / "tie_context_cnn.pt"
    initialization = str(rhythm_checkpoint)
    fine_tune_checkpoint = args.init_checkpoint or previous_tie_checkpoint
    if fine_tune_checkpoint.is_file() and not args.fresh:
        previous = torch.load(fine_tune_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(previous["model_state"], strict=False)
        initialization = f"{rhythm_checkpoint}; fine-tuned from {fine_tune_checkpoint}"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    presence_values = [int(record["tie_present"]) for record in train_dataset.records]
    count_values = [min(6, int(record["tied_note_count"])) for record in train_dataset.records]
    note_count_values = [min(6, int(record["score_note_count"])) for record in train_dataset.records]
    presence_weights = class_weights(presence_values, 2).to(device)
    count_weights = class_weights(count_values, len(COUNT_CLASSES)).to(device)
    note_count_weights = class_weights(note_count_values, len(COUNT_CLASSES)).to(device)
    bin_counts = np.zeros(Y_BINS, dtype=np.int64)
    for record in train_dataset.records:
        bin_counts[record["tied_y_bins"]] += 1
    y_positive_weights = torch.tensor(
        [min(20.0, math.sqrt((len(train_dataset) - count) / max(1, count))) for count in bin_counts],
        dtype=torch.float32,
        device=device,
    )

    model_root = database / "tie_events" / "models"
    report_root = database / "tie_events" / "reports"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_root / "tie_context_cnn.pt"
    history: list[dict] = []
    best_score = -1.0
    best_epoch = 0
    started = time.perf_counter()
    print(json.dumps({
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameters": parameter_count(model),
        "train": len(train_dataset),
        "validation": len(validation_dataset),
        "test": len(test_dataset),
        "train_positive": positives,
        "initialization": initialization,
    }, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        components = Counter()
        for images, presence, counts, note_counts, y_targets in train_loader:
            images = images.to(device, non_blocking=True)
            presence = presence.to(device, non_blocking=True)
            counts = counts.to(device, non_blocking=True)
            note_counts = note_counts.to(device, non_blocking=True)
            y_targets = y_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, loss_parts = calculate_loss(
                    model(images), presence, counts, note_counts, y_targets,
                    presence_weights, count_weights, note_count_weights, y_positive_weights,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            batch = images.shape[0]
            running_loss += float(loss) * batch
            seen += batch
            for key, value in loss_parts.items():
                components[key] += value * batch
        scheduler.step()
        validation_predictions = collect_predictions(model, validation_loader, device)
        threshold, threshold_metrics = tune_threshold(
            validation_predictions["probabilities"], validation_predictions["presence"]
        )
        validation_metrics = full_metrics(validation_predictions, threshold)
        score = validation_metrics["presence"]["f1"] + 0.15 * validation_metrics["target_y"]["f1"]
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "loss_components": {key: value / max(1, seen) for key, value in components.items()},
            "learning_rate": scheduler.get_last_lr()[0],
            "validation_presence": threshold_metrics,
            "validation_target_y_f1": validation_metrics["target_y"]["f1"],
            "validation_count_accuracy": validation_metrics["positive_count_accuracy"],
            "validation_score_note_count_accuracy": validation_metrics["score_note_count_accuracy"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
                    "y_bins": Y_BINS,
                    "count_classes": COUNT_CLASSES,
                    "presence_threshold": threshold,
                    "parameter_count": parameter_count(model),
                    "best_epoch": best_epoch,
                    "best_validation_score": best_score,
                    "scope": "Event-centred real TuxGuitar score_tab PDF crops; source-disjoint tie split.",
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    threshold = float(checkpoint["presence_threshold"])
    validation_metrics = full_metrics(collect_predictions(model, validation_loader, device), threshold)
    test_metrics = full_metrics(collect_predictions(model, test_loader, device), threshold)
    report = {
        "best_epoch": best_epoch,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_count": parameter_count(model),
        "presence_threshold": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "scope_warning": (
            f"Test has only {sum(record['tie_present'] for record in test_dataset.records)} positive tie events. "
            "Presence/count/y are recognized, but mapping y positions "
            "to exact string/fret edges and cross-system continuation still require sequence association."
        ),
    }
    (model_root / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_root / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    model.eval()
    traced = torch.jit.trace(model, torch.zeros(1, 1, INPUT_HEIGHT, INPUT_WIDTH, device=device))
    traced.save(str(model_root / "tie_context_cnn.torchscript.pt"))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
