from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from guitarocr.models.fret_token_model import CLASSES, FretTokenCNN
from guitarocr.paths import DATABASE_ROOT


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def ensure_packed(database: Path, root: Path, split: str, records: list[dict]) -> tuple[Path, Path]:
    packed_root = root / "packed"
    packed_root.mkdir(parents=True, exist_ok=True)
    image_path = packed_root / f"{split}_images.npy"
    label_path = packed_root / f"{split}_labels.npy"
    if image_path.is_file() and label_path.is_file():
        labels = np.load(label_path, mmap_mode="r")
        images = np.load(image_path, mmap_mode="r")
        if len(labels) == len(records) and len(images) == len(records):
            return image_path, label_path
    images = np.lib.format.open_memmap(
        image_path, mode="w+", dtype=np.uint8, shape=(len(records), 32, 64)
    )
    labels = np.empty(len(records), dtype=np.int64)
    for index, record in enumerate(records):
        with Image.open(database / record["image"]) as opened:
            images[index] = np.asarray(opened.convert("L"), dtype=np.uint8)
        labels[index] = int(record["class_index"])
        if (index + 1) % 10000 == 0:
            print(json.dumps({"packing": split, "done": index + 1, "total": len(records)}), flush=True)
    images.flush()
    np.save(label_path, labels)
    return image_path, label_path


class TokenDataset(Dataset):
    def __init__(self, image_path: Path, label_path: Path, augment: bool) -> None:
        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        array = np.asarray(self.images[index], dtype=np.float32)
        if self.augment:
            dx = random.randint(-4, 4)
            dy = random.randint(-2, 2)
            shifted = np.full_like(array, 255.0)
            source_x0, source_x1 = max(0, -dx), min(array.shape[1], array.shape[1] - dx)
            source_y0, source_y1 = max(0, -dy), min(array.shape[0], array.shape[0] - dy)
            target_x0, target_x1 = source_x0 + dx, source_x1 + dx
            target_y0, target_y1 = source_y0 + dy, source_y1 + dy
            shifted[target_y0:target_y1, target_x0:target_x1] = array[
                source_y0:source_y1, source_x0:source_x1
            ]
            contrast = random.uniform(0.85, 1.18)
            array = np.clip(255.0 + (shifted - 255.0) * contrast, 0.0, 255.0)
        tensor = torch.from_numpy(1.0 - array / 255.0).unsqueeze(0)
        return tensor, int(self.labels[index])


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    confusion = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    for images, targets in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        for target, prediction in zip(targets, predictions):
            confusion[int(target), int(prediction)] += 1
    total = int(confusion.sum())
    correct = int(confusion.diag().sum())
    expected_notes = int(confusion[1:].sum())
    predicted_notes = int(confusion[:, 1:].sum())
    correct_notes = int(confusion.diag()[1:].sum())
    precision = correct_notes / predicted_notes if predicted_notes else 0.0
    recall = correct_notes / expected_notes if expected_notes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    present_recalls = []
    for class_index in range(1, len(CLASSES)):
        support = int(confusion[class_index].sum())
        if support:
            present_recalls.append(int(confusion[class_index, class_index]) / support)
    return {
        "token_accuracy": correct / total if total else 0.0,
        "note_precision": precision,
        "note_recall": recall,
        "note_f1": f1,
        "note_macro_recall_present": sum(present_recalls) / max(1, len(present_recalls)),
        "expected_notes": expected_notes,
        "predicted_notes": predicted_notes,
        "correct_notes": correct_notes,
        "blank_false_positive_rate": (
            int(confusion[0, 1:].sum()) / max(1, int(confusion[0].sum()))
        ),
        "confusion": confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the compact event-conditioned fret-token CNN.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    database = args.database.resolve()
    root = database / "fret_token"
    train_records = read_jsonl(root / "manifests" / "train.jsonl")
    validation_records = read_jsonl(root / "manifests" / "validation.jsonl")
    if not train_records or not validation_records:
        raise ValueError("Both train and validation fret-token manifests must be non-empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images, train_labels = ensure_packed(database, root, "train", train_records)
    validation_images, validation_labels = ensure_packed(
        database, root, "validation", validation_records
    )
    train_loader = DataLoader(
        TokenDataset(train_images, train_labels, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        TokenDataset(validation_images, validation_labels, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    counts = Counter(int(record["class_index"]) for record in train_records)
    largest = max(counts.values())
    weights = []
    for class_index in range(len(CLASSES)):
        count = counts.get(class_index, 0)
        weights.append(min(6.0, (largest / max(1, count)) ** 0.5) if count else 0.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    model = FretTokenCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model_root = root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss) * len(targets)
            seen += len(targets)
        scheduler.step()
        metrics = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "learning_rate": scheduler.get_last_lr()[0],
            **{key: value for key, value in metrics.items() if key != "confusion"},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        score = metrics["note_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": CLASSES,
                    "input_size": [64, 32],
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "best_epoch": epoch,
                    "validation_metrics": metrics,
                    "training_class_counts": dict(counts),
                    "renderer_domains": ["guitarpro8", "tuxguitar"],
                },
                model_root / "fret_token_cnn.pt",
            )
    report = {
        "device": str(device),
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "best_epoch": best_epoch,
        "best_note_f1": best_score,
        "history": history,
    }
    (model_root / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}))


if __name__ == "__main__":
    main()
