from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from guitarocr.models.symbol_model import AtomicSymbolCNN, parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact GuitarOCR atomic-symbol CNN.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-checkpoint", type=Path, help="Optional compatible checkpoint for fine-tuning.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_loader(dataset: datasets.ImageFolder, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_count: int,
    collect_predictions: bool = False,
) -> tuple[float, float, np.ndarray | None, list[tuple[int, int, float]]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = np.zeros((class_count, class_count), dtype=np.int64) if collect_predictions else None
    predictions: list[tuple[int, int, float]] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        probabilities = logits.softmax(dim=1)
        confidence, predicted = probabilities.max(dim=1)
        total_loss += loss.item() * targets.numel()
        total_correct += (predicted == targets).sum().item()
        total_count += targets.numel()
        if confusion is not None:
            for truth, guess, score in zip(targets.cpu(), predicted.cpu(), confidence.cpu()):
                confusion[int(truth), int(guess)] += 1
                predictions.append((int(truth), int(guess), float(score)))
    return total_loss / total_count, total_correct / total_count, confusion, predictions


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    train_set = datasets.ImageFolder(args.data / "train", transform=transform)
    validation_set = datasets.ImageFolder(args.data / "validation", transform=transform)
    test_set = datasets.ImageFolder(args.data / "test", transform=transform)
    if not (train_set.class_to_idx == validation_set.class_to_idx == test_set.class_to_idx):
        raise RuntimeError("Class index mappings differ between splits")

    classes = train_set.classes
    train_loader = make_loader(train_set, args.batch_size, args.workers, shuffle=True)
    validation_loader = make_loader(validation_set, args.batch_size, args.workers, shuffle=False)
    test_loader = make_loader(test_set, args.batch_size, args.workers, shuffle=False)

    model = AtomicSymbolCNN(len(classes)).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if initial.get("classes") != classes:
            raise ValueError("Fine-tuning checkpoint classes do not match the current dataset")
        model.load_state_dict(initial["model_state"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "torch": torch.__version__,
                "classes": len(classes),
                "parameters": parameter_count(model),
                "train_images": len(train_set),
                "validation_images": len(validation_set),
                "test_images": len(test_set),
                "initial_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    best_accuracy = -1.0
    best_epoch = -1
    checkpoint_path = args.output / "atomic_symbol_cnn.pt"
    history: list[dict[str, float | int]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * targets.numel()
            running_correct += (logits.argmax(dim=1) == targets).sum().item()
            running_count += targets.numel()

        validation_loss, validation_accuracy, _, _ = evaluate(
            model, validation_loader, criterion, device, len(classes)
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": running_loss / running_count,
            "train_accuracy": running_correct / running_count,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "class_to_index": train_set.class_to_idx,
                    "image_size": 64,
                    "normalization": {"mean": [0.5], "std": [0.5]},
                    "architecture": "AtomicSymbolCNN",
                    "parameters": parameter_count(model),
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                    "torch_version": torch.__version__,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_accuracy, confusion, predictions = evaluate(
        model, test_loader, criterion, device, len(classes), collect_predictions=True
    )
    assert confusion is not None
    per_class = {}
    for index, name in enumerate(classes):
        count = int(confusion[index].sum())
        correct = int(confusion[index, index])
        per_class[name] = {"correct": correct, "count": count, "accuracy": correct / count if count else 0.0}

    report = {
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": parameter_count(model),
        "per_class": per_class,
        "scope_warning": "Metrics are on held-out synthetic symbols, not automatically cropped real PDF symbols.",
    }
    (args.output / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "test_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (args.output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth\\predicted", *classes])
        for name, row in zip(classes, confusion):
            writer.writerow([name, *row.tolist()])

    misclassified = []
    for (path, _), (truth, guess, confidence) in zip(test_set.samples, predictions):
        if truth != guess:
            misclassified.append(
                {
                    "path": str(Path(path).relative_to(args.data)),
                    "truth": classes[truth],
                    "predicted": classes[guess],
                    "confidence": confidence,
                }
            )
    with (args.output / "misclassified.json").open("w", encoding="utf-8") as handle:
        json.dump(misclassified, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    model_cpu = model.to("cpu").eval()
    scripted = torch.jit.trace(model_cpu, torch.zeros(1, 1, 64, 64))
    scripted.save(str(args.output / "atomic_symbol_cnn.torchscript.pt"))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
