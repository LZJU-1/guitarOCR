from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from guitarocr.data.build_score_rhythm_dataset import TECHNIQUE_CLASSES
from guitarocr.models.rhythm_context_model import INPUT_HEIGHT, INPUT_WIDTH
from guitarocr.models.technique_context_model import TechniqueContextCNN, parameter_count
from guitarocr.paths import DATABASE_ROOT


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TechniqueDataset(Dataset):
    def __init__(self, database: Path, manifest: Path, augment: bool = False):
        self.database = database
        self.records = read_jsonl(manifest)
        self.augment = augment
        self.targets: list[list[float]] = []
        for record in self.records:
            label = json.loads((database / record["label"]).read_text(encoding="utf-8"))
            primary = label["voices"][0]
            effects = primary.get("effects", {})
            self.targets.append([float(bool(effects.get(name, False))) for name in TECHNIQUE_CLASSES])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.database / self.records[index]["image"]) as opened:
            image = opened.convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(1.0 - array).unsqueeze(0)
        if self.augment:
            if random.random() < 0.7:
                tensor = (tensor * random.uniform(0.85, 1.15) + random.uniform(-0.04, 0.04)).clamp(0, 1)
            if random.random() < 0.4:
                tensor = (tensor + torch.randn_like(tensor) * random.uniform(0.002, 0.015)).clamp(0, 1)
        return tensor, torch.tensor(self.targets[index], dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, thresholds: torch.Tensor) -> dict:
    model.eval()
    tp = torch.zeros(len(TECHNIQUE_CLASSES), dtype=torch.long)
    fp = torch.zeros_like(tp); fn = torch.zeros_like(tp); support = torch.zeros_like(tp)
    for images, targets in loader:
        predictions = torch.sigmoid(model(images.to(device))).cpu() >= thresholds
        truth = targets.bool()
        tp += (predictions & truth).sum(0); fp += (predictions & ~truth).sum(0)
        fn += (~predictions & truth).sum(0); support += truth.sum(0)
    per_class = {}
    for index, name in enumerate(TECHNIQUE_CLASSES):
        precision = int(tp[index]) / max(1, int(tp[index] + fp[index]))
        recall = int(tp[index]) / max(1, int(tp[index] + fn[index]))
        per_class[name] = {
            "support": int(support[index]), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
        }
    macro_supported = [value["f1"] for value in per_class.values() if value["support"]]
    return {"macro_f1_supported": sum(macro_supported) / max(1, len(macro_supported)), "per_class": per_class}


@torch.no_grad()
def tune_thresholds(model: nn.Module, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval(); probabilities = []; targets = []
    for images, batch_targets in loader:
        probabilities.append(torch.sigmoid(model(images.to(device))).cpu()); targets.append(batch_targets.bool())
    probs = torch.cat(probabilities); truth = torch.cat(targets)
    thresholds = []
    for index in range(len(TECHNIQUE_CLASSES)):
        best = (0.0, 0.5)
        for threshold in torch.arange(0.10, 0.91, 0.05):
            pred = probs[:, index] >= threshold
            tp = int((pred & truth[:, index]).sum()); fp = int((pred & ~truth[:, index]).sum())
            fn = int((~pred & truth[:, index]).sum())
            precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            if f1 > best[0]: best = (f1, float(threshold))
        thresholds.append(best[1])
    return torch.tensor(thresholds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the compact event-context guitar-technique CNN")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-name", default="technique_context_cnn.pt")
    parser.add_argument("--boost-source", action="append", default=[])
    parser.add_argument("--boost-weight", type=float, default=12.0)
    parser.add_argument("--selection-source")
    parser.add_argument(
        "--freeze-existing-classes", action="store_true",
        help="Freeze the initialized backbone and restore existing classifier rows after every step.",
    )
    args = parser.parse_args()
    random.seed(20260714); np.random.seed(20260714); torch.manual_seed(20260714)
    database = args.database.resolve(); manifests = database / "rhythm_events" / "manifests"
    train = TechniqueDataset(database, manifests / "train.jsonl", True)
    validation = TechniqueDataset(database, manifests / "validation.jsonl")
    test = TechniqueDataset(database, manifests / "test.jsonl")
    positive_counts = np.asarray(train.targets).sum(axis=0)
    pos_weight = torch.tensor(
        np.minimum(30.0, (len(train) - positive_counts) / np.maximum(1.0, positive_counts)),
        dtype=torch.float32,
    )
    boosted_sources = set(args.boost_source)
    weights = [
        (1.0 + min(12.0, sum(target) * 4.0))
        * (args.boost_weight if record["source_id"] in boosted_sources else 1.0)
        for record, target in zip(train.records, train.targets)
    ]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, sampler=sampler, **kwargs)
    validation_loader = DataLoader(validation, shuffle=False, **kwargs)
    test_loader = DataLoader(test, shuffle=False, **kwargs)
    selection_loader = None
    if args.selection_source:
        selection = copy.copy(train)
        selected = [
            (record, target)
            for record, target in zip(train.records, train.targets)
            if record["source_id"] == args.selection_source
        ]
        if not selected:
            raise ValueError(f"Selection source is absent from training: {args.selection_source}")
        selection.records = [record for record, _ in selected]
        selection.targets = [target for _, target in selected]
        selection.augment = False
        selection_loader = DataLoader(selection, shuffle=False, **kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TechniqueContextCNN().to(device)
    frozen_classifier_rows: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    preserved_thresholds: dict[int, float] = {}
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        initial_classes = list(initial.get("classes", []))
        if not initial_classes:
            raise ValueError("Fine-tuning checkpoint has no technique class metadata")
        current = model.state_dict()
        for name, value in initial["model_state"].items():
            if name in current and current[name].shape == value.shape:
                current[name] = value
        # Permit adding technique classes without discarding the pretrained
        # visual backbone or the logits for classes that retain their names.
        for class_name in set(initial_classes) & set(TECHNIQUE_CLASSES):
            old_index = initial_classes.index(class_name)
            new_index = TECHNIQUE_CLASSES.index(class_name)
            current["head.4.weight"][new_index] = initial["model_state"]["head.4.weight"][old_index]
            current["head.4.bias"][new_index] = initial["model_state"]["head.4.bias"][old_index]
            if args.freeze_existing_classes:
                frozen_classifier_rows[new_index] = (
                    current["head.4.weight"][new_index].clone(),
                    current["head.4.bias"][new_index].clone(),
                )
                initial_thresholds = initial.get("thresholds", [0.5] * len(initial_classes))
                preserved_thresholds[new_index] = float(initial_thresholds[old_index])
        model.load_state_dict(current)
    if args.freeze_existing_classes:
        if not args.init_checkpoint:
            raise ValueError("--freeze-existing-classes requires --init-checkpoint")
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name in {"head.4.weight", "head.4.bias"}
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    model_root = database / "technique_events" / "models"; model_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_root / args.output_name
    best_score = -1.0; started = time.perf_counter(); history = []
    print(json.dumps({"device": str(device), "parameters": parameter_count(model), "train": len(train),
                      "positive_counts": dict(zip(TECHNIQUE_CLASSES, positive_counts.astype(int).tolist()))}))
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(images), targets)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            if frozen_classifier_rows:
                with torch.no_grad():
                    for row_index, (weight, bias) in frozen_classifier_rows.items():
                        model.head[4].weight[row_index].copy_(weight.to(device))
                        model.head[4].bias[row_index].copy_(bias.to(device))
            total_loss += float(loss) * images.shape[0]; seen += images.shape[0]
        scheduler.step(); thresholds = tune_thresholds(model, validation_loader, device)
        for row_index, value in preserved_thresholds.items():
            thresholds[row_index] = value
        metrics = evaluate(model, validation_loader, device, thresholds)
        selection_metrics = (
            evaluate(model, selection_loader, device, thresholds)
            if selection_loader is not None else None
        )
        row = {"epoch": epoch, "loss": total_loss / max(1, seen), "validation": metrics}
        if selection_metrics is not None:
            row["selection"] = selection_metrics
        history.append(row); print(json.dumps(row))
        score = (
            selection_metrics["macro_f1_supported"]
            if selection_metrics is not None else metrics["macro_f1_supported"]
        )
        if score > best_score:
            best_score = score
            torch.save({"model_state": model.state_dict(), "classes": TECHNIQUE_CLASSES,
                        "thresholds": thresholds.tolist(), "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
                        "parameter_count": parameter_count(model), "best_epoch": epoch}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"]); thresholds = torch.tensor(checkpoint["thresholds"])
    report = {"elapsed_seconds": time.perf_counter() - started, "best_epoch": checkpoint["best_epoch"],
              "validation": evaluate(model, validation_loader, device, thresholds),
              "test": evaluate(model, test_loader, device, thresholds)}
    if selection_loader is not None:
        report["selection"] = evaluate(model, selection_loader, device, thresholds)
    metrics_name = f"{Path(args.output_name).stem}_metrics.json"
    (model_root / metrics_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
