from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from guitarocr.models.rhythm_context_model import (
    DIVISION_CLASSES,
    DOT_CLASSES,
    DURATION_CLASSES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    STATE_CLASSES,
    RhythmContextCNN,
    parameter_count,
)
from guitarocr.paths import DATABASE_ROOT


STATE_INDEX = {value: index for index, value in enumerate(STATE_CLASSES)}
DURATION_INDEX = {value: index for index, value in enumerate(DURATION_CLASSES)}
DOT_INDEX = {value: index for index, value in enumerate(DOT_CLASSES)}
DIVISION_INDEX = {value: index for index, value in enumerate(DIVISION_CLASSES)}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class RhythmEventDataset(Dataset):
    def __init__(self, database: Path, manifest_path: Path, augment: bool = False):
        self.database = database
        self.records = read_jsonl(manifest_path)
        self.augment = augment
        self.targets: list[list[dict]] = []
        for record in self.records:
            label = json.loads((database / record["label"]).read_text(encoding="utf-8"))
            self.targets.append(label["voices"])

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.7:
            contrast = random.uniform(0.82, 1.18)
            brightness = random.uniform(-0.05, 0.05)
            tensor = (tensor * contrast + brightness).clamp(0.0, 1.0)
        if random.random() < 0.5:
            tensor = (tensor + torch.randn_like(tensor) * random.uniform(0.002, 0.018)).clamp(0.0, 1.0)
        if random.random() < 0.35:
            shift = random.randint(-3, 3)
            if shift:
                tensor = torch.roll(tensor, shifts=shift, dims=2)
                if shift > 0:
                    tensor[:, :, :shift] = 0.0
                else:
                    tensor[:, :, shift:] = 0.0
        return tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(self.database / record["image"]) as opened:
            image = opened.convert("L")
        if image.size != (INPUT_WIDTH, INPUT_HEIGHT):
            raise ValueError(f"Unexpected image size {image.size} for {record['sample_id']}")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(1.0 - array).unsqueeze(0)
        if self.augment:
            tensor = self._augment(tensor)

        encoded: list[int] = []
        for voice in self.targets[index]:
            state = STATE_INDEX[voice["state"]]
            duration = DURATION_INDEX.get(int(voice["duration_value"]), 0)
            dot = DOT_INDEX.get(voice["dot"], 0)
            division = DIVISION_INDEX.get(voice["division"], 0)
            encoded.extend([state, duration, dot, division])
        return tensor, torch.tensor(encoded, dtype=torch.long)


def capped_inverse_sqrt_weights(values: list[int], classes: int, cap: float = 8.0) -> torch.Tensor:
    counts = Counter(values)
    nonzero = [count for count in counts.values() if count > 0]
    baseline = max(nonzero) if nonzero else 1
    weights = []
    for class_index in range(classes):
        count = counts.get(class_index, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(min(cap, math.sqrt(baseline / count)))
    return torch.tensor(weights, dtype=torch.float32)


def build_loss_weights(dataset: RhythmEventDataset, device: torch.device) -> list[torch.Tensor]:
    weights: list[torch.Tensor] = []
    for voice_index in range(2):
        states = [STATE_INDEX[target[voice_index]["state"]] for target in dataset.targets]
        visible = [target[voice_index] for target in dataset.targets if target[voice_index]["state"] != "empty"]
        durations = [DURATION_INDEX[int(voice["duration_value"])] for voice in visible]
        dots = [DOT_INDEX[voice["dot"]] for voice in visible]
        divisions = [DIVISION_INDEX[voice["division"]] for voice in visible]
        weights.extend(
            [
                capped_inverse_sqrt_weights(states, len(STATE_CLASSES)).to(device),
                capped_inverse_sqrt_weights(durations, len(DURATION_CLASSES)).to(device),
                capped_inverse_sqrt_weights(dots, len(DOT_CLASSES)).to(device),
                capped_inverse_sqrt_weights(divisions, len(DIVISION_CLASSES)).to(device),
            ]
        )
    return weights


def calculate_loss(
    outputs: tuple[torch.Tensor, ...],
    targets: torch.Tensor,
    loss_weights: list[torch.Tensor],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for voice_index in range(2):
        offset = voice_index * 4
        state_targets = targets[:, offset]
        losses.append(nn.functional.cross_entropy(outputs[offset], state_targets, weight=loss_weights[offset]))
        visible = state_targets != STATE_INDEX["empty"]
        if visible.any():
            for task_index in range(1, 4):
                head_index = offset + task_index
                losses.append(
                    nn.functional.cross_entropy(
                        outputs[head_index][visible],
                        targets[visible, head_index],
                        weight=loss_weights[head_index],
                    )
                )
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    task_correct = [0] * 8
    task_total = [0] * 8
    visible_exact = [0, 0]
    visible_total = [0, 0]
    event_exact = 0
    event_total = 0
    duration_support: Counter[int] = Counter()
    duration_correct: Counter[int] = Counter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        predictions = torch.stack([output.argmax(dim=1) for output in outputs], dim=1)
        batch_exact = torch.ones(targets.shape[0], dtype=torch.bool, device=device)

        for voice_index in range(2):
            offset = voice_index * 4
            state_visible = targets[:, offset] != STATE_INDEX["empty"]
            state_match = predictions[:, offset] == targets[:, offset]
            task_correct[offset] += int(state_match.sum())
            task_total[offset] += targets.shape[0]
            voice_exact = state_match.clone()
            for task_index in range(1, 4):
                head_index = offset + task_index
                matches = predictions[:, head_index] == targets[:, head_index]
                task_correct[head_index] += int(matches[state_visible].sum())
                task_total[head_index] += int(state_visible.sum())
                voice_exact &= (~state_visible) | matches
            visible_exact[voice_index] += int(voice_exact[state_visible].sum())
            visible_total[voice_index] += int(state_visible.sum())
            batch_exact &= voice_exact

            visible_indices = torch.nonzero(state_visible, as_tuple=False).flatten()
            for sample_index in visible_indices.tolist():
                duration = DURATION_CLASSES[int(targets[sample_index, offset + 1])]
                duration_support[duration] += 1
                if predictions[sample_index, offset + 1] == targets[sample_index, offset + 1]:
                    duration_correct[duration] += 1

        event_exact += int(batch_exact.sum())
        event_total += targets.shape[0]

    task_names = []
    for voice_index in range(2):
        task_names.extend([f"voice_{voice_index}_state", f"voice_{voice_index}_duration",
                           f"voice_{voice_index}_dot", f"voice_{voice_index}_division"])
    metrics = {
        "event_exact_accuracy": event_exact / max(1, event_total),
        "event_count": event_total,
        "visible_voice_exact_accuracy": {
            f"voice_{voice_index}": visible_exact[voice_index] / max(1, visible_total[voice_index])
            for voice_index in range(2)
        },
        "visible_voice_support": {f"voice_{index}": visible_total[index] for index in range(2)},
        "task_accuracy": {
            task_names[index]: task_correct[index] / max(1, task_total[index])
            for index in range(8)
        },
        "task_support": {task_names[index]: task_total[index] for index in range(8)},
        "duration_accuracy": {
            str(duration): duration_correct[duration] / max(1, duration_support[duration])
            for duration in sorted(duration_support)
        },
        "duration_support": {str(duration): duration_support[duration] for duration in sorted(duration_support)},
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact event-centred TuxGuitar rhythm context CNN.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--init-checkpoint", type=Path, help="Optional compatible checkpoint for fine-tuning.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    database = args.database.resolve()
    manifest_root = database / "rhythm_events" / "manifests"
    train_dataset = RhythmEventDataset(database, manifest_root / "train.jsonl", augment=True)
    validation_dataset = RhythmEventDataset(database, manifest_root / "validation.jsonl")
    test_dataset = RhythmEventDataset(database, manifest_root / "test.jsonl")

    # Rare voice-1, rest, dotted and tuplet examples are sampled more often.
    sample_weights = []
    for targets in train_dataset.targets:
        weight = 1.0
        if targets[1]["state"] != "empty":
            weight += 10.0
        if any(voice["state"] == "rest" for voice in targets):
            weight += 3.0
        if any(voice["dot"] != "none" for voice in targets):
            weight += 2.0
        if any(voice["division"] != "1:1" for voice in targets):
            weight += 2.0
        sample_weights.append(weight)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RhythmContextCNN().to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(initial["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    loss_weights = build_loss_weights(train_dataset, device)

    model_root = database / "rhythm_events" / "models"
    report_root = database / "rhythm_events" / "reports"
    model_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_root / "rhythm_context_cnn.pt"
    history: list[dict] = []
    best_score = -1.0
    best_epoch = 0
    started = time.perf_counter()

    print(json.dumps({
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "parameters": parameter_count(model),
        "train_events": len(train_dataset),
        "validation_events": len(validation_dataset),
        "test_events": len(test_dataset),
        "initial_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
    }, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = calculate_loss(outputs, targets, loss_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss) * images.shape[0]
            sample_count += images.shape[0]
        scheduler.step()

        validation_metrics = evaluate(model, validation_loader, device)
        score = validation_metrics["visible_voice_exact_accuracy"]["voice_0"]
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, sample_count),
            "learning_rate": scheduler.get_last_lr()[0],
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "classes": {
                    "state": STATE_CLASSES,
                    "duration": DURATION_CLASSES,
                    "dot": DOT_CLASSES,
                    "division": DIVISION_CLASSES,
                },
                "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
                "parameter_count": parameter_count(model),
                "best_epoch": best_epoch,
                "best_validation_visible_voice_0_exact": best_score,
                "scope": "Event crops use ground-truth score_tab event centers.",
            }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation_metrics = evaluate(model, validation_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - started
    final_report = {
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed,
        "parameter_count": parameter_count(model),
        "validation": validation_metrics,
        "test": test_metrics,
        "scope_warning": (
            "Event crops use TuxGuitar ground-truth event centers. This evaluates rhythm context "
            "classification, not yet pixel-only event localization or full PDF-to-GP reconstruction."
        ),
    }
    (model_root / "metrics.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_root / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    model.eval()
    example = torch.zeros(1, 1, INPUT_HEIGHT, INPUT_WIDTH, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(model_root / "rhythm_context_cnn.torchscript.pt"))
    print(json.dumps(final_report, ensure_ascii=False))


if __name__ == "__main__":
    main()
