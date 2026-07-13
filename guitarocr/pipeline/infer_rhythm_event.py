from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

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
from guitarocr.paths import WEIGHTS_ROOT


TASKS = (
    ("state", STATE_CLASSES),
    ("duration", DURATION_CLASSES),
    ("dot", DOT_CLASSES),
    ("division", DIVISION_CLASSES),
)


def prepare_image(path: Path) -> tuple[torch.Tensor, tuple[int, int], bool]:
    with Image.open(path) as opened:
        image = opened.convert("L")
    original_size = image.size
    resized = original_size != (INPUT_WIDTH, INPUT_HEIGHT)
    if resized:
        image = image.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(1.0 - array).unsqueeze(0).unsqueeze(0)
    return tensor, original_size, resized


def decode_head(logits: torch.Tensor, classes: list, top_k: int) -> dict:
    probabilities = torch.softmax(logits, dim=1)[0]
    count = min(top_k, len(classes))
    values, indices = probabilities.topk(count)
    candidates = [
        {"value": classes[int(index)], "probability": round(float(value), 6)}
        for value, index in zip(values, indices)
    ]
    return {"value": candidates[0]["value"], "confidence": candidates[0]["probability"], "top": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize rhythm semantics from one 256x192 event-centred TuxGuitar score_tab crop."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=WEIGHTS_ROOT / "rhythm_context_cnn.pt",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = RhythmContextCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_tensor, original_size, resized = prepare_image(args.image)
    with torch.inference_mode():
        outputs = model(image_tensor.to(device))

    voices = []
    for voice_index in range(2):
        offset = voice_index * len(TASKS)
        decoded = {
            task_name: decode_head(outputs[offset + task_index], classes, args.top_k)
            for task_index, (task_name, classes) in enumerate(TASKS)
        }
        visible = decoded["state"]["value"] != "empty"
        voices.append(
            {
                "voice": voice_index,
                "visible": visible,
                "semantic_fields_applicable": visible,
                "note": (
                    "Duration, dot and division predictions are meaningful only when state is note or rest."
                    if not visible
                    else ""
                ),
                **decoded,
            }
        )

    report = {
        "image": str(args.image.resolve()),
        "model": str(args.model.resolve()),
        "device": str(device),
        "parameters": parameter_count(model),
        "input": {
            "original_width": original_size[0],
            "original_height": original_size[1],
            "model_width": INPUT_WIDTH,
            "model_height": INPUT_HEIGHT,
            "resized": resized,
            "scope": "ground-truth event-centred score_tab crop",
        },
        "voices": voices,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
