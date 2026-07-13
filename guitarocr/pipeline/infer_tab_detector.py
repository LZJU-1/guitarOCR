from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import WEIGHTS_ROOT
from guitarocr.training.train_tab_detector import decode_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect TAB digits/X in a 512x128 measure crop.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=WEIGHTS_ROOT / "tab_symbol_detector.pt",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = TabSymbolDetector(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with Image.open(args.image) as opened:
        image = opened.convert("L")
    if image.size != (512, 128):
        raise ValueError(f"Expected an already letterboxed 512x128 measure crop, got {image.size}")
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0).to(device)
    tensor = (tensor - 0.5) / 0.5
    with torch.inference_mode():
        outputs = model(tensor)
        detections = decode_detections(outputs, threshold=args.threshold)[0]
    result = [
        {"class": classes[item["class_index"]], "score": item["score"], "bbox": item["bbox"]}
        for item in detections
    ]
    print(json.dumps({"image": str(args.image.resolve()), "detections": result}, indent=2))


if __name__ == "__main__":
    main()
