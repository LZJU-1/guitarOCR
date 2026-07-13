from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from guitarocr.models.symbol_model import AtomicSymbolCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a pre-cropped GuitarOCR atomic symbol.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = AtomicSymbolCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = Image.open(args.image).convert("L").resize((64, 64), Image.Resampling.LANCZOS)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, mean=[0.5], std=[0.5]).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    values, indices = probabilities.topk(min(args.top_k, len(classes)))
    result = [
        {"class": classes[int(index)], "probability": float(value)}
        for value, index in zip(values.cpu(), indices.cpu())
    ]
    print(json.dumps({"image": str(args.image), "predictions": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
