from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image
import torch

from guitarocr.models.symbol_model import AtomicSymbolCNN
from guitarocr.pipeline.time_signature_recognizer import classify_components, prepare_component


@torch.inference_mode()
def recognize_tempo(
    page: Image.Image,
    first_system: dict,
    model: AtomicSymbolCNN,
    classes: list[str],
    device: torch.device,
) -> dict | None:
    """Read the metronome number immediately above the first printed system."""
    gray = np.asarray(page.convert("L"), dtype=np.uint8)
    spacing = float(first_system["score_spacing"])
    boundary = float(first_system["boundaries"][0])
    left = max(0, math.floor(boundary - 1.0 * spacing))
    right = min(gray.shape[1], math.ceil(boundary + 18.0 * spacing))
    top = max(0, math.floor(first_system["score_line_y"][0] - 8.0 * spacing))
    bottom = max(top + 1, math.floor(first_system["score_line_y"][0] - 0.55 * spacing))
    crop = gray[top:bottom, left:right]
    count, _, stats, _ = cv2.connectedComponentsWithStats((crop < 180).astype(np.uint8), connectivity=8)
    components: list[dict] = []
    images = []
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        if not (0.15 * spacing <= width <= 1.2 * spacing):
            continue
        if not (0.55 * spacing <= height <= 1.6 * spacing):
            continue
        if area < max(8, 0.06 * spacing * spacing):
            continue
        pad = 2
        images.append(prepare_component(crop[max(0, y-pad):min(crop.shape[0], y+height+pad),
                                                  max(0, x-pad):min(crop.shape[1], x+width+pad)]))
        components.append({
            "bbox": [left + x, top + y, width, height],
            "center_x": left + x + width / 2.0,
            "center_y": top + y + height / 2.0,
        })
    predictions = classify_components(images, model, classes, device)
    digits = []
    for component, prediction in zip(components, predictions):
        if prediction["global_class"].startswith("digit_") and prediction["global_probability"] >= 0.50:
            digits.append({**component, **prediction})
    digits.sort(key=lambda item: (item["center_y"], item["center_x"]))

    groups: list[list[dict]] = []
    for digit in digits:
        target = next((group for group in reversed(groups)
                       if abs(group[-1]["center_y"] - digit["center_y"]) <= 0.35 * spacing
                       and 0 <= digit["bbox"][0] - (group[-1]["bbox"][0] + group[-1]["bbox"][2]) <= 0.55 * spacing), None)
        if target is None:
            groups.append([digit])
        else:
            target.append(digit)
    candidates = []
    for group in groups:
        if not 2 <= len(group) <= 3:
            continue
        value = int("".join(str(item["digit"]) for item in group))
        if not 20 <= value <= 400:
            continue
        candidates.append({
            "tempo_quarter": value,
            "confidence": min(float(item["global_probability"]) for item in group),
            "bbox": [
                group[0]["bbox"][0], min(item["bbox"][1] for item in group),
                group[-1]["bbox"][0] + group[-1]["bbox"][2] - group[0]["bbox"][0],
                max(item["bbox"][3] for item in group),
            ],
            "digits": group,
        })
    if not candidates:
        return None
    # Tempo is normally the lowest complete digit group above the first staff.
    return max(candidates, key=lambda item: (item["bbox"][1], item["confidence"]))
