from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from guitarocr.models.symbol_model import AtomicSymbolCNN


SUPPORTED_SIGNATURES = {(1, 4), (2, 4), (3, 4), (4, 4), (6, 4), (6, 8), (9, 4), (10, 8)}


def prepare_component(image: np.ndarray) -> Image.Image:
    ink = 255 - image
    coordinates = cv2.findNonZero((ink > 30).astype(np.uint8))
    if coordinates is None:
        return Image.new("L", (64, 64), 255)
    x, y, width, height = cv2.boundingRect(coordinates)
    glyph = image[y : y + height, x : x + width]
    scale = 34.0 / max(width, height)
    resized = cv2.resize(
        glyph,
        (max(2, round(width * scale)), max(2, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    canvas = np.full((64, 64), 255, dtype=np.uint8)
    left = (64 - resized.shape[1]) // 2
    top = (64 - resized.shape[0]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return Image.fromarray(canvas, mode="L")


@torch.inference_mode()
def classify_components(
    crops: list[Image.Image],
    model: AtomicSymbolCNN,
    classes: list[str],
    device: torch.device,
) -> list[dict]:
    if not crops:
        return []
    arrays = np.stack([np.asarray(crop, dtype=np.float32) / 255.0 for crop in crops])
    tensor = torch.from_numpy(arrays).unsqueeze(1).to(device)
    tensor = (tensor - 0.5) / 0.5
    probabilities = model(tensor).softmax(dim=1)
    digit_indices = [index for index, name in enumerate(classes) if name.startswith("digit_")]
    restricted = probabilities[:, digit_indices]
    values, local_indices = restricted.max(dim=1)
    results = []
    for row, probability, local_index in zip(probabilities, values, local_indices):
        class_index = digit_indices[int(local_index)]
        results.append(
            {
                "digit": int(classes[class_index].removeprefix("digit_")),
                "digit_probability": float(probability),
                "global_probability": float(row[class_index]),
                "global_class": classes[int(row.argmax())],
                "digit_probabilities": {
                    str(digit): float(row[class_index])
                    for digit, class_index in enumerate(digit_indices)
                },
            }
        )
    return results


def component_candidates(
    page: Image.Image,
    measure: dict,
    system: dict,
    model: AtomicSymbolCNN,
    classes: list[str],
    device: torch.device,
) -> list[dict]:
    gray = np.asarray(page.convert("L"), dtype=np.uint8)
    spacing = float(system["score_spacing"])
    left = max(0, math.floor(float(measure["bbox"][0]) + 0.25 * spacing))
    right = min(gray.shape[1], math.ceil(float(measure["bbox"][0]) + min(float(measure["bbox"][2]), 24.0 * spacing)))
    top = max(0, math.floor(system["score_line_y"][0] - 1.8 * spacing))
    bottom = min(gray.shape[0], math.ceil(system["score_line_y"][-1] + 1.8 * spacing))
    if right <= left or bottom <= top:
        return []
    crop = gray[top:bottom, left:right]
    ink = (crop < 180).astype(np.uint8)
    horizontal_size = max(12, round(3.0 * spacing))
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    )
    cleaned = (ink & (1 - horizontal)).astype(np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 2), dtype=np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    raw: list[dict] = []
    component_images: list[Image.Image] = []
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        if not (0.22 * spacing <= width <= 3.2 * spacing):
            continue
        if not (0.65 * spacing <= height <= 2.7 * spacing):
            continue
        if area < max(7, 0.10 * spacing * spacing):
            continue
        pad = 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(crop.shape[1], x + width + pad), min(crop.shape[0], y + height + pad)
        component = crop[y0:y1, x0:x1].copy()
        # Whiten line pixels identified by the long-horizontal opening.
        component[horizontal[y0:y1, x0:x1] > 0] = 255
        component_images.append(prepare_component(component))
        raw.append(
            {
                "bbox": [float(left + x), float(top + y), float(width), float(height)],
                "center_x": float(left + x + width / 2.0),
                "center_y": float(top + y + height / 2.0),
            }
        )
    classifications = classify_components(component_images, model, classes, device)
    for item, classification in zip(raw, classifications):
        item.update(classification)
    return raw


def digit_groups(components: list[dict], spacing: float) -> list[dict]:
    ordered = sorted(components, key=lambda item: item["center_x"])
    groups: list[list[dict]] = []
    for component in ordered:
        if not groups:
            groups.append([component])
            continue
        previous = groups[-1][-1]
        previous_right = previous["bbox"][0] + previous["bbox"][2]
        gap = component["bbox"][0] - previous_right
        if len(groups[-1]) < 2 and gap <= 0.65 * spacing:
            groups[-1].append(component)
        else:
            groups.append([component])
    results = []
    for group in groups:
        value = int("".join(str(item["digit"]) for item in group))
        left = min(item["bbox"][0] for item in group)
        right = max(item["bbox"][0] + item["bbox"][2] for item in group)
        results.append(
            {
                "value": value,
                "center_x": (left + right) / 2.0,
                "left": left,
                "right": right,
                "confidence": min(item["global_probability"] for item in group),
                "digits": group,
            }
        )
    return results


def classify_signature_half(
    patch: np.ndarray,
    page_top: int,
    score_line_y: list[float],
    spacing: float,
    model: AtomicSymbolCNN,
    classes: list[str],
    device: torch.device,
) -> tuple[int, float, list[dict]] | None:
    cleaned = patch.copy()
    for line_y in score_line_y:
        local = int(round(line_y)) - page_top
        if -1 <= local < cleaned.shape[0] + 1:
            cleaned[max(0, local - 1) : min(cleaned.shape[0], local + 2), :] = 255
    mask = cleaned < 190
    active = np.where(mask.sum(axis=0) >= 2)[0]
    if active.size == 0:
        return None
    groups: list[list[int]] = []
    for raw in active:
        value = int(raw)
        if not groups or value > groups[-1][-1] + 2:
            groups.append([value])
        else:
            groups[-1].append(value)
    groups = [group for group in groups if len(group) >= max(2, round(0.12 * spacing))]
    if not 1 <= len(groups) <= 2:
        return None
    crops: list[Image.Image] = []
    metadata: list[dict] = []
    for group in groups:
        left, right = max(0, group[0] - 1), min(cleaned.shape[1], group[-1] + 2)
        glyph = cleaned[:, left:right]
        crops.append(prepare_component(glyph))
        metadata.append({"local_x": [left, right]})
    predictions = classify_components(crops, model, classes, device)
    value = int("".join(str(item["digit"]) for item in predictions))
    confidence = min(item["global_probability"] for item in predictions)
    for item, prediction in zip(metadata, predictions):
        item.update(prediction)
    return value, confidence, metadata


def stacked_signature_candidates(
    page: Image.Image,
    measure: dict,
    system: dict,
    model: AtomicSymbolCNN,
    classes: list[str],
    device: torch.device,
) -> list[dict]:
    gray = np.asarray(page.convert("L"), dtype=np.uint8)
    spacing = float(system["score_spacing"])
    left = max(0, math.floor(float(measure["bbox"][0]) + 0.25 * spacing))
    right = min(gray.shape[1], math.ceil(float(measure["bbox"][0]) + min(float(measure["bbox"][2]), 24.0 * spacing)))
    top = max(0, math.floor(system["score_line_y"][0] - 1.8 * spacing))
    bottom = min(gray.shape[0], math.ceil(system["score_line_y"][-1] + 1.8 * spacing))
    crop = gray[top:bottom, left:right]
    ink = (crop < 180).astype(np.uint8)
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, round(3.0 * spacing)), 1)),
    )
    cleaned = cv2.morphologyEx(
        (ink & (1 - horizontal)).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 2), dtype=np.uint8)
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    middle_page = int(round(system["score_line_y"][2]))
    candidates: list[dict] = []
    for index in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[index]]
        center_y = top + y + height / 2.0
        if not (0.35 * spacing <= width <= 4.5 * spacing):
            continue
        if not (2.8 * spacing <= height <= 5.0 * spacing):
            continue
        if abs(center_y - system["score_line_y"][2]) > 1.0 * spacing:
            continue
        if area < 0.5 * spacing * spacing:
            continue
        page_left, page_right = left + x, left + x + width
        page_top, page_bottom = top + y, top + y + height
        if not (page_top + 4 < middle_page < page_bottom - 4):
            continue
        numerator_patch = gray[page_top:middle_page, page_left:page_right]
        denominator_patch = gray[middle_page:page_bottom, page_left:page_right]
        numerator = classify_signature_half(
            numerator_patch, page_top, system["score_line_y"], spacing, model, classes, device
        )
        denominator = classify_signature_half(
            denominator_patch, middle_page, system["score_line_y"], spacing, model, classes, device
        )
        if numerator is None or denominator is None:
            continue
        numerator_value, numerator_confidence, numerator_digits = numerator
        denominator_value, denominator_confidence, denominator_digits = denominator
        supported = [
            signature for signature in SUPPORTED_SIGNATURES
            if len(str(signature[0])) == len(numerator_digits)
            and len(str(signature[1])) == len(denominator_digits)
        ]
        if supported:
            scored: list[tuple[float, tuple[int, int]]] = []
            for signature in supported:
                characters = str(signature[0]) + str(signature[1])
                digits = numerator_digits + denominator_digits
                probability = 1.0
                for character, digit in zip(characters, digits):
                    probability *= max(1e-12, float(digit["digit_probabilities"][character]))
                scored.append((probability ** (1.0 / len(characters)), signature))
            grammar_confidence, grammar_signature = max(scored, key=lambda item: item[0])
            numerator_value, denominator_value = grammar_signature
            numerator_confidence = denominator_confidence = grammar_confidence
        if (numerator_value, denominator_value) not in SUPPORTED_SIGNATURES:
            continue
        candidates.append(
            {
                "numerator": numerator_value,
                "denominator": denominator_value,
                "confidence": math.sqrt(max(0.0, numerator_confidence * denominator_confidence)),
                "x": (page_left + page_right) / 2.0,
                "bbox": [float(page_left), float(page_top), float(width), float(height)],
                "components": numerator_digits + denominator_digits,
                "method": "stacked_component",
            }
        )
    return candidates


def recognize_printed_time_signature(
    page: Image.Image,
    measure: dict,
    system: dict,
    model: AtomicSymbolCNN,
    classes: list[str],
    device: torch.device,
    threshold: float = 0.45,
) -> dict | None:
    components = component_candidates(page, measure, system, model, classes, device)
    middle = float(system["score_line_y"][2])
    spacing = float(system["score_spacing"])
    top_components = [item for item in components if item["center_y"] < middle - 0.10 * spacing]
    bottom_components = [item for item in components if item["center_y"] > middle + 0.10 * spacing]
    numerator_groups = digit_groups(top_components, spacing)
    denominator_groups = digit_groups(bottom_components, spacing)
    candidates: list[dict] = stacked_signature_candidates(page, measure, system, model, classes, device)
    for numerator in numerator_groups:
        if not 1 <= numerator["value"] <= 32:
            continue
        for denominator in denominator_groups:
            if denominator["value"] not in {2, 4, 8, 16}:
                continue
            alignment = abs(numerator["center_x"] - denominator["center_x"])
            if alignment > 1.1 * spacing:
                continue
            confidence = math.sqrt(max(0.0, numerator["confidence"] * denominator["confidence"]))
            confidence *= math.exp(-alignment / max(1.0, spacing))
            candidates.append(
                {
                    "numerator": numerator["value"],
                    "denominator": denominator["value"],
                    "confidence": confidence,
                    "x": (numerator["center_x"] + denominator["center_x"]) / 2.0,
                    "components": numerator["digits"] + denominator["digits"],
                    "method": "paired_components",
                }
            )
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item["confidence"])
    return best if best["confidence"] >= threshold else None


def propagate_time_signatures(
    page: Image.Image,
    systems: list[dict],
    model: AtomicSymbolCNN,
    classes: list[str],
    device: torch.device,
    initial: tuple[int, int] | None = None,
    threshold: float = 0.45,
) -> tuple[int, int] | None:
    current = initial
    for system in systems:
        for measure in system["measures"]:
            printed = recognize_printed_time_signature(page, measure, system, model, classes, device, threshold)
            if printed is not None:
                current = (int(printed["numerator"]), int(printed["denominator"]))
            measure["printed_time_signature"] = printed
            measure["time_signature"] = list(current) if current is not None else None
            measure["time_signature_source"] = "printed" if printed is not None else "carried" if current is not None else "unknown"
    return current


def load_atomic_model(checkpoint_path: Path, device: torch.device) -> tuple[AtomicSymbolCNN, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = AtomicSymbolCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes
