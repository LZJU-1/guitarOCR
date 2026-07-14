from __future__ import annotations

import numpy as np
from PIL import Image
import torch

from guitarocr.data.fret_token_crop import crop_fret_token
from guitarocr.models.fret_token_model import FretTokenCNN


def load_fret_token_model(path, device: torch.device) -> tuple[FretTokenCNN, list[str], dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = list(checkpoint["classes"])
    model = FretTokenCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes, checkpoint


@torch.inference_mode()
def classify_event_frets(
    page: Image.Image,
    systems: list[dict],
    model: FretTokenCNN,
    classes: list[str],
    device: torch.device,
    *,
    nonblank_threshold: float = 0.45,
    blank_suppression_threshold: float = 0.80,
    batch_size: int = 256,
) -> dict:
    records: list[tuple[torch.Tensor, dict, dict, int]] = []
    for system in systems:
        spacing = float(system["tab_spacing"])
        for measure in system["measures"]:
            for event in measure["events"]:
                event["fret_token_predictions"] = []
                for string_index, y in enumerate(system["tab_string_y"], start=1):
                    crop = crop_fret_token(page, float(event["x"]), float(y), spacing)
                    array = np.asarray(crop, dtype=np.float32) / 255.0
                    records.append(
                        (
                            torch.from_numpy(1.0 - array).unsqueeze(0),
                            measure,
                            event,
                            string_index,
                        )
                    )

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        tensors = torch.stack([record[0] for record in batch]).to(device)
        probabilities = torch.softmax(model(tensors), dim=1).cpu()
        for row, (_tensor, _measure, event, string_index) in zip(probabilities, batch):
            class_index = int(row.argmax())
            event["fret_token_predictions"].append(
                {
                    "string": string_index,
                    "class": classes[class_index],
                    "probability": float(row[class_index]),
                    "blank_probability": float(row[0]),
                    "top": [
                        {"class": classes[int(index)], "probability": float(row[int(index)])}
                        for index in torch.topk(row, min(3, len(classes))).indices
                    ],
                }
            )

    summary = {
        "events": 0,
        "classifier_notes": 0,
        "detector_fallback_notes": 0,
        "suppressed_detector_notes": 0,
        "orphan_detector_events": 0,
    }
    for system in systems:
        match_radius = max(6.0, float(system["tab_spacing"]) * 0.60)
        for measure in system["measures"]:
            detector_events = list(measure.get("tab_events", []))
            used_detector: set[int] = set()
            fused_events = []
            for event in measure["events"]:
                summary["events"] += 1
                nearest = None
                candidates = [
                    (abs(float(item["x"]) - float(event["x"])), index, item)
                    for index, item in enumerate(detector_events)
                    if index not in used_detector
                ]
                if candidates:
                    delta, index, item = min(candidates)
                    if delta <= match_radius:
                        nearest = item
                        used_detector.add(index)
                detector_by_string = {
                    int(note["string"]): note["fret"]
                    for note in (nearest or {}).get("notes", [])
                }
                notes = []
                for prediction in event["fret_token_predictions"]:
                    string_index = int(prediction["string"])
                    class_name = prediction["class"]
                    probability = float(prediction["probability"])
                    blank_probability = float(prediction["blank_probability"])
                    value = None
                    source = None
                    if class_name != "blank" and probability >= nonblank_threshold:
                        value = "X" if class_name == "X" else int(class_name)
                        source = "fret_token_cnn"
                        summary["classifier_notes"] += 1
                    elif string_index in detector_by_string and blank_probability < blank_suppression_threshold:
                        value = detector_by_string[string_index]
                        source = "tab_symbol_detector_fallback"
                        summary["detector_fallback_notes"] += 1
                    elif string_index in detector_by_string:
                        summary["suppressed_detector_notes"] += 1
                    if value is not None:
                        notes.append(
                            {
                                "string": string_index,
                                "fret": value,
                                "source": source,
                                "confidence": probability,
                            }
                        )
                if notes:
                    fused_events.append({"x": float(event["x"]), "notes": notes})
            for index, detector_event in enumerate(detector_events):
                if index not in used_detector:
                    fused_events.append(detector_event)
                    summary["orphan_detector_events"] += 1
            measure["tab_events"] = sorted(fused_events, key=lambda item: float(item["x"]))
    return summary
