from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import torch

from guitarocr.data.build_score_rhythm_dataset import build_event_crop
from guitarocr.models.tie_context_model import INPUT_HEIGHT, Y_BINS, TieContextCNN


SCORE_TAB_TIE_THRESHOLD = 0.05


def decode_y_bins(logits: torch.Tensor, count: int) -> list[int]:
    if count <= 0:
        return []
    probabilities = logits.sigmoid().tolist()
    ranked = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)
    selected: list[int] = []
    for candidate in ranked:
        if all(abs(candidate - existing) > 1 for existing in selected):
            selected.append(candidate)
            if len(selected) >= count:
                break
    return sorted(selected)


def bin_to_page_y(bin_index: int, transform: dict) -> float:
    crop_y = (bin_index + 0.5) * INPUT_HEIGHT / Y_BINS
    top = float(transform["page_crop_xyxy"][1])
    original_height = float(transform["original_crop_size"][1])
    return top + crop_y * original_height / INPUT_HEIGHT


@torch.inference_mode()
def classify_ties(
    page: Image.Image,
    systems: list[dict],
    model: TieContextCNN,
    device: torch.device,
    threshold: float,
    batch_size: int = 64,
    crop_builder: Callable[[Image.Image, float, list[float]], tuple[Image.Image, dict]] = build_event_crop,
    reference_line_key: str = "score_line_y",
) -> None:
    records: list[tuple[torch.Tensor, dict, dict]] = []
    page_ink = 1.0 - np.asarray(page.convert("L"), dtype=np.float32) / 255.0
    for system in systems:
        score_lines = [float(value) for value in system[reference_line_key]]
        score_spacing = (
            (score_lines[-1] - score_lines[0]) / max(1, len(score_lines) - 1)
        )
        profile_top = score_lines[0] - 12.0 * score_spacing
        profile_step = score_spacing / 2.0
        profile_count = int(round(
            (score_lines[-1] + 12.0 * score_spacing - profile_top) / profile_step
        )) + 1
        for measure in system["measures"]:
            for event in measure["events"]:
                crop, transform = crop_builder(
                    page, event["x"], system[reference_line_key]
                )
                array = np.asarray(crop, dtype=np.float32) / 255.0
                x_center = int(round(float(event["x"])))
                x_radius = max(3, int(round(0.60 * score_spacing)))
                y_radius = max(2, int(round(0.38 * score_spacing)))
                profile = []
                for profile_index in range(profile_count):
                    y_center = int(round(profile_top + profile_index * profile_step))
                    left = max(0, x_center - x_radius)
                    right = min(page_ink.shape[1], x_center + x_radius + 1)
                    top = max(0, y_center - y_radius)
                    bottom = min(page_ink.shape[0], y_center + y_radius + 1)
                    patch = page_ink[top:bottom, left:right]
                    profile.append(float(patch.mean()) if patch.size else 0.0)
                event["score_notehead_ink_profile"] = {
                    "top_y": profile_top,
                    "step_y": profile_step,
                    "values": profile,
                    "x_radius": x_radius,
                    "y_radius": y_radius,
                }
                records.append((torch.from_numpy(1.0 - array).unsqueeze(0), event, transform))

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        tensors = torch.stack([record[0] for record in batch]).to(device)
        presence_logits, tie_count_logits, note_count_logits, y_logits = model(tensors)
        presence_probabilities = presence_logits.softmax(dim=1)[:, 1]
        tie_count_probabilities = tie_count_logits.softmax(dim=1)
        note_count_probabilities = note_count_logits.softmax(dim=1)
        for index, (_, event, transform) in enumerate(batch):
            presence_probability = float(presence_probabilities[index])
            tie_count = int(tie_count_probabilities[index].argmax())
            score_note_count = int(note_count_probabilities[index].argmax())
            decode_count = max(1, tie_count) if presence_probability >= threshold else 0
            bins = decode_y_bins(y_logits[index].cpu(), decode_count)
            event["tie_prediction"] = {
                "visual_probability": presence_probability,
                "visual_positive": presence_probability >= threshold,
                "visual_threshold": threshold,
                "tie_note_count": tie_count,
                "tie_note_count_confidence": float(tie_count_probabilities[index, tie_count]),
                "score_note_count": score_note_count,
                "score_note_count_confidence": float(note_count_probabilities[index, score_note_count]),
                "target_y_bins": bins,
                "target_y_page": [bin_to_page_y(value, transform) for value in bins],
                "score_notehead_ink_profile": event.get("score_notehead_ink_profile"),
            }


def load_tie_model(checkpoint_path: Path, device: torch.device) -> tuple[TieContextCNN, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TieContextCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    # The production threshold is selected on the source-disjoint page-level
    # validation pipeline. Score/TAB missing-note consistency removes visual
    # slur false positives while preserving useful recall.
    return model, max(SCORE_TAB_TIE_THRESHOLD, float(checkpoint["presence_threshold"]))
