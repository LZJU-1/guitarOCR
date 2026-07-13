from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from guitarocr.data.build_score_event_locator_dataset import tile_score_measure
from guitarocr.data.build_score_rhythm_dataset import build_event_crop
from guitarocr.models.rhythm_context_model import RhythmContextCNN
from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import DATABASE_ROOT, WEIGHTS_ROOT
from guitarocr.pipeline.infer_rhythm_event import TASKS, decode_head
from guitarocr.pipeline.measure_rhythm_constraints import audit_score_ir
from guitarocr.pipeline.score_tab_fingering import build_score_ir, detect_tab_fingering, resolve_unambiguous_ties
from guitarocr.pipeline.score_tab_geometry import detect_score_tab_geometry
from guitarocr.pipeline.tie_inference import classify_ties, load_tie_model
from guitarocr.pipeline.time_signature_recognizer import load_atomic_model, propagate_time_signatures
from guitarocr.training.train_score_event_locator import decode_events


def map_event_to_page(event: dict, transform: dict) -> dict:
    left = float(transform["page_crop_xyxy"][0])
    scale = float(transform["scale"])
    offset = float(transform["offset_x"])
    tile_start = float(transform["tile_start_x"])
    return {
        "x": (float(event["x"]) - offset + tile_start) / scale + left,
        "locator_confidence": float(event["score"]),
    }


def merge_event_columns(events: list[dict], spacing: float) -> list[dict]:
    kept: list[dict] = []
    minimum_distance = max(2.0, spacing * 0.35)
    for event in sorted(events, key=lambda item: item["locator_confidence"], reverse=True):
        duplicate = next((item for item in kept if abs(item["x"] - event["x"]) <= minimum_distance), None)
        if duplicate is None:
            kept.append(event)
        elif event["locator_confidence"] > duplicate["locator_confidence"]:
            duplicate.update(event)
    kept.sort(key=lambda item: item["x"])
    return kept


@torch.inference_mode()
def locate_page_events(
    page: Image.Image,
    model: ScoreEventLocator,
    device: torch.device,
    threshold: float,
    batch_size: int = 32,
) -> list[dict]:
    systems = detect_score_tab_geometry(page)
    tile_records: list[tuple[torch.Tensor, dict, dict]] = []
    for system in systems:
        for measure in system["measures"]:
            for tile, _, transform in tile_score_measure(page, measure["bbox"], system["score_line_y"], []):
                array = np.asarray(tile, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(1.0 - array).unsqueeze(0)
                tile_records.append((tensor, transform, measure))

    for start in range(0, len(tile_records), batch_size):
        batch = tile_records[start : start + batch_size]
        tensors = torch.stack([item[0] for item in batch]).to(device)
        decoded = decode_events(model(tensors), threshold=threshold)
        for predictions, (_, transform, measure) in zip(decoded, batch):
            measure["events"].extend(map_event_to_page(event, transform) for event in predictions)

    for system in systems:
        for measure in system["measures"]:
            measure["events"] = merge_event_columns(measure["events"], system["score_spacing"])
    return systems


@torch.inference_mode()
def classify_rhythm(
    page: Image.Image,
    systems: list[dict],
    model: RhythmContextCNN,
    device: torch.device,
    crop_root: Path | None,
    batch_size: int = 64,
) -> None:
    records: list[tuple[torch.Tensor, dict, str]] = []
    for system in systems:
        for measure in system["measures"]:
            for event_index, event in enumerate(measure["events"]):
                crop, transform = build_event_crop(page, event["x"], system["score_line_y"])
                sample_id = f"s{system['system_index']:02d}_m{measure['measure_number']:03d}_e{event_index:03d}"
                if crop_root is not None:
                    crop.save(crop_root / f"{sample_id}.png", format="PNG", optimize=True)
                array = np.asarray(crop, dtype=np.float32) / 255.0
                records.append((torch.from_numpy(1.0 - array).unsqueeze(0), event, sample_id))
                event["rhythm_crop"] = f"crops/{sample_id}.png" if crop_root is not None else None
                event["rhythm_transform"] = transform

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        tensors = torch.stack([item[0] for item in batch]).to(device)
        outputs = model(tensors)
        for batch_index, (_, event, _) in enumerate(batch):
            voices = []
            for voice_index in range(2):
                offset = voice_index * len(TASKS)
                decoded = {
                    task_name: decode_head(outputs[offset + task_index][batch_index : batch_index + 1], classes, 3)
                    for task_index, (task_name, classes) in enumerate(TASKS)
                }
                visible = decoded["state"]["value"] != "empty"
                voices.append({"voice": voice_index, "visible": visible, **decoded})
            event["voices"] = voices


def draw_overlay(page: Image.Image, systems: list[dict], output: Path) -> None:
    overlay = page.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for system in systems:
        left, right = system["boundaries"][0], system["boundaries"][-1]
        for y in system["score_line_y"]:
            draw.line((left, y, right, y), fill=(0, 175, 70), width=1)
        for y in system["tab_string_y"]:
            draw.line((left, y, right, y), fill=(125, 125, 125), width=1)
        for boundary in system["boundaries"]:
            draw.line((boundary, system["score_line_y"][0], boundary, system["tab_string_y"][-1]), fill=(0, 135, 230), width=1)
        for measure in system["measures"]:
            printed_signature = measure.get("printed_time_signature")
            if printed_signature is not None and printed_signature.get("bbox") is not None:
                x, y, width, height = printed_signature["bbox"]
                draw.rectangle((x, y, x + width, y + height), outline=(135, 45, 220), width=2)
                draw.text(
                    (x, max(0, y - 12)),
                    f"{printed_signature['numerator']}/{printed_signature['denominator']}",
                    fill=(105, 25, 180),
                )
            for symbol in measure.get("tab_symbols", []):
                x, y, width, height = symbol["bbox"]
                color = (220, 0, 200) if symbol["class"] == "dead_x" else (225, 35, 35)
                draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
            for event in measure["events"]:
                x = event["x"]
                tie_prediction = event.get("tie_prediction")
                if tie_prediction is not None and tie_prediction["visual_positive"]:
                    for target_y in tie_prediction["target_y_page"]:
                        radius = max(3.0, system["score_spacing"] * 0.35)
                        draw.ellipse(
                            (x - radius, target_y - radius, x + radius, target_y + radius),
                            outline=(145, 20, 210), width=2,
                        )
                    draw.text(
                        (x + 2, system["score_line_y"][-1] + 4 * system["score_spacing"]),
                        f"T{tie_prediction['visual_probability']:.2f}",
                        fill=(125, 15, 185),
                    )
                draw.line(
                    (x, system["score_line_y"][0] - 6 * system["score_spacing"],
                     x, system["score_line_y"][-1] + 6 * system["score_spacing"]),
                    fill=(255, 120, 0), width=2,
                )
                draw.text((x + 2, system["score_line_y"][0] - 6 * system["score_spacing"]),
                          f"{event['locator_confidence']:.2f}", fill=(190, 70, 0))
    overlay.save(output, format="PNG", optimize=True)


def parse_time_signature(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        numerator, denominator = (int(part) for part in value.split("/", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("time signature must look like 4/4 or 6/8") from error
    if numerator <= 0 or denominator <= 0:
        raise argparse.ArgumentTypeError("time signature values must be positive")
    return numerator, denominator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locate score events and classify rhythm from a TuxGuitar score_tab page using pixels only."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--locator-model", type=Path,
        default=WEIGHTS_ROOT / "score_event_locator.pt",
    )
    parser.add_argument(
        "--rhythm-model", type=Path,
        default=WEIGHTS_ROOT / "rhythm_context_cnn.pt",
    )
    parser.add_argument(
        "--tab-model", type=Path,
        default=WEIGHTS_ROOT / "tab_symbol_detector.pt",
    )
    parser.add_argument(
        "--atomic-model", type=Path,
        default=WEIGHTS_ROOT / "atomic_symbol_cnn.pt",
    )
    parser.add_argument(
        "--tie-model", type=Path,
        default=WEIGHTS_ROOT / "tie_context_cnn.pt",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--measure-number-offset", type=int,
        help="One-based number of the first measure on this page; omit when unknown.",
    )
    parser.add_argument("--skip-rhythm", action="store_true")
    parser.add_argument("--skip-tab", action="store_true")
    parser.add_argument("--skip-time-signature", action="store_true")
    parser.add_argument("--skip-ties", action="store_true")
    parser.add_argument("--initial-time-signature", type=parse_time_signature)
    parser.add_argument("--time-signature-threshold", type=float, default=0.20)
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locator_checkpoint = torch.load(args.locator_model, map_location=device, weights_only=False)
    locator = ScoreEventLocator().to(device)
    locator.load_state_dict(locator_checkpoint["model_state"])
    locator.eval()
    threshold = args.threshold if args.threshold is not None else float(locator_checkpoint.get("detection_threshold", 0.3))

    with Image.open(args.image) as opened:
        page = opened.convert("L")
    output_root = args.output or (
        DATABASE_ROOT / "score_event_locator" / "page_inference" / args.image.stem
    )
    output_root.mkdir(parents=True, exist_ok=True)
    crop_root = output_root / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)

    systems = locate_page_events(page, locator, device, threshold)
    if not systems:
        raise RuntimeError("No paired score/TAB system was detected")
    if not args.skip_time_signature:
        atomic_model, atomic_classes = load_atomic_model(args.atomic_model, device)
        propagate_time_signatures(
            page,
            systems,
            atomic_model,
            atomic_classes,
            device,
            initial=args.initial_time_signature,
            threshold=args.time_signature_threshold,
        )
    if not args.skip_tab:
        tab_checkpoint = torch.load(args.tab_model, map_location=device, weights_only=False)
        tab_classes = tab_checkpoint["classes"]
        tab_model = TabSymbolDetector(len(tab_classes)).to(device)
        tab_model.load_state_dict(tab_checkpoint["model_state"])
        tab_model.eval()
        detect_tab_fingering(page, systems, tab_model, tab_classes, device, threshold=0.3)
    if not args.skip_rhythm:
        rhythm_checkpoint = torch.load(args.rhythm_model, map_location=device, weights_only=False)
        rhythm = RhythmContextCNN().to(device)
        rhythm.load_state_dict(rhythm_checkpoint["model_state"])
        rhythm.eval()
        classify_rhythm(page, systems, rhythm, device, crop_root)
    if not args.skip_ties:
        tie_model, tie_threshold = load_tie_model(args.tie_model, device)
        classify_ties(page, systems, tie_model, device, tie_threshold)

    overlay_path = output_root / "overlay.png"
    json_path = output_root / "events.json"
    draw_overlay(page, systems, overlay_path)
    score_ir = build_score_ir(systems, measure_number_offset=args.measure_number_offset)
    rhythm_audit = audit_score_ir(score_ir) if not args.skip_rhythm else None
    tie_summary = resolve_unambiguous_ties(score_ir) if not args.skip_ties else None
    report = {
        "schema_version": "1.0",
        "image": str(args.image.resolve()),
        "scope": "pixel-only TuxGuitar score_tab geometry and event localisation; optional rhythm classification",
        "device": str(device),
        "locator_threshold": threshold,
        "systems": systems,
        "score_ir": score_ir,
        "summary": {
            "systems": len(systems),
            "measures": sum(len(system["measures"]) for system in systems),
            "events": sum(len(measure["events"]) for system in systems for measure in system["measures"]),
            "tab_symbols": sum(len(measure.get("tab_symbols", [])) for system in systems for measure in system["measures"]),
            "tab_events": sum(len(measure.get("tab_events", [])) for system in systems for measure in system["measures"]),
            "matched_events": sum(
                sum(event["tab_match"] == "matched" for event in measure["events"])
                for measure in score_ir["tracks"][0]["measures"]
            ),
            "orphan_tab_events": sum(
                len(measure["orphan_tab_events"]) for measure in score_ir["tracks"][0]["measures"]
            ),
            "printed_time_signatures": sum(
                measure["printed_time_signature"] is not None
                for measure in score_ir["tracks"][0]["measures"]
            ),
            "known_time_signatures": sum(
                measure["time_signature"] is not None
                for measure in score_ir["tracks"][0]["measures"]
            ),
            "rhythm_audit": rhythm_audit,
            "ties": tie_summary,
        },
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    score_ir_path = output_root / "score_ir.json"
    score_ir_path.write_text(json.dumps(score_ir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        **report["summary"], "json": str(json_path), "score_ir": str(score_ir_path), "overlay": str(overlay_path)
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
