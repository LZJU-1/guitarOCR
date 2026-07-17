from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from guitarocr.data.build_gp8_measure_sequence_dataset import recognition_prompt
from guitarocr.data.gp_measure_sequence import (
    format_measure_target,
    format_previous_measure_context,
    parse_measure_target,
)
from guitarocr.data.measure_sequence_constraints import validate_measure_target
from guitarocr.export.measure_sequence_to_gp5 import DEFAULT_TUNING, write_targets_gp5
from guitarocr.pipeline.infer_tuxguitar_score_tab_document import expand_inputs
from guitarocr.pipeline.infer_tuxguitar_tab_page import (
    detect_tab_boundaries_for_lines,
    detect_tab_geometry,
)
from guitarocr.pipeline.layout_classifier import classify_notation_layout
from guitarocr.pipeline.pdf_vector_measures import (
    extract_pdf_vector_measure_boxes,
    extract_pdf_vector_tab_measure_boxes,
    extract_pdf_vector_tab_systems,
)
from guitarocr.pipeline.pdf_vector_metadata import extract_pdf_vector_metadata
from guitarocr.pipeline.pdf_page_renderer import MODEL_RENDER_DPI
from guitarocr.pipeline.score_tab_geometry import detect_score_tab_geometry


def _notation_mode(layout: str) -> str:
    values = {"tab_only": "tab", "score_only": "notation", "score_tab": "both"}
    if layout not in values:
        raise ValueError(f"Unsupported or undetected page layout: {layout}")
    return values[layout]


def _clean_notation_boundaries(boundaries: list[float]) -> list[float]:
    """Merge a score note stem that crosses all five staff lines.

    A real barline normally ends at the outer staff lines, but some short
    stems do as well. Such a stem splits one normal-width measure into one
    very narrow and one residual interval. Repeats and genuinely short bars
    are retained unless the two adjacent intervals recombine to the page's
    normal measure width.
    """

    values = list(boundaries)
    changed = True
    while changed and len(values) >= 4:
        changed = False
        widths = np.diff(values)
        median = float(np.median(widths))
        if median <= 0:
            break
        candidates = []
        for index in range(1, len(values) - 1):
            left = float(values[index] - values[index - 1])
            right = float(values[index + 1] - values[index])
            combined = left + right
            if min(left, right) >= 0.50 * median:
                continue
            if not 0.70 * median <= combined <= 1.45 * median:
                continue
            candidates.append((min(left, right) / median, index))
        if candidates:
            _ratio, index = min(candidates)
            del values[index]
            changed = True
    return values


def _measure_boxes(page: Image.Image, mode: str) -> list[dict[str, Any]]:
    boxes = []
    if mode == "both":
        systems = detect_score_tab_geometry(page)
        for system_index, system in enumerate(systems):
            y0 = float(system["score_line_y"][0]) - 4.0 * float(system["score_spacing"])
            y1 = float(system["tab_string_y"][-1]) + 3.0 * float(system["tab_spacing"])
            for measure_index in range(len(system["boundaries"]) - 1):
                left = float(system["boundaries"][measure_index])
                right = float(system["boundaries"][measure_index + 1])
                boxes.append({
                    "system_index": system_index,
                    "system_measure_index": measure_index,
                    "bbox": [left, y0, right - left, y1 - y0],
                })
        return boxes

    staffs = detect_tab_geometry(
        page, minimum_string_spacing=6.0 if mode == "notation" else None
    )
    for system_index, staff in enumerate(staffs):
        lines = [float(value) for value in staff["string_y"]]
        spacing = float(staff["spacing"])
        if mode == "tab" and not 4 <= len(lines) <= 8:
            continue
        if mode == "notation" and len(lines) != 5:
            continue
        y0 = lines[0] - (4.0 if mode == "tab" else 5.0) * spacing
        y1 = lines[-1] + (3.0 if mode == "tab" else 4.0) * spacing
        boundaries = [float(value) for value in staff["boundaries"]]
        if mode == "notation":
            boundaries = _clean_notation_boundaries(boundaries)
        for measure_index in range(len(boundaries) - 1):
            left, right = boundaries[measure_index:measure_index + 2]
            boxes.append({
                "system_index": system_index,
                "system_measure_index": measure_index,
                "bbox": [left, y0, right - left, y1 - y0],
            })
    return boxes


def _hybrid_tab_pdf_boxes(
    page: Image.Image,
    pixel_boxes: list[dict[str, Any]],
    vector_systems: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add whole TAB systems missed by pixels, retaining raster barlines."""

    groups: dict[int, list[dict[str, Any]]] = {}
    for box in pixel_boxes:
        groups.setdefault(int(box["system_index"]), []).append(box)
    combined: list[dict[str, Any]] = []
    pixel_centers = []
    for boxes in groups.values():
        boxes.sort(key=lambda item: int(item["system_measure_index"]))
        top = min(float(item["bbox"][1]) for item in boxes)
        bottom = max(float(item["bbox"][1] + item["bbox"][3]) for item in boxes)
        pixel_centers.append((0.5 * (top + bottom), boxes))
        combined.append({"center": 0.5 * (top + bottom), "boxes": boxes})

    matched_pixel: set[int] = set()
    for vector in vector_systems:
        string_y = [float(value) for value in vector["string_y"]]
        spacing = float(vector["spacing"])
        # Pixel crop centers sit half a string spacing above the mean string
        # position because the crop reserves four spaces above and three
        # below.  Match in that same coordinate system.
        center = float(np.mean(string_y)) - 0.5 * spacing
        candidates = [
            (abs(pixel_center - center), index)
            for index, (pixel_center, _boxes) in enumerate(pixel_centers)
            if index not in matched_pixel
        ]
        matched_index: int | None = None
        if candidates:
            distance, pixel_index = min(candidates)
            if distance <= max(20.0, 1.75 * spacing):
                matched_index = pixel_index
                matched_pixel.add(pixel_index)
                for box in pixel_centers[pixel_index][1]:
                    box["geometry_source"] = "pixel_staff+pdf_vector_system_check"

        # The vector rows already prove that this is a real TAB system, so a
        # dense run of fret glyphs must not fail the raster density guard.
        boundaries = detect_tab_boundaries_for_lines(
            page, string_y, minimum_horizontal_density=0.0
        )
        if len(boundaries) < 2:
            continue
        top = string_y[0] - 4.0 * spacing
        bottom = string_y[-1] + 3.0 * spacing
        boxes = []
        for measure_index, (left, right) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            boxes.append({
                "system_index": -1,
                "system_measure_index": measure_index,
                "bbox": [left, top, right - left, bottom - top],
                "geometry_source": vector["geometry_source"],
            })
        if matched_index is not None:
            pixel_group = pixel_centers[matched_index][1]
            # Exact vector-derived string rows can reveal a barline that was
            # missed when the pixel chain snapped to a neighbouring fragment.
            # Keep the established pixel result unless the seeded pass finds
            # strictly more complete measure intervals.
            if len(boxes) > len(pixel_group):
                for group in combined:
                    if group["boxes"] is pixel_group:
                        group["boxes"] = boxes
                        group["center"] = center
                        break
            continue
        combined.append({"center": center, "boxes": boxes})

    result = []
    for system_index, group in enumerate(sorted(combined, key=lambda item: item["center"])):
        for box in group["boxes"]:
            box["system_index"] = system_index
            box.setdefault("geometry_source", "pixel_staff_fallback")
            result.append(box)
    return result


def _crop(page: Image.Image, bbox: list[float]) -> Image.Image:
    left, top, width, height = (float(value) for value in bbox)
    # Match the physical padding used to build the GP8 measure-sequence
    # training crops: 1.5 mm horizontally and 4 mm vertically at 180 DPI.
    # Omitting the vertical context makes dense chords and technique labels
    # materially out-of-distribution during full-document inference.
    pixels_per_mm = MODEL_RENDER_DPI / 25.4
    pad_x = 1.5 * pixels_per_mm
    pad_y = 4.0 * pixels_per_mm
    x0 = max(0, round(left - pad_x))
    y0 = max(0, round(top - pad_y))
    x1 = min(page.width, round(left + width + pad_x))
    y1 = min(page.height, round(top + height + pad_y))
    return page.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))).convert("L")


def _repair_truncated_optional_text(target: str) -> tuple[str, list[str]]:
    """Drop only an unterminated optional text effect and close open voices."""

    text_start = target.rfind("<text:")
    if text_start < 0 or target.find(">", text_start) >= 0:
        return target, []
    repaired = target[:text_start].rstrip()
    missing_voice_closers = max(0, repaired.count("{") - repaired.count("}"))
    repaired += "}" * missing_voice_closers
    return repaired, ["drop_unterminated_text"]


def _active_time_signature(previous_targets: list[str]) -> tuple[int, int]:
    for target in reversed(previous_targets):
        value = parse_measure_target(target).get("time_signature")
        if value:
            numerator, denominator = str(value).split("/", maxsplit=1)
            return max(1, int(numerator)), max(1, int(denominator))
    return 4, 4


def _full_measure_rest_target(time_signature: tuple[int, int]) -> str:
    numerator, denominator = time_signature
    total_ticks = max(1, round(3840 * numerator / denominator))
    durations = (
        (3840, "w"),
        (2880, "h."),
        (1920, "h"),
        (1440, "q."),
        (960, "q"),
        (720, "e."),
        (480, "e"),
        (360, "s."),
        (240, "s"),
        (180, "t."),
        (120, "t"),
        (60, "f"),
    )
    events = []
    start = 0
    remaining = total_ticks
    while remaining > 0:
        ticks, token = next(
            ((ticks, token) for ticks, token in durations if ticks <= remaining),
            (remaining, f"d{max(1, round(3840 / remaining))}"),
        )
        events.append(f"@{start}:{token}:r")
        start += ticks
        remaining -= ticks
    return "M2 | V0{" + " ".join(events) + "}"


def prepare_document_crops(
    inputs: list[Path], output: Path, requested_mode: str, force_pdf_render: bool
) -> tuple[str, list[dict[str, Any]]]:
    rendered_root = output / "rendered_pages"
    temp_root = output / "tmp"
    pages = expand_inputs(inputs, rendered_root, temp_root, force_pdf_render)
    if not pages:
        raise ValueError("No supported PDF pages or images were found")
    mode = requested_mode
    records = []
    measure_number = 1
    vector_cache: dict[tuple[Path, str], dict[int, list[dict[str, Any]]]] = {}
    tab_measure_cache: dict[Path, dict[int, list[dict[str, Any]]]] = {}
    tab_system_cache: dict[Path, dict[int, list[dict[str, Any]]]] = {}
    for page_index, source in enumerate(pages, start=1):
        with Image.open(source["image"]) as opened:
            page = opened.convert("L")
        if mode == "auto":
            page_layout = classify_notation_layout(page)
            detected_mode = _notation_mode(page_layout["layout"])
            mode = detected_mode
        boxes = []
        source_pdf = source.get("source_pdf")
        pdf_page = source.get("pdf_page")
        if source_pdf is not None and pdf_page is not None:
            resolved_pdf = Path(source_pdf).resolve()
            if mode in {"notation", "both"}:
                cache_key = (resolved_pdf, mode)
                if cache_key not in vector_cache:
                    vector_cache[cache_key] = extract_pdf_vector_measure_boxes(
                        cache_key[0], mode
                    )
                boxes = vector_cache[cache_key].get(int(pdf_page), [])
            elif mode == "tab":
                if resolved_pdf not in tab_measure_cache:
                    tab_measure_cache[resolved_pdf] = (
                        extract_pdf_vector_tab_measure_boxes(resolved_pdf)
                    )
                boxes = tab_measure_cache[resolved_pdf].get(int(pdf_page), [])
                if not boxes:
                    if resolved_pdf not in tab_system_cache:
                        tab_system_cache[resolved_pdf] = extract_pdf_vector_tab_systems(
                            resolved_pdf
                        )
                    boxes = _hybrid_tab_pdf_boxes(
                        page,
                        _measure_boxes(page, mode),
                        tab_system_cache[resolved_pdf].get(int(pdf_page), []),
                    )
        if not boxes:
            boxes = _measure_boxes(page, mode)
            for box in boxes:
                box["geometry_source"] = "pixel_staff_fallback"
        if not boxes:
            raise ValueError(f"No {mode} measures detected on page {page_index}")
        overlay = page.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for box in boxes:
            crop_path = output / "measure_crops" / f"m{measure_number:04d}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            _crop(page, box["bbox"]).save(crop_path, format="PNG", compress_level=3)
            left, top, width, height = box["bbox"]
            draw.rectangle((left, top, left + width, top + height), outline=(220, 35, 35), width=2)
            draw.text((left + 2, top + 2), str(measure_number), fill=(220, 35, 35))
            records.append({
                "measure_number": measure_number,
                "page": page_index,
                "system_index": box["system_index"],
                "system_measure_index": box["system_measure_index"],
                "bbox": box["bbox"],
                "geometry_source": box["geometry_source"],
                "image": str(crop_path.resolve()),
                "source_page": str(Path(source["image"]).resolve()),
            })
            measure_number += 1
        overlay_path = output / "overlays" / f"page_{page_index:03d}.png"
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(overlay_path, format="PNG", compress_level=3)
    return mode, records


def recognize_crops(
    records: list[dict[str, Any]],
    mode: str,
    model_path: Path,
    adapter_path: Path | None,
    device: str,
    max_new_tokens: int,
    max_new_tokens_ceiling: int,
    tuning: list[int],
    maximum_attempts: int,
    diagnostics_path: Path,
    resume: bool,
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.float16, trust_remote_code=True
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(device).eval()
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    if diagnostics_path.is_file() and not resume:
        diagnostics_path.unlink()
    accepted: dict[int, dict[str, Any]] = {}
    if diagnostics_path.is_file():
        for line in diagnostics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("accepted"):
                accepted[int(value["measure_number"])] = value

    targets: list[str] = []
    for record in records:
        measure_number = int(record["measure_number"])
        if measure_number not in accepted:
            break
        value = accepted[measure_number]
        target = str(value["target"])
        _parsed, constraint_errors = validate_measure_target(
            target,
            mode,
            tuning=tuning,
            string_count=len(tuning),
        )
        if constraint_errors:
            raise ValueError(
                f"Invalid accepted checkpoint for measure {measure_number}: "
                f"{constraint_errors}"
            )
        previous_context = (
            "START" if not targets else format_previous_measure_context(targets[-1], mode)
        )
        targets.append(target)
        record["target"] = target
        record["previous_context"] = previous_context
        record["recognition_attempts"] = int(value["attempt"])
    if targets:
        print(f"[resume {len(targets)}/{len(records)}] accepted measures", flush=True)

    with diagnostics_path.open("a", encoding="utf-8") as diagnostics:
        for index in range(len(targets) + 1, len(records) + 1):
            record = records[index - 1]
            previous_context = (
                "START"
                if not targets
                else format_previous_measure_context(targets[-1], mode)
            )
            messages: list[dict[str, Any]] = [{
                "role": "user",
                "content": [
                    {"type": "image", "url": record["image"]},
                    {
                        "type": "text",
                        "text": recognition_prompt(mode, previous_context),
                    },
                ],
            }]
            target = ""
            constraint_errors: list[str] = []
            active_time_signature = _active_time_signature(targets)
            token_budget = max_new_tokens
            for attempt in range(1, maximum_attempts + 1):
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                inputs.pop("token_type_ids", None)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=token_budget, do_sample=False
                    )
                generated_token_count = int(
                    generated.shape[1] - inputs["input_ids"].shape[1]
                )
                hit_token_limit = generated_token_count >= token_budget
                value = processor.decode(
                    generated[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                start = value.find("M2")
                target = value[start:].strip() if start >= 0 else value
                target, deterministic_repairs = _repair_truncated_optional_text(target)
                _parsed, constraint_errors = validate_measure_target(
                    target,
                    mode,
                    tuning=tuning,
                    string_count=len(tuning),
                )
                if (
                    not constraint_errors
                    and deterministic_repairs
                    and hit_token_limit
                    and attempt < maximum_attempts
                ):
                    constraint_errors = [
                        "generation reached max_new_tokens inside optional text"
                    ]
                accepted_attempt = not constraint_errors
                diagnostics.write(json.dumps({
                    "measure_number": int(record["measure_number"]),
                    "mode": mode,
                    "image": record["image"],
                    "attempt": attempt,
                    "token_budget": token_budget,
                    "generated_token_count": generated_token_count,
                    "hit_token_limit": hit_token_limit,
                    "raw": value,
                    "target": target,
                    "deterministic_repairs": deterministic_repairs,
                    "constraint_errors": constraint_errors,
                    "accepted": accepted_attempt,
                }, ensure_ascii=False) + "\n")
                diagnostics.flush()
                if accepted_attempt:
                    break
                if attempt < maximum_attempts:
                    if constraint_errors == [
                        "generation reached max_new_tokens inside optional text"
                    ]:
                        correction = (
                            "The optional text annotation exhausted the output limit. Ignore all "
                            "text, section, and chord-name annotations. Return the same musical "
                            "events as one complete M2 fragment without any text effect."
                        )
                    else:
                        correction = (
                            "Correct the M2 using the same image. Constraint errors: "
                            + "; ".join(constraint_errors[:8])
                            + ". Return only one corrected M2 fragment."
                        )
                        if hit_token_limit:
                            token_budget = min(
                                max_new_tokens_ceiling, token_budget * 2
                            )
                    messages.extend([
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": target}],
                        },
                        {
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": correction,
                            }],
                        },
                    ])
            if constraint_errors:
                target = _full_measure_rest_target(active_time_signature)
                _parsed, fallback_errors = validate_measure_target(
                    target,
                    mode,
                    tuning=tuning,
                    string_count=len(tuning),
                )
                if fallback_errors:
                    raise ValueError(
                        f"Measure {record['measure_number']} failed M2 constraints after "
                        f"{maximum_attempts} attempts and its rest fallback was invalid: "
                        f"{fallback_errors}"
                    )
                diagnostics.write(json.dumps({
                    "measure_number": int(record["measure_number"]),
                    "mode": mode,
                    "image": record["image"],
                    "attempt": maximum_attempts + 1,
                    "token_budget": 0,
                    "generated_token_count": 0,
                    "hit_token_limit": False,
                    "raw": "",
                    "target": target,
                    "deterministic_repairs": ["fallback_full_measure_rest"],
                    "constraint_errors": [],
                    "fallback_reason": constraint_errors,
                    "accepted": True,
                }, ensure_ascii=False) + "\n")
                diagnostics.flush()
            targets.append(target)
            record["target"] = target
            record["previous_context"] = previous_context
            record["recognition_attempts"] = attempt
            print(
                f"[{index}/{len(records)}] measure {record['measure_number']}",
                flush=True,
            )
    return targets


def _apply_document_metadata(
    targets: list[str], mode: str, metadata: dict[str, Any]
) -> list[str]:
    if not targets or not metadata.get("tempo_quarter"):
        return targets
    first = parse_measure_target(targets[0])
    first["tempo_quarter"] = int(metadata["tempo_quarter"])
    return [format_measure_target(first, mode), *targets[1:]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recognize regular Guitar Pro/TuxGuitar PDF pages as M2 and GP5."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "tab", "notation", "both"), default="auto")
    parser.add_argument("--model", type=Path, default=Path("tools/models/GLM-OCR"))
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("weights/glm_ocr_measure_sequence_v2_lora"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--max-new-tokens-ceiling",
        type=int,
        default=2048,
        help=(
            "upper bound for adaptive structural retries; unterminated optional "
            "text retries retain --max-new-tokens"
        ),
    )
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from accepted rows in output/recognition.jsonl",
    )
    parser.add_argument("--force-pdf-render", action="store_true")
    parser.add_argument("--title")
    parser.add_argument("--artist")
    parser.add_argument("--tuning")
    parser.add_argument("--capo", type=int, default=0)
    args = parser.parse_args()
    if args.max_new_tokens_ceiling < args.max_new_tokens:
        parser.error("--max-new-tokens-ceiling must be >= --max-new-tokens")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mode, records = prepare_document_crops(
        args.inputs, output, args.mode, args.force_pdf_render
    )
    pdf_inputs = [path for path in args.inputs if path.suffix.lower() == ".pdf"]
    document_metadata = (
        extract_pdf_vector_metadata(pdf_inputs[0])
        if len(args.inputs) == 1 and len(pdf_inputs) == 1 else {}
    )
    tuning_values = (
        [int(value) for value in args.tuning.split(",") if value.strip()]
        if args.tuning
        else document_metadata.get("tuning_midi_high_to_low") or list(DEFAULT_TUNING)
    )
    targets = recognize_crops(
        records,
        mode,
        args.model.resolve(),
        args.adapter.resolve() if args.adapter else None,
        args.device,
        args.max_new_tokens,
        args.max_new_tokens_ceiling,
        tuning_values,
        args.maximum_attempts,
        output / "recognition.jsonl",
        args.resume,
    )
    targets = _apply_document_metadata(targets, mode, document_metadata)
    m2_path = output / "prediction.m2"
    m2_path.write_text("\n".join(targets) + "\n", encoding="utf-8")
    gp5_path = output / "PRE.gp5"
    write_targets_gp5(
        targets,
        gp5_path,
        mode=mode,
        title=args.title or document_metadata.get("title") or args.inputs[0].stem,
        artist=args.artist if args.artist is not None else document_metadata.get("artist") or "",
        tuning=tuning_values,
        capo=args.capo,
    )
    manifest = {
        "schema_version": "2.0",
        "mode": mode,
        "measures": len(records),
        "m2": str(m2_path),
        "gp5": str(gp5_path),
        "recognition_log": str(output / "recognition.jsonl"),
        "document_metadata": document_metadata,
        "tuning_used": tuning_values,
        "records": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("mode", "measures", "m2", "gp5")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
