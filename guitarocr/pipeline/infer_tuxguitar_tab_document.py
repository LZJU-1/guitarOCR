from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from guitarocr.data.build_tab_event_locator_dataset import tile_tab_measure
from guitarocr.data.build_tab_rhythm_dataset import build_tab_event_crop
from guitarocr.paths import PROJECT_ROOT, WEIGHTS_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_document import expand_inputs, load_models
from guitarocr.pipeline.fret_token_classifier import classify_event_frets
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import (
    classify_rhythm,
    map_event_to_page,
    merge_event_columns,
    parse_time_signature,
)
from guitarocr.pipeline.infer_tuxguitar_tab_page import detect_tab_geometry
from guitarocr.pipeline.measure_rhythm_constraints import (
    apply_plausible_rhythm_corrections,
    audit_score_ir,
    refine_time_signatures_from_rhythm,
)
from guitarocr.pipeline.pdf_page_renderer import MODEL_RENDER_DPI
from guitarocr.pipeline.pdf_vector_tab import (
    extract_pdf_text_spans,
    extract_pdf_vector_tempo,
)
from guitarocr.pipeline.score_tab_fingering import (
    build_score_ir,
    correct_multidigit_fret_outliers,
    detect_tab_fingering,
    recover_isolated_tab_events,
    resolve_unambiguous_ties,
)
from guitarocr.pipeline.tie_inference import classify_ties
from guitarocr.pipeline.tempo_recognizer import recognize_tempo
from guitarocr.pipeline.time_signature_recognizer import propagate_time_signatures
from guitarocr.training.train_score_event_locator import decode_events


def normalize_tab_systems(staffs: list[dict]) -> list[dict]:
    systems = []
    for staff in staffs:
        string_y = [float(value) for value in staff["string_y"]]
        spacing = float(staff["spacing"])
        # TuxGuitar prints a stacked time signature across the upper five TAB
        # lines.  This virtual staff lets the existing digit recognizer reuse
        # its geometry without pretending that a separate score staff exists.
        time_signature_lines = string_y[:5]
        measures = []
        for measure in staff["measures"]:
            copied = dict(measure)
            copied["events"] = []
            copied["tab_symbols"] = []
            copied["tab_events"] = []
            measures.append(copied)
        systems.append({
            "system_index": int(staff["staff_index"]),
            "score_line_y": time_signature_lines,
            "score_spacing": spacing,
            "tab_string_y": string_y,
            "tab_spacing": spacing,
            "boundaries": [float(value) for value in staff["boundaries"]],
            "measures": measures,
            "layout": "tab_only",
        })
    return systems


@torch.inference_mode()
def locate_tab_page_events(
    page: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float,
    batch_size: int = 32,
) -> list[dict]:
    systems = normalize_tab_systems(detect_tab_geometry(page))
    records: list[tuple[torch.Tensor, dict, dict, dict]] = []
    for system in systems:
        for measure in system["measures"]:
            for tile, _, transform in tile_tab_measure(
                page, measure["bbox"], system["tab_string_y"], []
            ):
                array = np.asarray(tile, dtype=np.float32) / 255.0
                records.append((
                    torch.from_numpy(1.0 - array).unsqueeze(0),
                    transform,
                    system,
                    measure,
                ))
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        tensors = torch.stack([record[0] for record in batch]).to(device)
        predictions = decode_events(model(tensors), threshold=threshold)
        for decoded, (_, transform, _, measure) in zip(predictions, batch):
            measure["events"].extend(
                map_event_to_page(event, transform) for event in decoded
            )
    for system in systems:
        for measure in system["measures"]:
            measure["events"] = merge_event_columns(
                measure["events"], system["tab_spacing"]
            )
    return systems


def draw_tab_overlay(page: Image.Image, systems: list[dict], output: Path) -> None:
    overlay = page.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for system in systems:
        for y in system["tab_string_y"]:
            draw.line(
                (system["boundaries"][0], y, system["boundaries"][-1], y),
                fill=(70, 160, 90), width=1,
            )
        for boundary in system["boundaries"]:
            draw.line(
                (boundary, system["tab_string_y"][0], boundary, system["tab_string_y"][-1]),
                fill=(0, 135, 230), width=1,
            )
        for measure in system["measures"]:
            for symbol in measure.get("tab_symbols", []):
                x, y, width, height = symbol["bbox"]
                color = (220, 0, 200) if symbol["class"] == "dead_x" else (225, 35, 35)
                draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
            for event in measure["events"]:
                x = float(event["x"])
                draw.line(
                    (x, system["tab_string_y"][0] - 5 * system["tab_spacing"],
                     x, system["tab_string_y"][-1] + 5 * system["tab_spacing"]),
                    fill=(255, 120, 0), width=2,
                )
                draw.text(
                    (x + 2, system["tab_string_y"][0] - 5 * system["tab_spacing"]),
                    f"{event['locator_confidence']:.2f}", fill=(190, 70, 0),
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pixel-only TuxGuitar tab_only PDF/image pipeline."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--locator-model", type=Path,
        default=WEIGHTS_ROOT / "tab_event_locator.pt",
    )
    parser.add_argument(
        "--rhythm-model", type=Path,
        default=WEIGHTS_ROOT / "tab_rhythm_context_cnn.pt",
    )
    parser.add_argument(
        "--tab-model", type=Path,
        default=WEIGHTS_ROOT / "tab_symbol_detector.pt",
    )
    parser.add_argument(
        "--fret-token-model", type=Path,
        default=WEIGHTS_ROOT / "fret_token_cnn.pt",
    )
    parser.add_argument(
        "--atomic-model", type=Path,
        default=WEIGHTS_ROOT / "atomic_symbol_cnn.pt",
    )
    parser.add_argument(
        "--tie-model", type=Path,
        default=WEIGHTS_ROOT / "tab_tie_context_cnn.pt",
    )
    parser.add_argument(
        "--technique-model", type=Path,
        default=WEIGHTS_ROOT / "tab_technique_context_cnn.pt",
    )
    parser.add_argument(
        "--pick-stroke-model", type=Path,
        default=WEIGHTS_ROOT / "tab_technique_context_cnn.pt",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--tab-threshold", type=float)
    parser.add_argument("--time-signature-threshold", type=float, default=0.20)
    parser.add_argument("--initial-time-signature", type=parse_time_signature)
    parser.add_argument("--first-measure-number", type=int, default=1)
    parser.add_argument("--force-pdf-render", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            PROJECT_ROOT / "database" / "tab_document_inference" / args.inputs[0].stem
        )
    page_sources = expand_inputs(
        args.inputs,
        args.output / "rendered_pages",
        PROJECT_ROOT / "database" / "tmp" / "pdfs",
        args.force_pdf_render,
    )
    if not page_sources:
        raise RuntimeError("No supported PDF or page images found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args, device)
    args.output.mkdir(parents=True, exist_ok=True)
    carried_signature = args.initial_time_signature
    next_measure_number = args.first_measure_number
    document_measures: list[dict] = []
    page_records: list[dict] = []
    string_count = None
    tempo_prediction = None

    for page_index, source in enumerate(page_sources, start=1):
        with Image.open(source["image"]) as opened:
            page = opened.convert("L")
        systems = locate_tab_page_events(
            page, models["locator"], device, models["locator_threshold"]
        )
        if not systems:
            raise RuntimeError(f"No TuxGuitar pure-TAB staff was detected on {source['image']}")
        if page_index == 1:
            tempo_prediction = recognize_tempo(
                page, systems[0], models["atomic"], models["atomic_classes"], device
            )
            if source["source_pdf"] is not None:
                vector_text_spans = extract_pdf_text_spans(
                    source["source_pdf"],
                    int(source["pdf_page"]),
                    page.size,
                )
                vector_tempo = extract_pdf_vector_tempo(vector_text_spans, systems)
                if vector_tempo is not None:
                    tempo_prediction = vector_tempo
        carried_signature = propagate_time_signatures(
            page, systems, models["atomic"], models["atomic_classes"], device,
            initial=carried_signature, threshold=args.time_signature_threshold,
        )
        detect_tab_fingering(
            page, systems, models["tab"], models["tab_classes"], device,
            threshold=models["tab_threshold"],
        )
        recover_isolated_tab_events(systems)
        fret_token_summary = None
        if models["fret_token"] is not None:
            fret_token_summary = classify_event_frets(
                page,
                systems,
                models["fret_token"],
                models["fret_token_classes"],
                device,
            )
        page_root = args.output / f"page_{page_index:03d}"
        crop_root = page_root / "crops"
        crop_root.mkdir(parents=True, exist_ok=True)
        classify_rhythm(
            page, systems, models["rhythm"], device, crop_root,
            models["technique"], models["technique_classes"], models["technique_thresholds"],
            models["pick_stroke"], models["pick_stroke_classes"], models["pick_stroke_thresholds"],
            crop_builder=build_tab_event_crop,
            reference_line_key="tab_string_y",
        )
        classify_ties(
            page, systems, models["tie"], device, models["tie_threshold"],
            crop_builder=build_tab_event_crop,
            reference_line_key="tab_string_y",
        )
        page_ir = build_score_ir(systems, measure_number_offset=next_measure_number)
        refine_time_signatures_from_rhythm(page_ir)
        audit_score_ir(page_ir)
        measures = page_ir["tracks"][0]["measures"]
        next_measure_number += len(measures)
        document_measures.extend(measures)
        string_count = string_count or page_ir["tracks"][0]["string_count"]
        overlay_path = page_root / "overlay.png"
        draw_tab_overlay(page, systems, overlay_path)
        page_output = copy.deepcopy(page_ir)
        resolve_unambiguous_ties(page_output)
        page_ir_path = page_root / "score_ir.json"
        page_ir_path.write_text(
            json.dumps(page_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        page_records.append({
            "page": page_index,
            "image": str(source["image"].resolve()),
            "source_pdf": str(source["source_pdf"].resolve()) if source["source_pdf"] else None,
            "pdf_page": source["pdf_page"],
            "measure_start": measures[0]["number"] if measures else None,
            "measure_end": measures[-1]["number"] if measures else None,
            "systems": len(systems),
            "measures": len(measures),
            "events": sum(len(measure["events"]) for measure in measures),
            "fret_token_fusion": fret_token_summary,
            "overlay": str(overlay_path),
            "score_ir": str(page_ir_path),
        })

    document_ir = {
        "schema_version": "1.0",
        "document": {
            "layout": "tab_only",
            "page_count": len(page_sources),
            "pdf_render_dpi": MODEL_RENDER_DPI,
            "pages": page_records,
        },
        "tracks": [{
            "track": 1,
            "string_count": string_count,
            "string_tuning_midi": [64, 59, 55, 50, 45, 40] if string_count == 6 else None,
            "capo": None,
            "tempo_quarter": tempo_prediction["tempo_quarter"] if tempo_prediction else None,
            "measures": document_measures,
        }],
        "scope": (
            "TuxGuitar tab_only pixel-only event, two-voice rhythm, fingering, "
            "tie and playing-technique IR."
        ),
    }
    time_signature_refinement = refine_time_signatures_from_rhythm(document_ir)
    audit_score_ir(document_ir)
    rhythm_corrections = apply_plausible_rhythm_corrections(document_ir)
    fret_corrections = correct_multidigit_fret_outliers(document_ir)
    tie_summary = resolve_unambiguous_ties(document_ir)
    output_path = args.output / "document_score_ir.json"
    output_path.write_text(
        json.dumps(document_ir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "layout": "tab_only",
        "pages": len(page_sources),
        "measures": len(document_measures),
        "events": sum(len(measure["events"]) for measure in document_measures),
        "rhythm_audit": document_ir["rhythm_audit_summary"],
        "rhythm_corrections": rhythm_corrections,
        "ties": tie_summary,
        "fret_corrections": fret_corrections,
        "time_signature_refinement": time_signature_refinement,
        "tempo_prediction": tempo_prediction,
        "score_ir": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
