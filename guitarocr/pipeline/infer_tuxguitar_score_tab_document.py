from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch

from guitarocr.models.rhythm_context_model import RhythmContextCNN
from guitarocr.models.score_event_locator_model import ScoreEventLocator
from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.paths import DATABASE_ROOT, PROJECT_ROOT, WEIGHTS_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_page import (
    classify_rhythm,
    draw_overlay,
    locate_page_events,
    parse_time_signature,
)
from guitarocr.pipeline.measure_rhythm_constraints import audit_score_ir
from guitarocr.pipeline.pdf_page_renderer import MODEL_RENDER_DPI, render_pdf_pages
from guitarocr.pipeline.score_tab_fingering import build_score_ir, detect_tab_fingering, resolve_unambiguous_ties
from guitarocr.pipeline.tie_inference import classify_ties, load_tie_model
from guitarocr.pipeline.time_signature_recognizer import load_atomic_model, propagate_time_signatures


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def expand_inputs(
    inputs: list[Path], render_root: Path, temp_root: Path, force_pdf_render: bool
) -> list[dict]:
    values: list[Path] = []
    for supplied in inputs:
        if supplied.is_dir():
            values.extend(
                path for path in supplied.iterdir()
                if path.is_file() and (path.suffix.lower() in IMAGE_SUFFIXES or path.suffix.lower() == ".pdf")
            )
        elif supplied.is_file() and (
            supplied.suffix.lower() in IMAGE_SUFFIXES or supplied.suffix.lower() == ".pdf"
        ):
            values.append(supplied)
        else:
            raise FileNotFoundError(f"Not a supported PDF, page image, or directory: {supplied}")
    unique = sorted(
        {path.resolve(): path.resolve() for path in values}.values(),
        key=lambda path: (path.parent.as_posix().lower(), path.name.lower()),
    )
    pages: list[dict] = []
    for value in unique:
        if value.suffix.lower() != ".pdf":
            pages.append({"image": value, "source_pdf": None, "pdf_page": None})
            continue
        identity = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
        pdf_output = render_root / f"{value.stem}_{identity}"
        rendered = render_pdf_pages(
            value,
            pdf_output,
            temp_root,
            dpi=MODEL_RENDER_DPI,
            force=force_pdf_render,
        )
        pages.extend(
            {"image": path, "source_pdf": value, "pdf_page": index}
            for index, path in enumerate(rendered, start=1)
        )
    return pages


def load_models(args: argparse.Namespace, device: torch.device) -> dict:
    locator_checkpoint = torch.load(args.locator_model, map_location=device, weights_only=False)
    locator = ScoreEventLocator().to(device)
    locator.load_state_dict(locator_checkpoint["model_state"])
    locator.eval()

    rhythm_checkpoint = torch.load(args.rhythm_model, map_location=device, weights_only=False)
    rhythm = RhythmContextCNN().to(device)
    rhythm.load_state_dict(rhythm_checkpoint["model_state"])
    rhythm.eval()

    tab_checkpoint = torch.load(args.tab_model, map_location=device, weights_only=False)
    tab_classes = tab_checkpoint["classes"]
    tab = TabSymbolDetector(len(tab_classes)).to(device)
    tab.load_state_dict(tab_checkpoint["model_state"])
    tab.eval()

    atomic, atomic_classes = load_atomic_model(args.atomic_model, device)
    tie, tie_threshold = load_tie_model(args.tie_model, device)
    return {
        "locator": locator,
        "locator_threshold": (
            args.threshold
            if args.threshold is not None
            else float(locator_checkpoint.get("detection_threshold", 0.3))
        ),
        "rhythm": rhythm,
        "tab": tab,
        "tab_classes": tab_classes,
        "atomic": atomic,
        "atomic_classes": atomic_classes,
        "tie": tie,
        "tie_threshold": tie_threshold,
    }


def main() -> None:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Run the pixel-only TuxGuitar score+TAB pipeline on PDF files or ordered page images, "
            "carrying time signatures and measure numbers across pages."
        )
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="PDF files, page images, or directories, processed lexically at the model's fixed 180 DPI.",
    )
    parser.add_argument("--output", type=Path)
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
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--time-signature-threshold", type=float, default=0.20)
    parser.add_argument("--initial-time-signature", type=parse_time_signature)
    parser.add_argument("--first-measure-number", type=int, default=1)
    parser.add_argument("--force-pdf-render", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            root / "database" / "score_event_locator" / "document_inference" / args.inputs[0].stem
        )

    page_sources = expand_inputs(
        args.inputs,
        args.output / "rendered_pages",
        root / "database" / "tmp" / "pdfs",
        args.force_pdf_render,
    )
    if not page_sources:
        raise RuntimeError("No supported PDF or page images found")
    if args.first_measure_number < 1:
        raise ValueError("--first-measure-number must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(args, device)
    args.output.mkdir(parents=True, exist_ok=True)

    carried_signature = args.initial_time_signature
    next_measure_number = args.first_measure_number
    document_measures: list[dict] = []
    page_records: list[dict] = []
    string_count: int | None = None

    for page_index, page_source in enumerate(page_sources, start=1):
        image_path = page_source["image"]
        with Image.open(image_path) as opened:
            page = opened.convert("L")
        systems = locate_page_events(
            page, models["locator"], device, models["locator_threshold"]
        )
        if not systems:
            raise RuntimeError(f"No paired score/TAB system was detected on {image_path}")
        carried_signature = propagate_time_signatures(
            page,
            systems,
            models["atomic"],
            models["atomic_classes"],
            device,
            initial=carried_signature,
            threshold=args.time_signature_threshold,
        )
        detect_tab_fingering(
            page, systems, models["tab"], models["tab_classes"], device, threshold=0.3
        )
        page_root = args.output / f"page_{page_index:03d}"
        crop_root = page_root / "crops"
        crop_root.mkdir(parents=True, exist_ok=True)
        classify_rhythm(page, systems, models["rhythm"], device, crop_root)
        classify_ties(page, systems, models["tie"], device, models["tie_threshold"])
        page_ir = build_score_ir(systems, measure_number_offset=next_measure_number)
        audit_score_ir(page_ir)
        measures = page_ir["tracks"][0]["measures"]
        next_measure_number += len(measures)
        document_measures.extend(measures)
        if string_count is None:
            string_count = page_ir["tracks"][0]["string_count"]

        overlay_path = page_root / "overlay.png"
        draw_overlay(page, systems, overlay_path)
        page_output_ir = copy.deepcopy(page_ir)
        resolve_unambiguous_ties(page_output_ir)
        page_ir_path = page_root / "score_ir.json"
        page_ir_path.write_text(json.dumps(page_output_ir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        page_records.append(
            {
                "page": page_index,
                "image": str(image_path.resolve()),
                "source_pdf": (
                    str(page_source["source_pdf"].resolve())
                    if page_source["source_pdf"] is not None else None
                ),
                "pdf_page": page_source["pdf_page"],
                "measure_start": measures[0]["number"] if measures else None,
                "measure_end": measures[-1]["number"] if measures else None,
                "systems": len(systems),
                "measures": len(measures),
                "events": sum(len(measure["events"]) for measure in measures),
                "printed_time_signatures": sum(
                    measure["printed_time_signature"] is not None for measure in measures
                ),
                "overlay": str(overlay_path),
                "score_ir": str(page_ir_path),
            }
        )

    document_ir = {
        "schema_version": "1.0",
        "document": {
            "page_count": len(page_sources),
            "pdf_render_dpi": MODEL_RENDER_DPI,
            "pages": page_records,
        },
        "tracks": [
            {
                "track": 1,
                "string_count": string_count,
                "string_tuning_midi": None,
                "capo": None,
                "tempo_quarter": None,
                "measures": document_measures,
            }
        ],
        "scope": (
            "Ordered page-image TuxGuitar score+TAB intermediate representation. Time signatures and measure "
            "numbers are propagated across pages. Exact beat positions, ties, tuning, tempo and ambiguous "
            "multi-voice note assignment remain unresolved."
        ),
    }
    audit = audit_score_ir(document_ir)
    tie_summary = resolve_unambiguous_ties(document_ir)
    output_path = args.output / "document_score_ir.json"
    output_path.write_text(json.dumps(document_ir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pages": len(page_sources),
        "measures": len(document_measures),
        "events": sum(len(measure["events"]) for measure in document_measures),
        "printed_time_signatures": sum(
            measure["printed_time_signature"] is not None for measure in document_measures
        ),
        "rhythm_audit": audit,
        "ties": tie_summary,
        "score_ir": str(output_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
