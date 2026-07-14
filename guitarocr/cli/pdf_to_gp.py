from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

from guitarocr.export.export_score_ir_to_gp import export_ir, parse_tuning_arg
from guitarocr.paths import PROJECT_ROOT
from guitarocr.pipeline.infer_tuxguitar_score_tab_document import expand_inputs
from guitarocr.pipeline.infer_tuxguitar_tab_page import detect_tab_geometry
from guitarocr.pipeline.score_tab_geometry import detect_score_tab_geometry


def main() -> None:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description=(
            "Run the TuxGuitar-focused PDF recognition pipeline and export all recognized "
            "voices to a GP5 file."
        )
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--preview-pdf", type=Path)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--reuse-ir", action="store_true")
    parser.add_argument("--tempo", type=int)
    parser.add_argument("--tuning", type=parse_tuning_arg)
    parser.add_argument("--title")
    parser.add_argument("--initial-time-signature")
    parser.add_argument("--first-measure-number", type=int, default=1)
    parser.add_argument("--force-pdf-render", action="store_true")
    parser.add_argument(
        "--layout",
        choices=("auto", "score_tab", "tab_only"),
        default="auto",
        help="Detect score+TAB versus pure TAB from the first page, or force a layout.",
    )
    args = parser.parse_args()

    input_pdf = args.pdf.resolve()
    output = args.output.resolve()
    if not input_pdf.is_file() or input_pdf.suffix.lower() != ".pdf":
        parser.error(f"Input is not a PDF file: {input_pdf}")
    if output.suffix.lower() != ".gp5":
        parser.error("The current end-to-end writer outputs .gp5 only")
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir
        else root / "database" / "end_to_end" / input_pdf.stem
    )
    score_ir = work_dir / "document_score_ir.json"
    inference_stdout = ""
    selected_layout = args.layout
    if not args.reuse_ir or not score_ir.is_file():
        if selected_layout == "auto":
            detection_pages = expand_inputs(
                [input_pdf],
                work_dir / "rendered_pages",
                root / "database" / "tmp" / "pdfs",
                args.force_pdf_render,
            )
            if not detection_pages:
                raise RuntimeError("PDF rendering produced no pages")
            with Image.open(detection_pages[0]["image"]) as opened:
                first_page = opened.convert("L")
            if detect_score_tab_geometry(first_page):
                selected_layout = "score_tab"
            elif detect_tab_geometry(first_page):
                selected_layout = "tab_only"
            else:
                raise RuntimeError(
                    "Could not detect a supported TuxGuitar score_tab or tab_only layout"
                )
        inference_module = (
            "guitarocr.pipeline.infer_tuxguitar_score_tab_document"
            if selected_layout == "score_tab"
            else "guitarocr.pipeline.infer_tuxguitar_tab_document"
        )
        command = [
            sys.executable,
            "-m",
            inference_module,
            str(input_pdf),
            "--output",
            str(work_dir),
            "--first-measure-number",
            str(args.first_measure_number),
        ]
        if args.initial_time_signature:
            command.extend(["--initial-time-signature", args.initial_time_signature])
        if args.force_pdf_render:
            command.append("--force-pdf-render")
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        inference_stdout = completed.stdout.strip()
        if completed.returncode:
            print(completed.stdout, end="", file=sys.stderr)
            print(completed.stderr, end="", file=sys.stderr)
            raise SystemExit(completed.returncode)
    if not score_ir.is_file():
        raise FileNotFoundError(f"Inference did not produce {score_ir}")

    if selected_layout == "auto":
        reused_document = json.loads(score_ir.read_text(encoding="utf-8"))
        selected_layout = reused_document.get("document", {}).get("layout", "score_tab")

    preview = None
    if not args.no_preview:
        preview = args.preview_pdf or root / "output" / "pdf" / f"{output.stem}_preview.pdf"
    report = export_ir(
        score_ir,
        output,
        preview=preview,
        voice=None,
        tempo=args.tempo,
        tuning=args.tuning,
        title=args.title or input_pdf.stem,
        preview_layout=selected_layout,
    )
    report["source_pdf"] = str(input_pdf)
    report["layout"] = selected_layout
    report["inference_work_dir"] = str(work_dir)
    if inference_stdout:
        try:
            report["inference"] = json.loads(inference_stdout.splitlines()[-1])
        except json.JSONDecodeError:
            report["inference_stdout"] = inference_stdout
    else:
        reused = json.loads(score_ir.read_text(encoding="utf-8"))
        selected_layout = reused.get("document", {}).get("layout", selected_layout)
        report["layout"] = selected_layout
        reused_measures = reused.get("tracks", [{}])[0].get("measures", [])
        report["inference"] = {
            "reused_ir": True,
            "pages": reused.get("document", {}).get("page_count"),
            "measures": len(reused_measures),
            "events": sum(len(measure.get("events", [])) for measure in reused_measures),
            "score_ir": str(score_ir),
        }
    Path(report["report_file"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
