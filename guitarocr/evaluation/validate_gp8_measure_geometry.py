from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import pymupdf
from PIL import Image

from guitarocr.pipeline.infer_glm_ocr_document import (
    _hybrid_tab_pdf_boxes,
    _measure_boxes,
)
from guitarocr.pipeline.pdf_vector_measures import (
    extract_pdf_vector_measure_boxes,
    extract_pdf_vector_tab_measure_boxes,
    extract_pdf_vector_tab_systems,
)


def _expected_pages(manifest: Path) -> tuple[list[str], dict[str, Any]]:
    order: list[str] = []
    counts: dict[str, Any] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_id = row["source_id"]
            if source_id not in order:
                order.append(source_id)
            counts[source_id][row["mode"]][int(row["page"])] += 1
    return order, counts


def _tab_page_counts(pdf: Path) -> dict[int, int]:
    text_boxes = extract_pdf_vector_tab_measure_boxes(pdf)
    vector_systems = extract_pdf_vector_tab_systems(pdf)
    result = {}
    with pymupdf.open(pdf) as document:
        for page_number, page in enumerate(document, start=1):
            if text_boxes.get(page_number):
                result[page_number] = len(text_boxes[page_number])
                continue
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(2.5, 2.5),
                alpha=False,
                colorspace=pymupdf.csGRAY,
            )
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("L")
            boxes = _hybrid_tab_pdf_boxes(
                image,
                _measure_boxes(image, "tab"),
                vector_systems.get(page_number, []),
            )
            result[page_number] = len(boxes)
    return result


def validate(dataset: Path, modes: list[str]) -> dict[str, Any]:
    source_ids, expected = _expected_pages(dataset / "manifests" / "test.jsonl")
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": str(dataset.resolve()),
        "split": "test",
        "sources": len(source_ids),
        "modes": {},
    }
    for mode in modes:
        started = perf_counter()
        songs_exact = pages_exact = pages_total = 0
        measures_actual = measures_expected = 0
        failures = []
        for source_id in source_ids:
            expected_pages = dict(sorted(expected[source_id][mode].items()))
            pdf = dataset / "pdf" / mode / f"{source_id}.pdf"
            if mode == "tab":
                actual_pages = _tab_page_counts(pdf)
            else:
                actual_pages = {
                    page: len(boxes)
                    for page, boxes in extract_pdf_vector_measure_boxes(
                        pdf, mode
                    ).items()
                }
            page_numbers = set(expected_pages) | set(actual_pages)
            songs_exact += actual_pages == expected_pages
            pages_exact += sum(
                expected_pages.get(page) == actual_pages.get(page)
                for page in page_numbers
            )
            pages_total += len(page_numbers)
            measures_actual += sum(actual_pages.values())
            measures_expected += sum(expected_pages.values())
            if actual_pages != expected_pages:
                failures.append({
                    "source_id": source_id,
                    "expected": expected_pages,
                    "actual": actual_pages,
                })
        report["modes"][mode] = {
            "songs_exact": songs_exact,
            "songs_total": len(source_ids),
            "pages_exact": pages_exact,
            "pages_total": pages_total,
            "measures_actual": measures_actual,
            "measures_expected": measures_expected,
            "failures": failures,
            "seconds": round(perf_counter() - started, 3),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate source-disjoint GP8 PDF measure geometry."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("database/gp8_measure_sequence_v1"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("tab", "notation", "both"),
        default=["notation", "both", "tab"],
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(args.dataset.resolve(), args.modes)
    value = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(value, encoding="utf-8")
    print(value, end="")
    if any(mode["failures"] for mode in report["modes"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
