from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from PIL import Image

from guitarocr.pipeline.infer_tuxguitar_tab_page import detect_tab_geometry
from guitarocr.pipeline.score_tab_geometry import detect_score_tab_geometry


def evaluate_split(dataset: Path, split: str, modes: list[str]) -> dict:
    annotation_path = dataset / "annotations" / f"instance_{split}.json"
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_names = {item["id"]: item["name"] for item in coco["categories"]}
    measure_category = next(
        category_id for category_id, name in category_names.items() if name == "measure"
    )
    expected_by_image: dict[int, int] = defaultdict(int)
    for annotation in coco["annotations"]:
        if annotation["category_id"] == measure_category:
            expected_by_image[annotation["image_id"]] += 1

    mode_results: dict[str, dict] = {}
    for mode in modes:
        detector = detect_tab_geometry if mode == "tab" else detect_score_tab_geometry
        pages: list[dict] = []
        for image_record in coco["images"]:
            if not image_record["file_name"].startswith(f"{mode}_"):
                continue
            image_path = dataset / "images" / image_record["file_name"]
            with Image.open(image_path) as opened:
                systems = detector(opened.convert("L"))
            predicted = sum(len(system["measures"]) for system in systems)
            expected = expected_by_image[image_record["id"]]
            pages.append(
                {
                    "file_name": image_record["file_name"],
                    "source_group": image_record.get("source_group"),
                    "expected_measures": expected,
                    "predicted_measures": predicted,
                    "measure_error": predicted - expected,
                    "detected_systems": len(systems),
                }
            )

        expected_total = sum(page["expected_measures"] for page in pages)
        predicted_total = sum(page["predicted_measures"] for page in pages)
        failures = [page for page in pages if page["measure_error"] != 0]
        mode_results[mode] = {
            "pages": len(pages),
            "detected_pages": sum(page["detected_systems"] > 0 for page in pages),
            "page_exact": len(pages) - len(failures),
            "page_exact_rate": (len(pages) - len(failures)) / max(1, len(pages)),
            "expected_measures": expected_total,
            "predicted_measures": predicted_total,
            "absolute_measure_count_error": sum(abs(page["measure_error"]) for page in pages),
            "failures": failures,
        }
    return {
        "dataset": str(dataset),
        "split": split,
        "modes": mode_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pixel-only Guitar Pro TAB and score+TAB page geometry."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--modes", nargs="+", choices=("tab", "both"), default=["tab", "both"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    result = evaluate_split(args.dataset.resolve(), args.split, args.modes)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.fail_on_error and any(
        mode["absolute_measure_count_error"] for mode in result["modes"].values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
