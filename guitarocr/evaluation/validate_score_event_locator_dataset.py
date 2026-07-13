from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from guitarocr.data.build_score_event_locator_dataset import INPUT_HEIGHT, INPUT_WIDTH
from guitarocr.paths import DATABASE_ROOT


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate score event-locator tiles and source splits.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    args = parser.parse_args()
    database = args.database.resolve()
    root = database / "score_event_locator"

    sources: dict[str, str] = {}
    represented: set[tuple[str, int, int, str]] = set()
    tile_count = 0
    assignment_count = 0
    for split in ("train", "validation", "test"):
        for record in read_jsonl(root / "manifests" / f"{split}.jsonl"):
            source_id = record["source_id"]
            previous = sources.setdefault(source_id, split)
            if previous != split:
                raise ValueError(f"Source leakage: {source_id} is in {previous} and {split}")
            image_path = database / record["image"]
            label_path = database / record["label"]
            with Image.open(image_path) as image:
                if image.size != (INPUT_WIDTH, INPUT_HEIGHT):
                    raise ValueError(f"Bad image size for {image_path}: {image.size}")
            label = json.loads(label_path.read_text(encoding="utf-8"))
            if label["split"] != split or label["source_id"] != source_id:
                raise ValueError(f"Manifest/label mismatch for {record['sample_id']}")
            transform = label["transform"]
            left = float(transform["page_crop_xyxy"][0])
            scale = float(transform["scale"])
            offset = float(transform["offset_x"])
            tile_start = float(transform["tile_start_x"])
            for event in label["events"]:
                x = float(event["x"])
                if not 0.0 <= x < INPUT_WIDTH:
                    raise ValueError(f"Event outside tile in {record['sample_id']}: {x}")
                recovered = (x - offset + tile_start) / scale + left
                if abs(recovered - float(event["x_page"])) > 1e-5:
                    raise ValueError(f"Transform mismatch in {record['sample_id']}: {recovered}")
                represented.add((source_id, int(label["page_index"]), int(label["measure_number"]), event["event_id"]))
                assignment_count += 1
            tile_count += 1

    expected: set[tuple[str, int, int, str]] = set()
    for page_path in (database / "labels" / "pages" / "score_tab_rhythm").glob("*/*.json"):
        page = json.loads(page_path.read_text(encoding="utf-8"))
        for measure in page["measures"]:
            for event in measure["events"]:
                expected.add((page["source_id"], int(page["page_index"]), int(measure["measure_number"]), event["event_id"]))
    if represented != expected:
        missing = expected - represented
        extra = represented - expected
        raise ValueError(f"Event coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    report = {
        "status": "passed",
        "sources": len(sources),
        "tiles": tile_count,
        "unique_events": len(represented),
        "event_assignments": assignment_count,
        "source_leakage": 0,
    }
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
