from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from guitarocr.paths import DATABASE_ROOT


def expected_symbols(song: dict) -> list[tuple]:
    expected: list[tuple] = []
    for measure in song["measures"]:
        for beat_index, beat in enumerate(measure["beats"]):
            for voice in beat["voices"]:
                for note_index, note in enumerate(voice["notes"]):
                    if note["tied"]:
                        continue
                    common = (
                        measure["index"], beat_index, voice["index"], note_index,
                        note["string"], note["fret"],
                    )
                    if note["effects"]["dead"]:
                        expected.append(("dead_x", *common))
                    else:
                        expected.extend((f"digit_{digit}", *common) for digit in str(note["fret"]))
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TuxGuitar page-coordinate annotations.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    args = parser.parse_args()
    database = args.database.resolve()

    source_records = [
        json.loads(line)
        for line in (database / "manifests" / "sources.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures: list[str] = []
    totals = Counter()
    class_counts = Counter()
    for source in source_records:
        source_id = source["id"]
        song = json.loads((database / source["label_json"]).read_text(encoding="utf-8"))
        page_paths = sorted((database / "labels" / "pages" / "tab_only" / source_id).glob("page_*.json"))
        image_paths = sorted((database / "output" / "images" / "tab_only" / source_id).glob("page_*.png"))
        if len(page_paths) != len(image_paths):
            failures.append(f"{source_id}: page labels={len(page_paths)}, images={len(image_paths)}")
            continue

        actual: list[tuple] = []
        measure_indices: list[int] = []
        for page_path in page_paths:
            page = json.loads(page_path.read_text(encoding="utf-8"))
            if page["source_id"] != source_id:
                failures.append(f"{page_path}: wrong source_id {page['source_id']}")
            width = page["image_size"]["width"]
            height = page["image_size"]["height"]
            totals["pages"] += 1
            for measure in page["measures"]:
                measure_indices.append(measure["measure_index"])
                totals["measures"] += 1
                string_y = measure["tab_staff"]["string_y"]
                for symbol in measure["symbols"]:
                    totals["symbols"] += 1
                    class_counts[symbol["class"]] += 1
                    x, y, box_width, box_height = symbol["bbox"]
                    if box_width <= 0 or box_height <= 0 or x + box_width < 0 or y + box_height < 0 \
                            or x >= width or y >= height:
                        failures.append(f"{page_path}: invalid {symbol['class']} bbox {symbol['bbox']}")
                    string_index = symbol["string"] - 1
                    if not 0 <= string_index < len(string_y):
                        failures.append(f"{page_path}: invalid string {symbol['string']}")
                    elif abs(symbol["center"][1] - string_y[string_index]) > 0.01:
                        failures.append(
                            f"{page_path}: {symbol['class']} center is not on string {symbol['string']}"
                        )
                    actual.append(
                        (
                            symbol["class"], measure["measure_index"], symbol["beat_index"],
                            symbol["voice_index"], symbol["note_index"], symbol["string"], symbol["fret"],
                        )
                    )

        expected = expected_symbols(song)
        if sorted(actual) != sorted(expected):
            failures.append(f"{source_id}: expected {len(expected)} symbols, found {len(actual)}")
        expected_measures = list(range(source["measure_count"]))
        if sorted(measure_indices) != expected_measures:
            failures.append(
                f"{source_id}: expected measures {expected_measures}, found {sorted(measure_indices)}"
            )

    if failures:
        print("Annotation validation failed:")
        for failure in failures[:50]:
            print(f"- {failure}")
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "status": "passed",
                "sources": len(source_records),
                "pages": totals["pages"],
                "measures": totals["measures"],
                "symbols": totals["symbols"],
                "classes": dict(sorted(class_counts.items())),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
