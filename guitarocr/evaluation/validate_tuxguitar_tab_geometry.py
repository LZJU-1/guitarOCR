from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from guitarocr.paths import DATABASE_ROOT
from guitarocr.pipeline.infer_tuxguitar_tab_page import detect_tab_geometry


def expected_systems(page: dict) -> list[dict]:
    grouped: dict[tuple[float, ...], list[dict]] = {}
    for measure in page["measures"]:
        key = tuple(measure["tab_staff"]["string_y"])
        grouped.setdefault(key, []).append(measure)
    systems: list[dict] = []
    for string_y, measures in grouped.items():
        measures.sort(key=lambda measure: measure["bbox"][0])
        boundaries = [measure["bbox"][0] for measure in measures]
        last = measures[-1]["bbox"]
        boundaries.append(last[0] + last[2])
        systems.append({"string_y": list(string_y), "boundaries": boundaries})
    return sorted(systems, key=lambda system: system["string_y"][0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pixel-only TuxGuitar TAB staff/measure geometry.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    args = parser.parse_args()
    database = args.database.resolve()
    failures: list[str] = []
    page_count = staff_count = measure_count = 0

    for label_path in sorted((database / "labels" / "pages" / "tab_only").glob("*/*.json")):
        page = json.loads(label_path.read_text(encoding="utf-8"))
        with Image.open(database / page["image"]) as opened:
            detected = detect_tab_geometry(opened.convert("L"))
        expected = expected_systems(page)
        page_count += 1
        if len(detected) != len(expected):
            failures.append(f"{label_path}: expected {len(expected)} staffs, detected {len(detected)}")
            continue
        for index, (actual_staff, expected_staff) in enumerate(zip(detected, expected)):
            if len(actual_staff["string_y"]) != len(expected_staff["string_y"]):
                failures.append(f"{label_path} staff {index}: wrong string count")
                continue
            line_error = max(
                abs(actual - truth)
                for actual, truth in zip(actual_staff["string_y"], expected_staff["string_y"])
            )
            if line_error > 1.5:
                failures.append(f"{label_path} staff {index}: line error {line_error:.2f}px")
            if len(actual_staff["boundaries"]) != len(expected_staff["boundaries"]):
                failures.append(
                    f"{label_path} staff {index}: expected {len(expected_staff['boundaries'])} boundaries, "
                    f"detected {len(actual_staff['boundaries'])}"
                )
                continue
            boundary_error = max(
                abs(actual - truth)
                for actual, truth in zip(actual_staff["boundaries"], expected_staff["boundaries"])
            )
            if boundary_error > 16.0:
                failures.append(f"{label_path} staff {index}: boundary error {boundary_error:.2f}px")
            staff_count += 1
            measure_count += len(actual_staff["measures"])

    result = {
        "pages": page_count,
        "staffs": staff_count,
        "measures": measure_count,
        "failures": len(failures),
    }
    print(json.dumps(result))
    if failures:
        for failure in failures[:50]:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
