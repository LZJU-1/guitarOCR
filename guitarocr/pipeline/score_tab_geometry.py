from __future__ import annotations

import numpy as np
from PIL import Image

from guitarocr.pipeline.infer_tuxguitar_tab_page import detect_tab_geometry


def detect_score_tab_geometry(image: Image.Image) -> list[dict]:
    """Pair five-line score staffs with the following TAB staff using pixels only.

    The underlying horizontal-run detector deliberately returns both five-line
    and TAB staffs. Measure boundaries are taken from TAB, where stems cannot be
    mistaken for barlines.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    black = gray < 160
    raw_staffs = detect_tab_geometry(image)
    # At the fixed TuxGuitar render scale, score spacing is about 17.5 pixels
    # and TAB spacing is 20 pixels. Standard-score chains may accidentally gain
    # ledger lines, so identify the clean TAB chains first and recover the five
    # score lines independently inside the TAB horizontal span.
    tab_staffs = [
        staff for staff in raw_staffs
        if staff["string_count"] >= 4 and staff["spacing"] >= 18.75
    ]
    tab_staffs.sort(key=lambda item: item["string_y"][0])
    systems: list[dict] = []
    measure_number = 1
    previous_tab_bottom = -1.0
    for tab in tab_staffs:
        boundaries = tab["boundaries"]
        left = max(0, int(boundaries[0]))
        right = min(black.shape[1], int(boundaries[-1]) + 1)
        if right - left < 30:
            continue
        longest_runs = np.zeros(black.shape[0], dtype=np.int32)
        for row_index, row in enumerate(black[:, left:right]):
            edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
            if edges.size:
                longest_runs[row_index] = int((edges[1::2] - edges[::2]).max(initial=0))

        expected_spacing = float(tab["spacing"]) * 0.875
        search_top = max(0, int(previous_tab_bottom + 2.0 * tab["spacing"]))
        search_bottom = int(tab["string_y"][0] - 2.0 * tab["spacing"] - 4.0 * expected_spacing)
        best: tuple[float, list[float]] | None = None
        spacing_values = np.arange(expected_spacing - 1.5, expected_spacing + 1.51, 0.25)
        minimum_run = max(25.0, (right - left) * 0.045)
        for spacing in spacing_values:
            for start in range(search_top, max(search_top, search_bottom) + 1):
                positions: list[float] = []
                strengths: list[float] = []
                for line_index in range(5):
                    expected_y = start + line_index * float(spacing)
                    center = int(round(expected_y))
                    low = max(0, center - 1)
                    high = min(len(longest_runs), center + 2)
                    local = longest_runs[low:high]
                    local_index = int(local.argmax()) if local.size else 0
                    positions.append(float(low + local_index))
                    strengths.append(float(local[local_index]) if local.size else 0.0)
                if min(strengths) < minimum_run:
                    continue
                score = sum(strengths) + 2.0 * min(strengths)
                if best is None or score > best[0]:
                    best = (score, positions)
        if best is None:
            previous_tab_bottom = tab["string_y"][-1]
            continue
        score_line_y = best[1]
        score_spacing = (score_line_y[-1] - score_line_y[0]) / 4.0
        measures: list[dict] = []
        score_top = score_line_y[0] - score_spacing / 2.0
        score_height = score_line_y[-1] - score_line_y[0] + score_spacing
        for local_index in range(len(boundaries) - 1):
            left, right = boundaries[local_index], boundaries[local_index + 1]
            measures.append(
                {
                    "measure_number": measure_number,
                    "system_measure_index": local_index,
                    "bbox": [float(left), float(score_top), float(right - left), float(score_height)],
                    "events": [],
                }
            )
            measure_number += 1
        systems.append(
            {
                "system_index": len(systems),
                "score_line_y": score_line_y,
                "score_spacing": score_spacing,
                "tab_string_y": [float(value) for value in tab["string_y"]],
                "tab_spacing": float(tab["spacing"]),
                "boundaries": [float(value) for value in boundaries],
                "measures": measures,
            }
        )
        previous_tab_bottom = tab["string_y"][-1]
    return systems
