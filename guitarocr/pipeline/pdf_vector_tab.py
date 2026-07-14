from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from statistics import median


TAB_CHARACTERS = frozenset("0123456789xX()")
VECTOR_TECHNIQUE_TEXT = {
    "sl.": "slide",
    "full": "bend",
    "1/2": "bend",
    "1/4": "bend",
    "3/4": "bend",
    "½": "bend",
    "¼": "bend",
    "¾": "bend",
    "h": "harmonic",
    "p.m.": "palm_mute",
    "let ring": "let_ring",
}

TUPLET_DIVISIONS = {
    3: "3:2",
    4: "4:2",
    5: "5:4",
    6: "6:4",
    7: "7:4",
    9: "9:8",
    10: "10:8",
    12: "12:8",
}


def _has_gp8_font_signature(spans: list[dict]) -> bool:
    fonts = {str(span.get("font", "")).lower() for span in spans}
    return "arial-0-50" in fonts and any(
        font.startswith("timesnewroman-") for font in fonts
    )


def extract_pdf_tab_glyphs(
    pdf_path: Path,
    page_number: int,
    rendered_size: tuple[int, int],
) -> list[dict]:
    """Extract positioned TAB-like glyphs from a vector PDF page.

    PyMuPDF reports PDF point coordinates while the recognition pipeline works
    on a fixed-DPI raster page.  Keeping the conversion here lets the normal
    score/TAB geometry decide which characters really lie on a TAB string;
    headers, measure numbers and time signatures are discarded later.
    """
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover - exercised by installation checks
        raise RuntimeError(
            "Vector-PDF TAB extraction requires PyMuPDF. Reinstall GuitarOCR "
            "with its declared dependencies."
        ) from error

    pdf_path = Path(pdf_path).resolve()
    if page_number < 1:
        raise ValueError("page_number is one-based and must be at least 1")
    width, height = rendered_size
    with pymupdf.open(pdf_path) as document:
        if page_number > document.page_count:
            raise ValueError(
                f"PDF page {page_number} is outside the {document.page_count}-page document"
            )
        page = document[page_number - 1]
        scale_x = float(width) / float(page.rect.width)
        scale_y = float(height) / float(page.rect.height)
        raw = page.get_text("rawdict")

    glyphs: list[dict] = []
    for block in raw.get("blocks", []):
        if int(block.get("type", -1)) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = str(span.get("font", ""))
                font_size = float(span.get("size", 0.0))
                for character in span.get("chars", []):
                    value = str(character.get("c", ""))
                    if value not in TAB_CHARACTERS:
                        continue
                    x0, y0, x1, y1 = (float(item) for item in character["bbox"])
                    bbox = [
                        x0 * scale_x,
                        y0 * scale_y,
                        x1 * scale_x,
                        y1 * scale_y,
                    ]
                    glyphs.append(
                        {
                            "char": value,
                            "bbox": bbox,
                            "x": (bbox[0] + bbox[2]) / 2.0,
                            "y": (bbox[1] + bbox[3]) / 2.0,
                            "font": font,
                            "font_size_pt": font_size,
                        }
                    )
    return glyphs


def extract_pdf_text_spans(
    pdf_path: Path,
    page_number: int,
    rendered_size: tuple[int, int],
) -> list[dict]:
    """Return positioned text spans for deterministic GP8 annotations."""
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Vector-PDF text extraction requires PyMuPDF") from error
    width, height = rendered_size
    with pymupdf.open(Path(pdf_path).resolve()) as document:
        page = document[page_number - 1]
        scale_x = float(width) / float(page.rect.width)
        scale_y = float(height) / float(page.rect.height)
        page_dict = page.get_text("dict")
        drawings = page.get_drawings()
    spans = []
    for block in page_dict.get("blocks", []):
        if int(block.get("type", -1)) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                value = str(span.get("text", "")).strip()
                if not value:
                    continue
                x0, y0, x1, y1 = (float(item) for item in span["bbox"])
                bbox = [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
                spans.append({
                    "text": value,
                    "bbox": bbox,
                    "x": (bbox[0] + bbox[2]) / 2.0,
                    "y": (bbox[1] + bbox[3]) / 2.0,
                    "font": str(span.get("font", "")),
                    "font_size_pt": float(span.get("size", 0.0)),
                })

    # Tuplet brackets are two short stroked vector paths separated by the
    # printed number. Attach their full horizontal extent to that number so
    # group membership does not have to be guessed from the glyph centre.
    strokes = []
    for drawing in drawings:
        if drawing.get("type") != "s":
            continue
        rect = drawing.get("rect")
        if rect is None:
            continue
        bbox = [
            float(rect.x0) * scale_x,
            float(rect.y0) * scale_y,
            float(rect.x1) * scale_x,
            float(rect.y1) * scale_y,
        ]
        stroke_width = bbox[2] - bbox[0]
        stroke_height = bbox[3] - bbox[1]
        if stroke_width >= 10.0 and stroke_height <= 12.0:
            strokes.append({"bbox": bbox, "x": (bbox[0] + bbox[2]) / 2.0,
                            "y": (bbox[1] + bbox[3]) / 2.0})

    for span in spans:
        text = str(span["text"]).strip()
        font = str(span.get("font", "")).lower()
        size = float(span.get("font_size_pt", 0.0))
        if not text.isdigit() or "timesnewroman-1" not in font or not (8.0 <= size <= 10.0):
            continue
        marker_x = float(span["x"])
        marker_y = float(span["y"])
        nearby = [
            stroke for stroke in strokes
            if abs(float(stroke["y"]) - marker_y) <= 18.0
        ]
        left = [
            stroke for stroke in nearby
            if float(stroke["bbox"][2]) <= marker_x + 2.0
            and marker_x - float(stroke["bbox"][2]) <= 90.0
        ]
        right = [
            stroke for stroke in nearby
            if float(stroke["bbox"][0]) >= marker_x - 2.0
            and float(stroke["bbox"][0]) - marker_x <= 90.0
        ]
        if not left or not right:
            continue
        left_stroke = min(
            left,
            key=lambda item: (
                abs(float(item["y"]) - marker_y),
                marker_x - float(item["bbox"][2]),
            ),
        )
        right_stroke = min(
            right,
            key=lambda item: (
                abs(float(item["y"]) - marker_y),
                float(item["bbox"][0]) - marker_x,
            ),
        )
        if abs(float(left_stroke["y"]) - float(right_stroke["y"])) > 3.0:
            continue
        span["tuplet_bracket_bbox"] = [
            float(left_stroke["bbox"][0]),
            min(float(left_stroke["bbox"][1]), float(right_stroke["bbox"][1])),
            float(right_stroke["bbox"][2]),
            max(float(left_stroke["bbox"][3]), float(right_stroke["bbox"][3])),
        ]
    return spans


def extract_pdf_vector_beams(
    pdf_path: Path,
    page_number: int,
    rendered_size: tuple[int, int],
) -> list[dict]:
    """Return filled quadrilateral beam paths in rendered-page coordinates.

    Guitar Pro emits beam bars as black four-line filled paths.  Noteheads use
    Bezier curves, while staff/bar lines are strokes, so this structural filter
    is substantially less ambiguous than trying to rediscover the same bars
    from raster pixels.
    """
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Vector-PDF beam extraction requires PyMuPDF") from error
    width, height = rendered_size
    with pymupdf.open(Path(pdf_path).resolve()) as document:
        page = document[page_number - 1]
        scale_x = float(width) / float(page.rect.width)
        scale_y = float(height) / float(page.rect.height)
        drawings = page.get_drawings()
    beams = []
    for drawing in drawings:
        rect = drawing.get("rect")
        fill = drawing.get("fill")
        items = drawing.get("items", [])
        if (
            rect is None
            or fill is None
            or max(float(value) for value in fill) > 0.15
            or len(items) != 4
            or any(item[0] != "l" for item in items)
        ):
            continue
        width_pt = float(rect.x1 - rect.x0)
        height_pt = float(rect.y1 - rect.y0)
        if width_pt < 2.5 or not 1.0 <= height_pt <= 20.0:
            continue
        bbox = [
            float(rect.x0) * scale_x,
            float(rect.y0) * scale_y,
            float(rect.x1) * scale_x,
            float(rect.y1) * scale_y,
        ]
        beams.append({
            "bbox": bbox,
            "width": bbox[2] - bbox[0],
            "height": bbox[3] - bbox[1],
        })
    return beams


def extract_pdf_vector_tempo(spans: list[dict], systems: list[dict]) -> dict | None:
    """Read Guitar Pro's explicit ``note = BPM`` header from PDF text.

    Tempo is unusually suitable for a deterministic path: GP8 preserves the
    number as searchable text, while raster OCR has to distinguish it from bar
    numbers and fret digits.  Restricting candidates to the header above the
    first staff keeps the text-layer path high precision.
    """
    if not spans or not systems:
        return None
    score_lines = [
        float(value)
        for system in systems
        for value in system.get("score_line_y", [])
    ]
    if not score_lines:
        return None
    header_bottom = min(score_lines)
    candidates = []
    for span in spans:
        if float(span["y"]) >= header_bottom:
            continue
        match = re.search(r"=\s*(\d{2,3})\b", str(span["text"]))
        if match is None:
            continue
        value = int(match.group(1))
        if 20 <= value <= 400:
            candidates.append((float(span["y"]), value, span))
    if not candidates:
        return None
    _y, value, span = min(candidates, key=lambda item: item[0])
    return {
        "tempo_quarter": value,
        "source": "pdf_vector_text",
        "confidence": 1.0,
        "text": span["text"],
        "bbox": span["bbox"],
    }


def _measure_contains_x(measure: dict, x: float, margin: float) -> bool:
    left, _top, width, _height = (float(value) for value in measure["bbox"])
    return left - margin <= x <= left + width + margin


def _group_string_glyphs(glyphs: list[dict], spacing: float) -> list[dict]:
    """Join the characters of one fret token without joining adjacent events."""
    if not glyphs:
        return []
    maximum_gap = max(0.75, spacing * 0.06)
    groups: list[list[dict]] = []
    for glyph in sorted(glyphs, key=lambda item: (float(item["bbox"][0]), float(item["x"]))):
        if not groups:
            groups.append([glyph])
            continue
        previous = groups[-1][-1]
        gap = float(glyph["bbox"][0]) - float(previous["bbox"][2])
        if gap <= maximum_gap:
            groups[-1].append(glyph)
        else:
            groups.append([glyph])

    tokens: list[dict] = []
    for group in groups:
        text = "".join(str(item["char"]) for item in group)
        normalized = text.strip("()")
        if not normalized or any(char not in "0123456789xX" for char in normalized):
            continue
        if normalized.lower() == "x":
            fret: int | str = "X"
        elif normalized.isdigit():
            fret = int(normalized)
            if fret > 36:
                # A value above Guitar Pro's practical fret range almost
                # certainly means two adjacent tokens were accidentally joined.
                continue
        else:
            continue
        left = min(float(item["bbox"][0]) for item in group)
        top = min(float(item["bbox"][1]) for item in group)
        right = max(float(item["bbox"][2]) for item in group)
        bottom = max(float(item["bbox"][3]) for item in group)
        tokens.append(
            {
                "fret": fret,
                "tie_in": text.startswith("(") and text.endswith(")"),
                "text": text,
                "bbox": [left, top, right, bottom],
                "x": (left + right) / 2.0,
                "y": (top + bottom) / 2.0,
                "font": group[0].get("font"),
                "font_size_pt": group[0].get("font_size_pt"),
            }
        )
    return tokens


def _cluster_tokens(tokens: list[dict], spacing: float) -> list[list[dict]]:
    """Group simultaneous fret tokens across strings into one printed event."""
    clusters: list[list[dict]] = []
    radius = max(3.0, spacing * 0.32)
    for token in sorted(tokens, key=lambda item: float(item["x"])):
        if not clusters:
            clusters.append([token])
            continue
        center = median(float(item["x"]) for item in clusters[-1])
        if abs(float(token["x"]) - center) <= radius:
            clusters[-1].append(token)
        else:
            clusters.append([token])
    return clusters


def apply_pdf_vector_tab_glyphs(systems: list[dict], glyphs: list[dict]) -> dict:
    """Map vector PDF glyphs to strings, measures and score-event positions.

    The function deliberately creates an event for a clear printed TAB token
    when the score notehead locator missed it. Rhythm is still inferred later
    from the standard-notation crop, so no duration is guessed here.
    """
    summary = {
        "available_glyphs": len(glyphs),
        "tab_aligned_glyphs": 0,
        "tokens": 0,
        "vector_events": 0,
        "vector_notes": 0,
        "recovered_score_events": 0,
        "rejected_duplicate_string_tokens": 0,
    }
    for system in systems:
        spacing = float(system["tab_spacing"])
        line_radius = max(3.0, spacing * 0.42)
        measure_glyphs: dict[int, dict[int, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for glyph in glyphs:
            string_index, distance = min(
                enumerate(system["tab_string_y"], start=1),
                key=lambda item: abs(float(glyph["y"]) - float(item[1])),
            )
            if abs(float(glyph["y"]) - float(distance)) > line_radius:
                continue
            measure_index = next(
                (
                    index
                    for index, measure in enumerate(system["measures"])
                    if _measure_contains_x(measure, float(glyph["x"]), spacing * 0.20)
                ),
                None,
            )
            if measure_index is None:
                continue
            measure_glyphs[measure_index][string_index].append(glyph)
            summary["tab_aligned_glyphs"] += 1

        for measure_index, measure in enumerate(system["measures"]):
            tokens: list[dict] = []
            for string_index, items in measure_glyphs.get(measure_index, {}).items():
                for token in _group_string_glyphs(items, spacing):
                    token["string"] = int(string_index)
                    tokens.append(token)
            summary["tokens"] += len(tokens)
            clusters = _cluster_tokens(tokens, spacing)

            score_events = sorted(measure.get("events", []), key=lambda item: float(item["x"]))
            used_score_events: set[int] = set()
            vector_events: list[dict] = []
            match_radius = max(7.0, spacing * 0.70)
            for cluster in clusters:
                cluster_x = median(float(item["x"]) for item in cluster)
                candidates = [
                    (abs(float(event["x"]) - cluster_x), index, event)
                    for index, event in enumerate(score_events)
                    if index not in used_score_events
                ]
                if candidates and min(candidates)[0] <= match_radius:
                    _delta, matched_index, score_event = min(candidates)
                    used_score_events.add(matched_index)
                    event_x = float(score_event["x"])
                else:
                    event_x = float(cluster_x)
                    score_event = {
                        "x": event_x,
                        "locator_confidence": 1.0,
                        "locator_source": "pdf_vector_tab_recovery",
                    }
                    measure.setdefault("events", []).append(score_event)
                    score_events.append(score_event)
                    used_score_events.add(len(score_events) - 1)
                    summary["recovered_score_events"] += 1

                notes_by_string: dict[int, dict] = {}
                for token in sorted(cluster, key=lambda item: int(item["string"])):
                    string_index = int(token["string"])
                    if string_index in notes_by_string:
                        summary["rejected_duplicate_string_tokens"] += 1
                        continue
                    notes_by_string[string_index] = {
                        "string": string_index,
                        "fret": token["fret"],
                        "source": "pdf_vector_text",
                        "confidence": 1.0,
                        "tie_in": bool(token["tie_in"]),
                        "printed_text": token["text"],
                        "bbox": token["bbox"],
                    }
                if notes_by_string:
                    vector_events.append(
                        {"x": event_x, "notes": list(notes_by_string.values())}
                    )
                    summary["vector_notes"] += len(notes_by_string)

            measure["events"] = sorted(
                measure.get("events", []), key=lambda item: float(item["x"])
            )
            measure["vector_tab_events"] = sorted(
                vector_events, key=lambda item: float(item["x"])
            )
            summary["vector_events"] += len(vector_events)
    # A populated positioned-text layer is authoritative for printed attacks:
    # official GP8 score+TAB PDFs do not selectively rasterize individual fret
    # numbers. Events without a token are therefore rests or unprinted tie
    # continuations, not opportunities for a raster classifier to invent one.
    authoritative = summary["tokens"] >= 3
    summary["authoritative_for_printed_attacks"] = authoritative
    if authoritative:
        for system in systems:
            system["pdf_vector_tab_authoritative"] = True
            for measure in system.get("measures", []):
                measure["pdf_vector_tab_authoritative"] = True
    return summary


def _set_voice_prediction(voice: dict, task: str, value, source: str) -> None:
    prediction = voice.get(task)
    if not isinstance(prediction, dict):
        return
    prediction["value"] = value
    prediction["confidence"] = 1.0
    prediction["vector_override"] = source


def _vector_attack_xs(measure: dict) -> list[float]:
    return [
        float(vector_event["x"])
        for vector_event in measure.get("vector_tab_events", [])
        if any(not bool(note.get("tie_in", False)) for note in vector_event.get("notes", []))
    ]


def apply_pdf_vector_rhythm_text(systems: list[dict], spans: list[dict]) -> dict:
    """Constrain attack state and tuplet membership from GP8 vector text.

    A tuplet number is a relation over a *group* of events.  Predicting the
    division independently for every crop is therefore structurally wrong and
    commonly produces only one or two triplet-labelled notes.  GP8 writes the
    group number in a dedicated bold Times span.  We assign that marker to the
    closest event interval by its horizontal centre, then apply the division to
    the whole interval and clear impossible isolated tuplet predictions.
    """
    summary = {
        "text_spans": len(spans),
        "gp8_font_signature": _has_gp8_font_signature(spans),
        "attack_state_overrides": 0,
        "tuplet_markers": 0,
        "tuplet_markers_with_brackets": 0,
        "tuplet_groups": 0,
        "tuplet_events": 0,
        "cleared_isolated_tuplets": 0,
        "by_division": {},
    }
    if not systems:
        return summary
    gp8_signature = bool(summary["gp8_font_signature"])

    system_centers: list[tuple[float, dict]] = []
    for system in systems:
        vertical = [
            *[float(value) for value in system.get("score_line_y", [])],
            *[float(value) for value in system.get("tab_string_y", [])],
        ]
        if vertical:
            system_centers.append((sum(vertical) / len(vertical), system))

    markers_by_system: dict[int, list[dict]] = defaultdict(list)
    for span in spans:
        text = str(span["text"]).strip()
        font = str(span.get("font", "")).lower()
        size = float(span.get("font_size_pt", 0.0))
        if not text.isdigit() or not (8.0 <= size <= 10.0):
            continue
        # Official GP8 uses TimesNewRoman-1-50 for tuplet numbers. Measure
        # numbers are non-bold Times at 6 pt and TAB digits are Arial at 7 pt.
        if "timesnewroman-1" not in font:
            continue
        numerator = int(text)
        division = TUPLET_DIVISIONS.get(numerator)
        if division is None or not system_centers:
            continue
        system = min(
            system_centers,
            key=lambda item: abs(float(span["y"]) - item[0]),
        )[1]
        markers_by_system[id(system)].append(
            {**span, "numerator": numerator, "division": division}
        )
        summary["tuplet_markers"] += 1
        summary["tuplet_markers_with_brackets"] += int(
            span.get("tuplet_bracket_bbox") is not None
        )

    division_counts: dict[str, int] = defaultdict(int)
    for system in systems:
        if not system.get("pdf_vector_tab_authoritative"):
            continue
        system["pdf_vector_gp8_signature"] = gp8_signature
        spacing = float(system["tab_spacing"])
        measure_markers: dict[int, list[dict]] = defaultdict(list)
        for marker in markers_by_system.get(id(system), []):
            measure_index = next(
                (
                    index
                    for index, measure in enumerate(system.get("measures", []))
                    if _measure_contains_x(measure, float(marker["x"]), spacing * 0.65)
                ),
                None,
            )
            if measure_index is not None:
                measure_markers[measure_index].append(marker)

        for measure_index, measure in enumerate(system.get("measures", [])):
            measure["pdf_vector_gp8_signature"] = gp8_signature
            attacks = _vector_attack_xs(measure)
            for event in measure.get("events", []):
                if not attacks or min(abs(float(event["x"]) - x) for x in attacks) > 1.0:
                    continue
                primary = next(
                    (voice for voice in event.get("voices", []) if int(voice.get("voice", -1)) == 0),
                    None,
                )
                if primary is not None and primary.get("state", {}).get("value") != "note":
                    _set_voice_prediction(primary, "state", "note", "pdf_vector_tab_attack")
                    primary["visible"] = True
                    summary["attack_state_overrides"] += 1

            if not gp8_signature:
                continue

            active: list[tuple[int, dict, dict]] = []
            for event_index, event in enumerate(measure.get("events", [])):
                primary = next(
                    (voice for voice in event.get("voices", []) if int(voice.get("voice", -1)) == 0),
                    None,
                )
                if primary is None or primary.get("state", {}).get("value") == "empty":
                    continue
                active.append((event_index, event, primary))

            selected: dict[int, str] = {}
            group_records = []
            for marker in sorted(measure_markers.get(measure_index, []), key=lambda item: float(item["x"])):
                if len(active) < 2:
                    continue
                best = None
                expected = int(marker["numerator"])
                bracket = marker.get("tuplet_bracket_bbox")
                if bracket is not None:
                    margin = max(2.0, spacing * 0.12)
                    interval = [
                        item for item in active
                        if float(bracket[0]) - margin <= float(item[1]["x"]) <= float(bracket[2]) + margin
                        and item[0] not in selected
                    ]
                    if len(interval) >= 2:
                        best = (0.0, interval)
                if best is None:
                    maximum = min(7, len(active))
                    for length in range(2, maximum + 1):
                        for start in range(0, len(active) - length + 1):
                            interval = active[start : start + length]
                            if any(item[0] in selected for item in interval):
                                continue
                            left = float(interval[0][1]["x"])
                            right = float(interval[-1][1]["x"])
                            center_error = abs((left + right) / 2.0 - float(marker["x"])) / max(spacing, 1.0)
                            length_penalty = 0.12 * abs(length - expected)
                            candidate_score = center_error + length_penalty
                            if best is None or candidate_score < best[0]:
                                best = (candidate_score, interval)
                if best is None:
                    continue
                _score, interval = best
                division = str(marker["division"])
                for event_index, _event, _primary in interval:
                    selected[event_index] = division
                group_records.append({
                    "text": marker["text"],
                    "division": division,
                    "bbox": marker["bbox"],
                    "bracket_bbox": marker.get("tuplet_bracket_bbox"),
                    "event_indices": [item[0] for item in interval],
                })
                summary["tuplet_groups"] += 1

            for event_index, _event, primary in active:
                predicted = primary.get("division", {}).get("value")
                resolved = selected.get(event_index, "1:1")
                if predicted != resolved:
                    if predicted != "1:1" and resolved == "1:1":
                        summary["cleared_isolated_tuplets"] += 1
                    _set_voice_prediction(primary, "division", resolved, "pdf_vector_tuplet_text")
                if resolved != "1:1":
                    summary["tuplet_events"] += 1
                    division_counts[resolved] += 1
            measure["pdf_vector_tuplet_groups"] = group_records

    summary["by_division"] = dict(sorted(division_counts.items()))
    return summary


def apply_pdf_vector_beam_rhythm(
    systems: list[dict],
    beams: list[dict],
    gp8_font_signature: bool,
) -> dict:
    """Use an unambiguous two-beam stack to confirm sixteenth notes.

    One beam can be either an eighth or a sixteenth whose secondary beamlet is
    outside the event centre, so it is deliberately not used as an override.
    Exactly two beam polygons crossing the event x coordinate is a stable GP8
    sixteenth-note signature.  Multi-voice events are left to the CNN because
    independent voice beams may overlap at the same x.
    """
    summary = {
        "beam_paths": len(beams),
        "gp8_font_signature": bool(gp8_font_signature),
        "two_beam_events": 0,
        "duration_overrides": 0,
    }
    if not gp8_font_signature:
        return summary
    for system in systems:
        if not system.get("pdf_vector_tab_authoritative"):
            continue
        score_lines = [float(value) for value in system.get("score_line_y", [])]
        if len(score_lines) < 2:
            continue
        score_spacing = (score_lines[-1] - score_lines[0]) / (len(score_lines) - 1)
        vertical_top = score_lines[0] - 5.0 * score_spacing
        vertical_bottom = score_lines[-1] + 3.0 * score_spacing
        spacing = float(system.get("tab_spacing") or score_spacing)
        system_beams = [
            beam for beam in beams
            if float(beam["bbox"][3]) >= vertical_top
            and float(beam["bbox"][1]) <= vertical_bottom
        ]
        for measure in system.get("measures", []):
            left, _top, measure_width, _height = (
                float(value) for value in measure["bbox"]
            )
            local_beams = [
                beam for beam in system_beams
                if float(beam["bbox"][2]) >= left
                and float(beam["bbox"][0]) <= left + measure_width
            ]
            attacks = _vector_attack_xs(measure)
            for event in measure.get("events", []):
                event_x = float(event["x"])
                if not attacks or min(abs(event_x - value) for value in attacks) > 1.0:
                    continue
                active = [
                    voice for voice in event.get("voices", [])
                    if voice.get("state", {}).get("value") != "empty"
                ]
                primary = next(
                    (voice for voice in active if int(voice.get("voice", -1)) == 0),
                    None,
                )
                if (
                    primary is None
                    or len(active) != 1
                    or primary.get("state", {}).get("value") != "note"
                ):
                    continue
                connected = [
                    beam for beam in local_beams
                    if float(beam["bbox"][0]) - 4.0 <= event_x
                    <= float(beam["bbox"][2]) + 4.0
                ]
                if len(connected) != 2:
                    continue
                if max(float(beam["width"]) for beam in connected) < 1.2 * spacing:
                    continue
                summary["two_beam_events"] += 1
                if int(primary.get("duration", {}).get("value", 0)) != 16:
                    _set_voice_prediction(
                        primary, "duration", 16, "pdf_vector_two_beam_stack"
                    )
                    summary["duration_overrides"] += 1
    return summary


def apply_pdf_vector_technique_text(systems: list[dict], spans: list[dict]) -> dict:
    """Attach high-precision textual GP8 techniques to preceding attacks.

    Guitar Pro writes slide labels twice (score and TAB) at the same x. Bend
    amount text is placed at the arrow end, so it belongs to the latest printed
    attack to its left, not necessarily to the geometrically nearest event.
    """
    markers = []
    for span in spans:
        normalized = str(span["text"]).strip().lower()
        technique = VECTOR_TECHNIQUE_TEXT.get(normalized)
        if technique is not None:
            markers.append({**span, "technique": technique})
    summary = {
        "text_spans": len(spans),
        "gp8_font_signature": _has_gp8_font_signature(spans),
        "technique_markers": len(markers),
        "deduplicated_markers": 0,
        "attached_events": 0,
        "by_class": {},
    }
    if not summary["gp8_font_signature"] or not markers or not systems:
        return summary

    system_centers = []
    for system in systems:
        vertical = [
            *[float(value) for value in system.get("score_line_y", [])],
            *[float(value) for value in system.get("tab_string_y", [])],
        ]
        if vertical:
            system_centers.append((sum(vertical) / len(vertical), system))

    assigned: dict[int, list[dict]] = defaultdict(list)
    for marker in markers:
        if not system_centers:
            continue
        system = min(
            system_centers,
            key=lambda item: abs(float(marker["y"]) - item[0]),
        )[1]
        assigned[id(system)].append(marker)

    class_counts: dict[str, int] = defaultdict(int)
    for system in systems:
        spacing = float(system["tab_spacing"])
        unique: list[dict] = []
        for marker in sorted(
            assigned.get(id(system), []),
            key=lambda item: (str(item["technique"]), float(item["x"]), float(item["y"])),
        ):
            duplicate = next(
                (
                    item for item in unique
                    if item["technique"] == marker["technique"]
                    and abs(float(item["x"]) - float(marker["x"])) <= max(1.5, spacing * 0.12)
                ),
                None,
            )
            if duplicate is None:
                unique.append(marker)
            else:
                summary["deduplicated_markers"] += 1

        for marker in unique:
            measure = next(
                (
                    item for item in system.get("measures", [])
                    if _measure_contains_x(item, float(marker["x"]), spacing * 0.65)
                ),
                None,
            )
            if measure is None:
                continue
            attacks = [
                event
                for event in measure.get("events", [])
                if any(
                    not bool(note.get("tie_in", False))
                    for vector_event in measure.get("vector_tab_events", [])
                    if abs(float(vector_event["x"]) - float(event["x"])) <= 1.0
                    for note in vector_event.get("notes", [])
                )
            ]
            if not attacks:
                continue
            preceding = [
                event for event in attacks if float(event["x"]) <= float(marker["x"])
            ]
            event = (
                max(preceding, key=lambda item: float(item["x"]))
                if preceding else
                min(attacks, key=lambda item: abs(float(item["x"]) - float(marker["x"])))
            )
            event.setdefault("pdf_vector_techniques", {})[marker["technique"]] = {
                "text": marker["text"],
                "bbox": marker["bbox"],
                "source": "pdf_vector_text",
            }
            class_counts[str(marker["technique"])] += 1

    for system in systems:
        if not system.get("pdf_vector_tab_authoritative"):
            continue
        for measure in system.get("measures", []):
            for event in measure.get("events", []):
                prediction = event.get("technique_prediction")
                if prediction is None:
                    continue
                for name in prediction.get("positive", {}):
                    prediction["positive"][name] = False
                for name in event.get("pdf_vector_techniques", {}):
                    prediction["positive"][name] = True
                prediction["vector_text_override"] = event.get(
                    "pdf_vector_techniques", {}
                )
                summary["attached_events"] += int(bool(event.get("pdf_vector_techniques")))
    summary["by_class"] = dict(sorted(class_counts.items()))
    return summary
