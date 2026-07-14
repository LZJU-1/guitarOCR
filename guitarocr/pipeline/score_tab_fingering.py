from __future__ import annotations

import copy
from collections import Counter

import numpy as np
from PIL import Image
import torch

from guitarocr.data.build_tab_detector_dataset import tile_measure
from guitarocr.models.tab_detector_model import TabSymbolDetector
from guitarocr.pipeline.infer_tuxguitar_tab_page import group_symbols, map_detection_to_page, nms_same_class
from guitarocr.training.train_tab_detector import decode_detections


@torch.inference_mode()
def detect_tab_fingering(
    page: Image.Image,
    systems: list[dict],
    model: TabSymbolDetector,
    classes: list[str],
    device: torch.device,
    threshold: float = 0.3,
    batch_size: int = 16,
) -> None:
    records: list[tuple[Image.Image, dict, dict, dict]] = []
    for system in systems:
        tab_top = system["tab_string_y"][0] - system["tab_spacing"] / 2.0
        tab_height = system["tab_string_y"][-1] - system["tab_string_y"][0] + system["tab_spacing"]
        for measure in system["measures"]:
            measure["tab_symbols"] = []
            measure["tab_events"] = []
            tab_bbox = [measure["bbox"][0], tab_top, measure["bbox"][2], tab_height]
            measure["tab_bbox"] = tab_bbox
            for tile, _, transform in tile_measure(page, tab_bbox, []):
                records.append((tile, transform, system, measure))

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        arrays = [np.asarray(item[0], dtype=np.float32) / 255.0 for item in batch]
        tensor = torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(device)
        tensor = (tensor - 0.5) / 0.5
        decoded = decode_detections(model(tensor), threshold=threshold)
        for detections, (_, transform, _, measure) in zip(decoded, batch):
            measure["tab_symbols"].extend(map_detection_to_page(item, transform) for item in detections)

    for system in systems:
        for measure in system["measures"]:
            measure["tab_symbols"] = nms_same_class(measure["tab_symbols"])
            for symbol in measure["tab_symbols"]:
                symbol["class"] = classes[int(symbol.pop("class_index"))]
            measure["tab_events"] = group_symbols(
                measure["tab_symbols"], system["tab_string_y"], system["tab_spacing"]
            )


def recover_isolated_tab_events(systems: list[dict]) -> dict:
    """Recover a whole-measure event when the score locator misses it.

    Dead/X whole notes can have no conventional filled notehead, while their
    TAB token remains unambiguous.  Only an otherwise empty measure with one
    TAB event is recovered; crowded or ambiguous measures are left untouched.
    The normal rhythm and technique classifiers then inspect the recovered x
    position, so no duration or effect is guessed here.
    """
    summary = {"candidates": 0, "recovered": 0}
    for system in systems:
        for measure in system.get("measures", []):
            if measure.get("events") or len(measure.get("tab_events", [])) != 1:
                continue
            summary["candidates"] += 1
            tab_event = measure["tab_events"][0]
            measure["events"] = [{
                "x": float(tab_event["x"]),
                "locator_confidence": 0.0,
                "locator_source": "isolated_tab_event_recovery",
            }]
            summary["recovered"] += 1
    return summary


def compact_rhythm_voice(voice: dict) -> dict:
    state = voice["state"]["value"]
    return {
        "voice": int(voice["voice"]),
        "state": state,
        "duration_value": int(voice["duration"]["value"]) if state != "empty" else None,
        "dot": voice["dot"]["value"] if state != "empty" else None,
        "division": voice["division"]["value"] if state != "empty" else None,
        "confidence": {
            "state": float(voice["state"]["confidence"]),
            "duration": float(voice["duration"]["confidence"]),
            "dot": float(voice["dot"]["confidence"]),
            "division": float(voice["division"]["confidence"]),
        },
        "candidates": {
            "state": voice["state"]["top"],
            "duration": voice["duration"]["top"],
            "dot": voice["dot"]["top"],
            "division": voice["division"]["top"],
        },
    }


def build_score_ir(systems: list[dict], measure_number_offset: int | None = None) -> dict:
    measures: list[dict] = []
    for system in systems:
        for measure in system["measures"]:
            unused_tab = set(range(len(measure.get("tab_events", []))))
            events: list[dict] = []
            for order, score_event in enumerate(measure["events"]):
                candidates = [
                    (abs(score_event["x"] - measure["tab_events"][index]["x"]), index)
                    for index in unused_tab
                ]
                matched_index: int | None = None
                delta: float | None = None
                if candidates:
                    delta, candidate_index = min(candidates)
                    if delta <= max(6.0, system["tab_spacing"] * 0.60):
                        matched_index = candidate_index
                        unused_tab.remove(candidate_index)
                tab_event = measure["tab_events"][matched_index] if matched_index is not None else None
                voices = [compact_rhythm_voice(voice) for voice in score_event.get("voices", [])]
                visible_note_voices = [voice["voice"] for voice in voices if voice["state"] == "note"]
                notes = [] if tab_event is None else [
                    {
                        "string": int(note["string"]),
                        "fret": note["fret"],
                        "voice": visible_note_voices[0] if len(visible_note_voices) == 1 else None,
                        "source": "printed_tab",
                        "tie_in": False,
                        "tie_out": False,
                        "dead": isinstance(note["fret"], str) and note["fret"].upper() == "X",
                        "effects": {
                            **score_event.get("technique_prediction", {}).get("positive", {}),
                            # A dead/X note has no sustained pitch to modulate;
                            # wavy slide-out marks can otherwise resemble vibrato.
                            "vibrato": bool(
                                score_event.get("technique_prediction", {})
                                .get("positive", {}).get("vibrato", False)
                            ) and not (isinstance(note["fret"], str) and note["fret"].upper() == "X"),
                            "dead": isinstance(note["fret"], str) and note["fret"].upper() == "X",
                        },
                    }
                    for note in tab_event["notes"]
                ]
                if tab_event is not None:
                    match_status = "matched"
                elif any(voice["state"] == "rest" for voice in voices):
                    match_status = "score_only_rest_or_unresolved_voice"
                else:
                    match_status = "score_only_no_visible_tab"
                tie_visual = score_event.get("tie_prediction")
                tie_relation = None
                if tie_visual is not None:
                    missing_note_count = max(0, int(tie_visual["score_note_count"]) - len(notes))
                    pure_tab = system.get("layout") == "tab_only"
                    candidate_count = (
                        int(tie_visual["tie_note_count"])
                        if pure_tab else missing_note_count
                    )
                    consistent = bool(
                        tie_visual["visual_positive"]
                        and candidate_count > 0
                        and (
                            not pure_tab
                            or len(tie_visual.get("target_y_page", [])) == candidate_count
                        )
                    )
                    tie_relation = {
                        **tie_visual,
                        "target_reference": (
                            "tab_string"
                            if system.get("layout") == "tab_only" else "score_notehead"
                        ),
                        "attacked_tab_note_count": len(notes),
                        "missing_score_note_count": missing_note_count,
                        "candidate": consistent,
                        "candidate_tie_count": candidate_count if consistent else 0,
                        "status": (
                            "tab_visual_target_candidate"
                            if consistent and pure_tab else
                            "score_tab_consistent_candidate"
                            if consistent else
                            "visual_rejected_no_missing_score_note"
                            if tie_visual["visual_positive"] else
                            "no_visual_tie"
                        ),
                    }
                events.append(
                    {
                        "order": order,
                        "x": float(score_event["x"]),
                        "locator_confidence": float(score_event["locator_confidence"]),
                        "tab_match": match_status,
                        "tab_x_delta": delta if matched_index is not None else None,
                        "voices": voices,
                        "notes": notes,
                        "technique_prediction": score_event.get("technique_prediction"),
                        "tie_relation": tie_relation,
                    }
                )
            orphan_tab = [measure["tab_events"][index] for index in sorted(unused_tab)]
            measures.append(
                {
                    "number": (
                        measure_number_offset + int(measure["measure_number"]) - 1
                        if measure_number_offset is not None else None
                    ),
                    "page_measure_index": int(measure["measure_number"]),
                    "system_index": int(system["system_index"]),
                    "bbox": [float(value) for value in measure["bbox"]],
                    "tab_string_y": [float(value) for value in system["tab_string_y"]],
                    "tab_spacing": float(system["tab_spacing"]),
                    "time_signature": measure.get("time_signature"),
                    "time_signature_source": measure.get("time_signature_source", "unknown"),
                    "printed_time_signature": measure.get("printed_time_signature"),
                    "printed_time_signature_candidate": measure.get(
                        "printed_time_signature_candidate"
                    ),
                    "printed_time_signature_shape_hint": measure.get(
                        "printed_time_signature_shape_hint"
                    ),
                    "events": events,
                    "orphan_tab_events": orphan_tab,
                }
            )
    return {
        "schema_version": "1.0",
        "tracks": [
            {
                "track": 1,
                "string_count": len(systems[0]["tab_string_y"]) if systems else None,
                "string_tuning_midi": None,
                "capo": None,
                "tempo_quarter": None,
                "measures": measures,
            }
        ],
        "scope": (
            "Pixel-only event/fingering/rhythm intermediate representation. "
            "Measure number is null unless an external page offset is supplied. Printed time signatures can be "
            "recognized and propagated before this IR is built. Exact beat positions, ties, tuning, tempo and "
            "ambiguous multi-voice note assignment remain unresolved."
        ),
    }


def resolve_unambiguous_ties(score_ir: dict) -> dict:
    """Resolve conservative continuations, using TAB-string y when available."""
    edges: list[dict] = []
    summary = {
        "visual_candidates": 0,
        "score_tab_consistent_candidates": 0,
        "auto_resolved_events": 0,
        "auto_resolved_notes": 0,
        "unresolved_candidates": 0,
    }
    for track in score_ir["tracks"]:
        previous: tuple[dict, dict] | None = None
        last_note_by_string: dict[int, tuple[dict, dict, int, dict]] = {}

        def remember_notes(measure: dict, event: dict) -> None:
            for note_index, note in enumerate(event["notes"]):
                if not note.get("dead"):
                    last_note_by_string[int(note["string"])] = (
                        measure, event, note_index, note
                    )

        for measure in track["measures"]:
            for event in measure["events"]:
                relation = event.get("tie_relation")
                if relation is not None:
                    summary["visual_candidates"] += int(relation["visual_positive"])
                    summary["score_tab_consistent_candidates"] += int(relation["candidate"])
                if relation is None or not relation["candidate"]:
                    remember_notes(measure, event)
                    previous = (measure, event)
                    continue
                missing = int(relation["candidate_tie_count"])
                attacked_strings = {int(note["string"]) for note in event["notes"]}
                continuation_candidates: list[tuple[dict, dict, int, dict]] = []
                resolution_method = "adjacent_score_tab_consistency"
                if (
                    relation.get("target_reference") == "tab_string"
                    and relation.get("target_y_page")
                    and measure.get("tab_string_y")
                ):
                    string_y = [float(value) for value in measure["tab_string_y"]]
                    spacing = (
                        (string_y[-1] - string_y[0]) / max(1, len(string_y) - 1)
                    )
                    target_strings = {
                        min(
                            range(len(string_y)),
                            key=lambda index: abs(string_y[index] - float(target_y)),
                        ) + 1
                        for target_y in relation["target_y_page"]
                        if min(abs(value - float(target_y)) for value in string_y)
                        <= 0.6 * spacing
                    }
                    continuation_candidates = [
                        last_note_by_string[string]
                        for string in sorted(target_strings)
                        if string not in attacked_strings and string in last_note_by_string
                    ]
                    resolution_method = "tab_string_y_sequence_consistency"
                if previous is not None:
                    adjacent_candidates = [
                        (previous[0], previous[1], note_index, note)
                        for note_index, note in enumerate(previous[1]["notes"])
                        if int(note["string"]) not in attacked_strings
                    ]
                    if len(continuation_candidates) != missing:
                        continuation_candidates = adjacent_candidates
                        resolution_method = "adjacent_score_tab_consistency"
                can_resolve = missing > 0 and len(continuation_candidates) == missing
                if not can_resolve:
                    relation["status"] = "unresolved_partial_or_nonadjacent"
                    summary["unresolved_candidates"] += 1
                    remember_notes(measure, event)
                    previous = (measure, event)
                    continue
                continued_notes = []
                for source_measure, source_event, note_index, source_note in continuation_candidates:
                    source_note["tie_out"] = True
                    continued = copy.deepcopy(source_note)
                    continued["source"] = "tie_continuation"
                    continued["tie_in"] = True
                    continued["tie_out"] = False
                    current_effects = (
                        (event.get("technique_prediction") or {}).get("positive", {})
                    )
                    continued["effects"] = {
                        **current_effects,
                        "vibrato": bool(current_effects.get("vibrato", False))
                        and not bool(source_note.get("dead")),
                        "dead": bool(source_note.get("dead")),
                    }
                    continued_notes.append(continued)
                    edges.append(
                        {
                            "from": {
                                "measure": source_measure["number"],
                                "page_measure_index": source_measure["page_measure_index"],
                                "event_order": source_event["order"],
                                "note_index": note_index,
                            },
                            "to": {
                                "measure": measure["number"],
                                "page_measure_index": measure["page_measure_index"],
                                "event_order": event["order"],
                                "note_index": len(event["notes"]) + len(continued_notes) - 1,
                            },
                            "string": source_note["string"],
                            "fret": source_note["fret"],
                            "method": resolution_method,
                        }
                    )
                event["notes"].extend(continued_notes)
                event["notes"].sort(key=lambda note: int(note["string"]))
                relation["status"] = f"auto_resolved_{resolution_method}"
                summary["auto_resolved_events"] += 1
                summary["auto_resolved_notes"] += len(continued_notes)
                remember_notes(measure, event)
                previous = (measure, event)
    score_ir["tie_edges"] = edges
    score_ir["tie_summary"] = summary
    return summary


def correct_multidigit_fret_outliers(score_ir: dict, window: int = 3) -> dict:
    """Repair rare digit-join errors using same-string musical continuity.

    A detector can merge a printed fret 2 with a nearby false-positive 3 into
    23. We only remove the trailing digit when both neighbouring evidence and
    the printed prefix strongly agree; genuine playing around frets 20--36 is
    retained because its neighbours are high as well.
    """
    summary = {"candidates": 0, "corrected": 0}
    for track in score_ir.get("tracks", []):
        timeline: list[dict] = []
        for measure in track.get("measures", []):
            for event in measure.get("events", []):
                for note in event.get("notes", []):
                    if isinstance(note.get("fret"), int):
                        timeline.append(note)
        by_string: dict[int, list[dict]] = {}
        for note in timeline:
            by_string.setdefault(int(note["string"]), []).append(note)
        for notes in by_string.values():
            original = [int(note["fret"]) for note in notes]
            support = Counter(original)
            for index, fret in enumerate(original):
                if not 20 <= fret <= 39:
                    continue
                summary["candidates"] += 1
                prefix = fret // 10
                neighbours = original[max(0, index - window):index] + original[index + 1:index + window + 1]
                low = [value for value in neighbours if 0 <= value <= 19]
                near_prefix = sum(abs(value - prefix) <= 2 for value in low)
                near_original = sum(abs(value - fret) <= 4 for value in neighbours)
                global_prefix_support = support[prefix] >= max(6, support[fret] * 3)
                local_support = near_prefix >= 2
                if (near_original and not global_prefix_support) or not (local_support or global_prefix_support):
                    continue
                notes[index]["fret_context_correction"] = {
                    "from": fret,
                    "to": prefix,
                    "reason": "isolated_high_fret_digit_join",
                    "low_neighbours": low,
                    "global_prefix_support": support[prefix],
                    "global_joined_support": support[fret],
                }
                notes[index]["fret"] = prefix
                summary["corrected"] += 1
    score_ir["fret_context_corrections"] = summary
    return summary
