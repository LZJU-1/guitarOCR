from __future__ import annotations

import copy

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
                    consistent = bool(tie_visual["visual_positive"] and missing_note_count > 0)
                    tie_relation = {
                        **tie_visual,
                        "attacked_tab_note_count": len(notes),
                        "missing_score_note_count": missing_note_count,
                        "candidate": consistent,
                        "candidate_tie_count": missing_note_count if consistent else 0,
                        "status": (
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
                    "time_signature": measure.get("time_signature"),
                    "time_signature_source": measure.get("time_signature_source", "unknown"),
                    "printed_time_signature": measure.get("printed_time_signature"),
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
    """Resolve only adjacent full-event continuations; retain every other candidate."""
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
        for measure in track["measures"]:
            for event in measure["events"]:
                relation = event.get("tie_relation")
                if relation is not None:
                    summary["visual_candidates"] += int(relation["visual_positive"])
                    summary["score_tab_consistent_candidates"] += int(relation["candidate"])
                if relation is None or not relation["candidate"]:
                    previous = (measure, event)
                    continue
                missing = int(relation["candidate_tie_count"])
                can_resolve = (
                    previous is not None
                    and not event["notes"]
                    and len(previous[1]["notes"]) == missing
                    and missing > 0
                )
                if not can_resolve:
                    relation["status"] = "unresolved_partial_or_nonadjacent"
                    summary["unresolved_candidates"] += 1
                    previous = (measure, event)
                    continue
                previous_measure, previous_event = previous
                continued_notes = []
                for note_index, source_note in enumerate(previous_event["notes"]):
                    source_note["tie_out"] = True
                    continued = copy.deepcopy(source_note)
                    continued["source"] = "tie_continuation"
                    continued["tie_in"] = True
                    continued["tie_out"] = False
                    continued_notes.append(continued)
                    edges.append(
                        {
                            "from": {
                                "measure": previous_measure["number"],
                                "page_measure_index": previous_measure["page_measure_index"],
                                "event_order": previous_event["order"],
                                "note_index": note_index,
                            },
                            "to": {
                                "measure": measure["number"],
                                "page_measure_index": measure["page_measure_index"],
                                "event_order": event["order"],
                                "note_index": note_index,
                            },
                            "string": source_note["string"],
                            "fret": source_note["fret"],
                            "method": "adjacent_full_event_score_tab_consistency",
                        }
                    )
                event["notes"] = continued_notes
                relation["status"] = "auto_resolved_adjacent_full_event"
                summary["auto_resolved_events"] += 1
                summary["auto_resolved_notes"] += len(continued_notes)
                previous = (measure, event)
    score_ir["tie_edges"] = edges
    score_ir["tie_summary"] = summary
    return summary
