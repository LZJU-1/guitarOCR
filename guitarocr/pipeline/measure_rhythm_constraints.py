from __future__ import annotations

import math
from fractions import Fraction

from guitarocr.pipeline.time_signature_recognizer import SUPPORTED_SIGNATURES


PROPOSAL_RELATIVE_PROBABILITY_THRESHOLD = 0.20


SIGNATURE_PREFERENCE_BY_CAPACITY = {
    Fraction(1, 4): (1, 4),
    Fraction(1, 2): (2, 4),
    Fraction(5, 8): (5, 8),
    Fraction(3, 4): (3, 4),
    Fraction(1, 1): (4, 4),
    Fraction(5, 4): (5, 4),
    Fraction(3, 2): (6, 4),
    Fraction(2, 1): (8, 4),
    Fraction(9, 4): (9, 4),
}


def duration_fraction(duration_value: int, dot: str, division: str) -> Fraction:
    value = Fraction(1, int(duration_value))
    if dot == "single":
        value *= Fraction(3, 2)
    elif dot == "double":
        value *= Fraction(7, 4)
    enters, times = (int(part) for part in division.split(":", 1))
    value *= Fraction(times, enters)
    return value


def primary_voice_total(measure: dict) -> Fraction:
    total = Fraction(0, 1)
    for event in measure.get("events", []):
        voice = next(
            (item for item in event.get("voices", []) if int(item.get("voice", -1)) == 0),
            None,
        )
        if voice is None or voice.get("state") == "empty":
            continue
        total += duration_fraction(
            int(voice["duration_value"]), str(voice["dot"]), str(voice["division"])
        )
    return total


def capacity_signature(
    total: Fraction,
    current: tuple[int, int] | None,
    numerator_digit_count: int | None = None,
) -> tuple[int, int] | None:
    candidates = sorted(
        signature for signature in SUPPORTED_SIGNATURES
        if Fraction(signature[0], signature[1]) == total
    )
    if not candidates:
        return None
    if current in candidates:
        return current
    digit_count_matches = [
        signature for signature in candidates
        if len(str(signature[0])) == numerator_digit_count
    ]
    if numerator_digit_count is not None and len(digit_count_matches) == 1:
        return digit_count_matches[0]
    preferred = SIGNATURE_PREFERENCE_BY_CAPACITY.get(total)
    if preferred in candidates:
        return preferred
    return candidates[0]


def refine_time_signatures_from_rhythm(score_ir: dict) -> dict:
    """Cross-check pure-TAB visual meters against predicted measure capacity.

    TuxGuitar writes a complete rhythmic voice, including rests, in every
    measure.  We therefore trust an exact supported capacity at a visually
    printed signature, and across a run of at least two measures.  Isolated
    disagreements without a printed signature stay unchanged so one bad
    rhythm class cannot invent a meter change.
    """
    summary = {
        "measures": 0,
        "capacity_supported": 0,
        "visually_confirmed_changes": 0,
        "sequence_confirmed_changes": 0,
        "unchanged": 0,
    }
    for track in score_ir.get("tracks", []):
        measures = track.get("measures", [])
        totals = [primary_voice_total(measure) for measure in measures]
        for index, (measure, total) in enumerate(zip(measures, totals)):
            summary["measures"] += 1
            raw_current = measure.get("time_signature")
            current = (
                (int(raw_current[0]), int(raw_current[1]))
                if raw_current and len(raw_current) == 2 else None
            )
            shape_hint = measure.get("printed_time_signature_shape_hint") or {}
            numerator_digit_count = shape_hint.get("numerator_digit_count")
            inferred = capacity_signature(total, current, numerator_digit_count)
            if inferred is None:
                summary["unchanged"] += 1
                continue
            summary["capacity_supported"] += 1
            if inferred == current:
                summary["unchanged"] += 1
                continue
            printed = measure.get("printed_time_signature")
            visual_candidate = (
                measure.get("printed_time_signature_candidate") or printed
            )
            candidate_confidence = (
                float(visual_candidate.get("confidence", 0.0))
                if visual_candidate is not None else 0.0
            )
            candidate_bbox = (
                visual_candidate.get("bbox") if visual_candidate is not None else None
            )
            measure_bbox = measure.get("bbox")
            spacing = float(measure.get("tab_spacing") or 0.0)
            candidate_shape = bool(
                candidate_bbox
                and measure_bbox
                and spacing > 0.0
                and float(candidate_bbox[2]) >= 0.75 * spacing
                and 0.0 <= float(visual_candidate["x"]) - float(measure_bbox[0]) <= 4.0 * spacing
            )
            neighbouring_support = any(
                0 <= neighbour < len(totals) and totals[neighbour] == total
                for neighbour in (index - 1, index + 1)
            )
            # A high-confidence printed reading wins over one isolated rhythm
            # disagreement. Low-confidence stacked ink still proves that a
            # meter was printed, while the rhythmic capacity disambiguates its
            # digits (notably TuxGuitar's TAB-only 2 versus 9 shape).
            partial_shape_support = bool(
                numerator_digit_count is not None
                and float(shape_hint.get("confidence", 0.0)) >= 0.50
                and len([
                    signature for signature in SUPPORTED_SIGNATURES
                    if Fraction(signature[0], signature[1]) == total
                ]) > 1
                and len([
                    signature for signature in SUPPORTED_SIGNATURES
                    if Fraction(signature[0], signature[1]) == total
                    and len(str(signature[0])) == numerator_digit_count
                ]) == 1
            )
            visually_confirmed = (
                candidate_shape and (candidate_confidence < 0.40 or neighbouring_support)
            ) or partial_shape_support
            if not visually_confirmed and not neighbouring_support:
                summary["unchanged"] += 1
                continue
            measure["time_signature_visual"] = list(current) if current is not None else None
            measure["time_signature"] = list(inferred)
            measure["time_signature_source"] = (
                "printed_rhythm_capacity" if visually_confirmed else "rhythm_capacity_sequence"
            )
            measure["time_signature_capacity_evidence"] = fraction_json(total)
            key = "visually_confirmed_changes" if visually_confirmed else "sequence_confirmed_changes"
            summary[key] += 1
    score_ir["time_signature_refinement"] = summary
    return summary


def fraction_json(value: Fraction) -> dict:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "text": f"{value.numerator}/{value.denominator}",
        "whole_notes": float(value),
    }


def top_values(items: list[dict], limit: int, minimum: float = 0.005) -> list[tuple[object, float]]:
    result = []
    for item in items[:limit]:
        probability = float(item["probability"])
        if probability >= minimum or not result:
            result.append((item["value"], probability))
    return result


def event_options(event: dict, voice: dict) -> list[dict]:
    duration_values = top_values(voice["candidates"]["duration"], 3)
    dot_values = top_values(voice["candidates"]["dot"], 2)
    if voice.get("vector_overrides", {}).get("division"):
        division_values = [(voice["division"], 1.0)]
    else:
        division_values = top_values(voice["candidates"]["division"], 3)
    options_by_fraction: dict[Fraction, dict] = {}
    for (duration, duration_probability) in duration_values:
        for dot, dot_probability in dot_values:
            for division, division_probability in division_values:
                fraction = duration_fraction(int(duration), str(dot), str(division))
                probability = max(1e-12, duration_probability * dot_probability * division_probability)
                option = {
                    "fraction": fraction,
                    "duration_value": int(duration),
                    "dot": str(dot),
                    "division": str(division),
                    "probability": probability,
                    "remove_event": False,
                }
                existing = options_by_fraction.get(fraction)
                if existing is None or probability > existing["probability"]:
                    options_by_fraction[fraction] = option
    current_fraction = duration_fraction(int(voice["duration_value"]), voice["dot"], voice["division"])
    current_probability = max(
        1e-12,
        float(voice["confidence"]["duration"])
        * float(voice["confidence"]["dot"])
        * float(voice["confidence"]["division"]),
    )
    options_by_fraction[current_fraction] = {
        "fraction": current_fraction,
        "duration_value": int(voice["duration_value"]),
        "dot": voice["dot"],
        "division": voice["division"],
        "probability": current_probability,
        "remove_event": False,
    }
    if (
        (
            float(event["locator_confidence"]) < 0.60
            or (
                event.get("pdf_vector_gp8_signature")
                and float(event["locator_confidence"]) < 0.75
            )
        )
        and event.get("tab_match") != "matched"
    ):
        removal_probability = max(0.01, 1.0 - float(event["locator_confidence"]))
        options_by_fraction[Fraction(0, 1)] = {
            "fraction": Fraction(0, 1),
            "duration_value": None,
            "dot": None,
            "division": None,
            "probability": removal_probability,
            "remove_event": True,
        }
    return list(options_by_fraction.values())


def correction_proposal(events: list[tuple[int, dict, dict]], target: Fraction) -> dict | None:
    if not events:
        return None
    states: dict[Fraction, tuple[float, list[dict]]] = {Fraction(0, 1): (0.0, [])}
    current_cost = 0.0
    for event_index, event, voice in events:
        options = event_options(event, voice)
        current_probability = max(
            1e-12,
            float(voice["confidence"]["duration"])
            * float(voice["confidence"]["dot"])
            * float(voice["confidence"]["division"]),
        )
        current_cost += -math.log(current_probability)
        next_states: dict[Fraction, tuple[float, list[dict]]] = {}
        for total, (cost, choices) in states.items():
            for option in options:
                updated_total = total + option["fraction"]
                if updated_total > target * 2:
                    continue
                updated_cost = cost - math.log(max(1e-12, option["probability"]))
                existing = next_states.get(updated_total)
                if existing is None or updated_cost < existing[0]:
                    next_states[updated_total] = (updated_cost, choices + [{"event_index": event_index, **option}])
        if len(next_states) > 2500:
            ranked = sorted(
                next_states.items(),
                key=lambda item: (abs(item[0] - target), item[1][0]),
            )[:2500]
            next_states = dict(ranked)
        states = next_states
    if target not in states:
        return None
    proposal_cost, choices = states[target]
    modifications: list[dict] = []
    for choice, (_, event, voice) in zip(choices, events):
        changed = (
            choice["remove_event"]
            or int(choice["duration_value"]) != int(voice["duration_value"])
            or choice["dot"] != voice["dot"]
            or choice["division"] != voice["division"]
        )
        if changed:
            modifications.append(
                {
                    "event_index": choice["event_index"],
                    "from": {
                        "duration_value": voice["duration_value"],
                        "dot": voice["dot"],
                        "division": voice["division"],
                    },
                    "to": None if choice["remove_event"] else {
                        "duration_value": choice["duration_value"],
                        "dot": choice["dot"],
                        "division": choice["division"],
                    },
                    "action": "remove_low_confidence_event" if choice["remove_event"] else "change_rhythm",
                    "candidate_probability": choice["probability"],
                }
            )
    if not modifications:
        return None
    relative_probability = math.exp(max(-60.0, min(60.0, current_cost - proposal_cost)))
    return {
        "fills_measure": True,
        "modifications": modifications,
        "relative_probability_vs_current": relative_probability,
        "plausible": relative_probability >= PROPOSAL_RELATIVE_PROBABILITY_THRESHOLD,
    }


def audit_measure(measure: dict) -> dict:
    signature = measure.get("time_signature")
    if not signature:
        return {"status": "unknown_time_signature", "capacity": None, "voices": {}}
    target = Fraction(int(signature[0]), int(signature[1]))
    voice_reports: dict[str, dict] = {}
    for voice_index in range(2):
        active: list[tuple[int, dict, dict]] = []
        cursor = Fraction(0, 1)
        for event_index, event in enumerate(measure["events"]):
            if voice_index >= len(event["voices"]):
                continue
            voice = event["voices"][voice_index]
            if voice["state"] != "empty":
                duration = duration_fraction(
                    int(voice["duration_value"]), voice["dot"], voice["division"]
                )
                voice["onset"] = fraction_json(cursor)
                voice["duration_fraction"] = fraction_json(duration)
                voice["end"] = fraction_json(cursor + duration)
                active.append((event_index, event, voice))
                cursor += duration
            else:
                voice["onset"] = None
                voice["duration_fraction"] = None
                voice["end"] = None
        if not active:
            voice_reports[f"voice_{voice_index}"] = {
                "status": "inactive", "total": fraction_json(Fraction(0, 1)), "delta": None,
                "correction_proposal": None,
            }
            continue
        total = cursor
        delta = target - total
        status = "exact" if delta == 0 else "underfilled" if delta > 0 else "overfilled"
        voice_reports[f"voice_{voice_index}"] = {
            "status": status,
            "total": fraction_json(total),
            "delta": fraction_json(delta),
            "correction_proposal": None if delta == 0 else correction_proposal(active, target),
        }
    return {
        "status": "audited",
        "capacity": fraction_json(target),
        "voices": voice_reports,
    }


def audit_score_ir(score_ir: dict) -> dict:
    counts = {
        "measures": 0,
        "unknown_time_signature": 0,
        "primary_exact": 0,
        "primary_underfilled": 0,
        "primary_overfilled": 0,
        "primary_with_proposal": 0,
        "primary_with_plausible_proposal": 0,
    }
    for track in score_ir["tracks"]:
        for measure in track["measures"]:
            audit = audit_measure(measure)
            measure["rhythm_audit"] = audit
            counts["measures"] += 1
            if audit["status"] == "unknown_time_signature":
                counts["unknown_time_signature"] += 1
                continue
            primary = audit["voices"].get("voice_0", {})
            status = primary.get("status")
            if status in {"exact", "underfilled", "overfilled"}:
                counts[f"primary_{status}"] += 1
            proposal = primary.get("correction_proposal")
            if proposal is not None:
                counts["primary_with_proposal"] += 1
                counts["primary_with_plausible_proposal"] += int(proposal["plausible"])
    score_ir["rhythm_audit_summary"] = counts
    return counts


def apply_plausible_rhythm_corrections(score_ir: dict) -> dict:
    """Apply only measure-filling alternatives supported by CNN probabilities.

    The optimiser never invents a pitch. It changes a duration/dot/tuplet class,
    or removes a low-confidence score-only event, only when the full measure is
    exactly filled and the joint candidate probability is at least 20% of the
    unconstrained prediction. A single non-destructive change may also close an
    otherwise invalid measure when its CNN probability is at least 0.5%; this
    narrow override is recorded as ``unique_measure_closure``.
    """
    summary = {"measures_changed": 0, "events_changed": 0, "events_removed": 0}
    for track in score_ir.get("tracks", []):
        for measure in track.get("measures", []):
            primary = (
                measure.get("rhythm_audit", {}).get("voices", {}).get("voice_0", {})
            )
            proposal = primary.get("correction_proposal")
            if not proposal:
                continue
            modifications = proposal.get("modifications", [])
            closure_override = (
                primary.get("status") in {"underfilled", "overfilled"}
                and proposal.get("fills_measure")
                and len(modifications) == 1
                and modifications[0].get("to") is not None
                and float(modifications[0].get("candidate_probability", 0.0)) >= 0.005
            )
            vector_duplicate_override = False
            if (
                measure.get("pdf_vector_gp8_signature")
                and primary.get("status") in {"underfilled", "overfilled"}
                and proposal.get("fills_measure")
                and len(modifications) == 2
            ):
                removals = [item for item in modifications if item.get("to") is None]
                changes = [item for item in modifications if item.get("to") is not None]
                if len(removals) == 1 and len(changes) == 1:
                    removal_event = measure["events"][int(removals[0]["event_index"])]
                    vector_duplicate_override = bool(
                        removal_event.get("tab_match") != "matched"
                        and float(removal_event.get("locator_confidence", 1.0)) < 0.40
                        and float(changes[0].get("candidate_probability", 0.0)) >= 0.05
                    )
            vector_two_change_closure = bool(
                measure.get("pdf_vector_gp8_signature")
                and primary.get("status") in {"underfilled", "overfilled"}
                and proposal.get("fills_measure")
                and len(modifications) == 2
                and all(item.get("to") is not None for item in modifications)
                and all(
                    float(item.get("candidate_probability", 0.0)) >= 0.02
                    for item in modifications
                )
                and float(proposal.get("relative_probability_vs_current", 0.0)) >= 0.01
            )
            if (
                not proposal.get("plausible")
                and not closure_override
                and not vector_duplicate_override
                and not vector_two_change_closure
            ):
                continue
            changed = False
            for modification in modifications:
                event_index = int(modification["event_index"])
                if not 0 <= event_index < len(measure.get("events", [])):
                    continue
                event = measure["events"][event_index]
                voice = next(
                    (item for item in event.get("voices", []) if int(item.get("voice", -1)) == 0),
                    None,
                )
                if voice is None:
                    continue
                target = modification.get("to")
                if target is None:
                    voice.update({
                        "state": "empty", "duration_value": None,
                        "dot": None, "division": None,
                    })
                    if all(
                        item.get("state") == "empty"
                        for item in event.get("voices", [])
                    ):
                        event["notes"] = []
                        event["suppressed_as_false_event"] = True
                        if event.get("tie_relation") is not None:
                            event["tie_relation"]["candidate"] = False
                            event["tie_relation"]["candidate_tie_count"] = 0
                            event["tie_relation"]["status"] = (
                                "suppressed_by_measure_rhythm_constraint"
                            )
                    summary["events_removed"] += 1
                else:
                    voice.update({
                        "duration_value": int(target["duration_value"]),
                        "dot": target["dot"],
                        "division": target["division"],
                    })
                event["rhythm_constraint_correction"] = {
                    **modification,
                    "relative_probability_vs_current": proposal["relative_probability_vs_current"],
                    "selection": (
                        "plausible" if proposal.get("plausible") else
                        "vector_duplicate_measure_closure" if vector_duplicate_override else
                        "vector_two_change_measure_closure" if vector_two_change_closure else
                        "unique_measure_closure"
                    ),
                }
                summary["events_changed"] += 1
                changed = True
            summary["measures_changed"] += int(changed)
    audit_score_ir(score_ir)
    score_ir["rhythm_constraint_corrections"] = summary
    return summary


def fill_single_event_vector_measures(score_ir: dict) -> dict:
    """Fill a measure represented by one authoritative printed attack."""
    mapping = {
        Fraction(1, 1): (1, "none"),
        Fraction(3, 4): (2, "single"),
        Fraction(1, 2): (2, "none"),
        Fraction(3, 8): (4, "single"),
        Fraction(1, 4): (4, "none"),
        Fraction(3, 16): (8, "single"),
        Fraction(1, 8): (8, "none"),
    }
    summary = {"measures_changed": 0, "events_changed": 0}
    for track in score_ir.get("tracks", []):
        for measure in track.get("measures", []):
            if not measure.get("pdf_vector_gp8_signature"):
                continue
            signature = measure.get("time_signature")
            if not signature:
                continue
            active = []
            for event in measure.get("events", []):
                voice = next(
                    (item for item in event.get("voices", []) if int(item.get("voice", -1)) == 0),
                    None,
                )
                if voice is not None and voice.get("state") != "empty":
                    active.append((event, voice))
            if len(active) != 1:
                continue
            event, voice = active[0]
            if event.get("tab_match") != "matched" or voice.get("state") != "note":
                continue
            capacity = Fraction(int(signature[0]), int(signature[1]))
            resolved = mapping.get(capacity)
            if resolved is None:
                continue
            duration_value, dot = resolved
            current = duration_fraction(
                int(voice["duration_value"]), str(voice["dot"]), str(voice["division"])
            )
            if current >= capacity:
                continue
            voice.update({
                "duration_value": duration_value,
                "dot": dot,
                "division": "1:1",
            })
            event["rhythm_constraint_correction"] = {
                "from_fraction": fraction_json(current),
                "to_fraction": fraction_json(capacity),
                "selection": "single_authoritative_attack_measure_fill",
            }
            summary["measures_changed"] += 1
            summary["events_changed"] += 1
    audit_score_ir(score_ir)
    score_ir["single_event_vector_measure_fill"] = summary
    return summary


def _remap_tie_edges_after_pruning(
    score_ir: dict,
    order_maps: dict[tuple[int | None, int | None], dict[int, int]],
) -> None:
    """Keep relation endpoints aligned when post-tie pruning renumbers events."""
    if "tie_edges" not in score_ir:
        return
    retained_edges = []
    for edge in score_ir.get("tie_edges", []):
        updated = dict(edge)
        valid = True
        for endpoint_name in ("from", "to"):
            endpoint = dict(edge[endpoint_name])
            key = (endpoint.get("measure"), endpoint.get("page_measure_index"))
            mapping = order_maps.get(key)
            old_order = int(endpoint["event_order"])
            if mapping is not None:
                if old_order not in mapping:
                    valid = False
                    break
                endpoint["event_order"] = mapping[old_order]
            updated[endpoint_name] = endpoint
        if valid:
            retained_edges.append(updated)
    score_ir["tie_edges"] = retained_edges


def prune_empty_ir_events(score_ir: dict) -> dict:
    """Remove locator proposals that carry neither rhythm nor pitch."""
    summary = {"removed": 0}
    order_maps = {}
    for track in score_ir.get("tracks", []):
        for measure in track.get("measures", []):
            retained = []
            order_map = {}
            for event in measure.get("events", []):
                old_order = int(event.get("order", len(retained)))
                empty = all(
                    voice.get("state") == "empty"
                    for voice in event.get("voices", [])
                )
                if empty and not event.get("notes"):
                    summary["removed"] += 1
                    continue
                order_map[old_order] = len(retained)
                retained.append(event)
            for order, event in enumerate(retained):
                event["order"] = order
            measure["events"] = retained
            order_maps[(measure.get("number"), measure.get("page_measure_index"))] = order_map
    _remap_tie_edges_after_pruning(score_ir, order_maps)
    score_ir["empty_event_pruning"] = summary
    return summary


def prune_nearby_gp8_duplicate_score_events(score_ir: dict) -> dict:
    """Remove split score-head proposals immediately beside a vector attack.

    GP8 may draw chord noteheads on opposite sides of a stem.  The score
    locator can interpret the left-hand head and the TAB-aligned chord as two
    consecutive events.  After tie resolution we reject the left proposal
    only when it is within one TAB-line spacing of an authoritative attack and
    either conflicts on the same string or lacks strong tie evidence.  A
    high-confidence continuation on disjoint strings is retained.
    """
    summary = {"removed": 0, "measures_changed": 0}
    order_maps = {}
    for track in score_ir.get("tracks", []):
        for measure in track.get("measures", []):
            events = measure.get("events", [])
            if not (
                measure.get("pdf_vector_gp8_signature")
                and measure.get("pdf_vector_tab_authoritative")
                and events
            ):
                continue
            spacing = float(measure.get("tab_spacing") or 0.0)
            retained = []
            order_map = {}
            changed = False
            for index, event in enumerate(events):
                old_order = int(event.get("order", index))
                next_event = events[index + 1] if index + 1 < len(events) else None
                notes = event.get("notes", [])
                note_sources = {note.get("source") for note in notes}
                continuation_only = bool(
                    not notes or note_sources <= {"tie_continuation"}
                )
                close_to_attack = bool(
                    spacing > 0.0
                    and next_event is not None
                    and next_event.get("tab_match") == "matched"
                    and 0.0 < float(next_event["x"]) - float(event["x"]) <= 0.90 * spacing
                )
                relation = event.get("tie_relation") or {}
                candidate_strings = {int(note["string"]) for note in notes}
                next_strings = {
                    int(note["string"]) for note in (next_event or {}).get("notes", [])
                }
                string_conflict = bool(candidate_strings & next_strings)
                weak_tie = float(relation.get("visual_probability", 0.0)) < 0.50
                duplicate = bool(
                    event.get("tab_match") != "matched"
                    and continuation_only
                    and close_to_attack
                    and (not notes or string_conflict or weak_tie)
                )
                if duplicate:
                    summary["removed"] += 1
                    changed = True
                    continue
                order_map[old_order] = len(retained)
                retained.append(event)
            if changed:
                summary["measures_changed"] += 1
            for order, event in enumerate(retained):
                event["order"] = order
            measure["events"] = retained
            order_maps[(measure.get("number"), measure.get("page_measure_index"))] = order_map
    _remap_tie_edges_after_pruning(score_ir, order_maps)
    score_ir["nearby_gp8_duplicate_score_event_pruning"] = summary
    return summary


def prune_unpitched_gp8_note_events(score_ir: dict) -> dict:
    """Drop GP8 score proposals that cannot represent a musical event.

    Official GP8 score+TAB PDFs sometimes place a stem-side notehead far
    enough from its TAB attack for the score locator to emit a second event.
    We cannot reject that proposal before tie resolution because a real tied
    continuation also has no printed TAB number.  After tie resolution,
    however, an event whose active voices are all ``note`` and which still has
    no note is neither a rest nor an exportable continuation.  Restricting the
    cleanup to authoritative GP8 vector PDFs keeps the raster and TuxGuitar
    paths unchanged.
    """
    summary = {"removed": 0, "measures_changed": 0}
    order_maps = {}
    for track in score_ir.get("tracks", []):
        for measure in track.get("measures", []):
            if not (
                measure.get("pdf_vector_gp8_signature")
                and measure.get("pdf_vector_tab_authoritative")
            ):
                continue
            retained = []
            order_map = {}
            changed = False
            for event in measure.get("events", []):
                old_order = int(event.get("order", len(retained)))
                active_voices = [
                    voice for voice in event.get("voices", [])
                    if voice.get("state") != "empty"
                ]
                unpitched_note = bool(
                    active_voices
                    and all(voice.get("state") == "note" for voice in active_voices)
                    and not event.get("notes")
                    and event.get("tab_match") != "matched"
                )
                if unpitched_note:
                    summary["removed"] += 1
                    changed = True
                    continue
                order_map[old_order] = len(retained)
                retained.append(event)
            if changed:
                summary["measures_changed"] += 1
            for order, event in enumerate(retained):
                event["order"] = order
            measure["events"] = retained
            order_maps[(measure.get("number"), measure.get("page_measure_index"))] = order_map
    _remap_tie_edges_after_pruning(score_ir, order_maps)
    score_ir["unpitched_gp8_note_event_pruning"] = summary
    return summary
