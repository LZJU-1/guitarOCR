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
        float(event["locator_confidence"]) < 0.60
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
            if not proposal.get("plausible") and not closure_override:
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
                    "selection": "plausible" if proposal.get("plausible") else "unique_measure_closure",
                }
                summary["events_changed"] += 1
                changed = True
            summary["measures_changed"] += int(changed)
    audit_score_ir(score_ir)
    score_ir["rhythm_constraint_corrections"] = summary
    return summary
