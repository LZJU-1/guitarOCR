from __future__ import annotations

import math
from fractions import Fraction


PROPOSAL_RELATIVE_PROBABILITY_THRESHOLD = 0.20


def duration_fraction(duration_value: int, dot: str, division: str) -> Fraction:
    value = Fraction(1, int(duration_value))
    if dot == "single":
        value *= Fraction(3, 2)
    elif dot == "double":
        value *= Fraction(7, 4)
    enters, times = (int(part) for part in division.split(":", 1))
    value *= Fraction(times, enters)
    return value


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
