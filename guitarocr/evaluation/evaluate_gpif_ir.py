from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from guitarocr.data.gpif import load_gpif_score


def _voice(event: dict, voice_index: int = 0) -> dict | None:
    if "state" in event and "duration_value" in event:
        return event if int(event.get("voice", 0)) == voice_index else None
    return next((value for value in event.get("voices", []) if int(value.get("voice", -1)) == voice_index), None)


def _notes(event: dict) -> Counter:
    result: Counter = Counter()
    for note in event.get("notes", []):
        fret = note.get("printed_fret", note.get("fret"))
        if isinstance(fret, str):
            fret = fret.upper()
        result[(note.get("string"), fret)] += 1
    return result


def _rhythm(event: dict) -> tuple | None:
    voice = _voice(event)
    if voice is None:
        return None
    return (
        voice.get("state"), voice.get("duration_value"),
        voice.get("dot", "none"), voice.get("division", "1:1"),
    )


def _effects(event: dict) -> set[str]:
    result: set[str] = set()
    for name, value in (event.get("effects") or {}).items():
        if value is True:
            result.add(name)
    for note in event.get("notes", []):
        for name, value in (note.get("effects") or {}).items():
            if value is True and name != "slide_flags":
                result.add(name)
        if note.get("dead") or note.get("muted") or str(note.get("fret", "")).upper() == "X":
            result.add("muted")
    return result


def _pair_cost(expected: dict, actual: dict) -> float:
    er, ar = _rhythm(expected), _rhythm(actual)
    en, an = _notes(expected), _notes(actual)
    intersection = sum((en & an).values())
    union = sum((en | an).values())
    note_cost = 1.0 - (intersection / union if union else 1.0)
    state_cost = 0.0 if er and ar and er[0] == ar[0] else 1.0
    rhythm_cost = 0.0 if er == ar else 0.5
    return 0.65 * note_cost + 0.20 * state_cost + 0.15 * rhythm_cost


def _align(expected: list[dict], actual: list[dict]) -> list[tuple[int | None, int | None]]:
    rows, cols = len(expected), len(actual)
    dp = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    step: list[list[str]] = [[""] * (cols + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        dp[row][0], step[row][0] = float(row), "delete"
    for col in range(1, cols + 1):
        dp[0][col], step[0][col] = float(col), "insert"
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            choices = [
                (dp[row - 1][col - 1] + _pair_cost(expected[row - 1], actual[col - 1]), "match"),
                (dp[row - 1][col] + 1.0, "delete"),
                (dp[row][col - 1] + 1.0, "insert"),
            ]
            dp[row][col], step[row][col] = min(choices, key=lambda value: value[0])
    aligned: list[tuple[int | None, int | None]] = []
    row, col = rows, cols
    while row or col:
        action = step[row][col]
        if action == "match":
            row -= 1; col -= 1; aligned.append((row, col))
        elif action == "delete":
            row -= 1; aligned.append((row, None))
        else:
            col -= 1; aligned.append((None, col))
    return list(reversed(aligned))


def evaluate(gpif: dict, score_ir: dict) -> dict:
    expected_measures = gpif["measures"]
    tracks = score_ir.get("tracks", [])
    actual_measures = tracks[0].get("measures", []) if tracks else []
    totals = Counter()
    string_totals: dict[int, Counter] = {}
    technique_totals: dict[str, Counter] = {}
    cases: list[dict] = []
    measure_rows: list[dict] = []
    for index in range(max(len(expected_measures), len(actual_measures))):
        expected = expected_measures[index] if index < len(expected_measures) else {"events": []}
        actual = actual_measures[index] if index < len(actual_measures) else {"events": []}
        expected_events = [event for event in expected.get("events", []) if int(event.get("voice", 0)) == 0]
        actual_events = [event for event in actual.get("events", []) if _voice(event) is not None]
        totals["expected_events"] += len(expected_events)
        totals["actual_events"] += len(actual_events)
        local = Counter()
        differences: list[dict] = []
        for expected_index, actual_index in _align(expected_events, actual_events):
            if expected_index is None:
                local["extra_events"] += 1
                extra_notes = _notes(actual_events[actual_index])
                local["actual_notes"] += sum(extra_notes.values())
                for (string, _), count in extra_notes.items():
                    string_totals.setdefault(int(string), Counter())["actual"] += count
                differences.append({"kind": "extra_event", "actual": actual_index})
                continue
            if actual_index is None:
                local["missed_events"] += 1
                missed_notes = _notes(expected_events[expected_index])
                local["expected_notes"] += sum(missed_notes.values())
                for (string, _), count in missed_notes.items():
                    string_totals.setdefault(int(string), Counter())["expected"] += count
                differences.append({"kind": "missed_event", "expected": expected_index})
                continue
            local["matched_events"] += 1
            expected_event, actual_event = expected_events[expected_index], actual_events[actual_index]
            er, ar = _rhythm(expected_event), _rhythm(actual_event)
            en, an = _notes(expected_event), _notes(actual_event)
            common = sum((en & an).values())
            local["expected_notes"] += sum(en.values())
            local["actual_notes"] += sum(an.values())
            local["correct_notes"] += common
            for string in {int(key[0]) for key in en} | {int(key[0]) for key in an}:
                expected_on_string = Counter({key: count for key, count in en.items() if int(key[0]) == string})
                actual_on_string = Counter({key: count for key, count in an.items() if int(key[0]) == string})
                values = string_totals.setdefault(string, Counter())
                values["expected"] += sum(expected_on_string.values())
                values["actual"] += sum(actual_on_string.values())
                values["correct"] += sum((expected_on_string & actual_on_string).values())
            local["rhythm_exact"] += int(er == ar)
            local["notes_exact"] += int(en == an)
            local["event_exact"] += int(er == ar and en == an)
            expected_effects, actual_effects = _effects(expected_event), _effects(actual_event)
            for name in expected_effects | actual_effects:
                values = technique_totals.setdefault(name, Counter())
                values["tp"] += int(name in expected_effects and name in actual_effects)
                values["fp"] += int(name not in expected_effects and name in actual_effects)
                values["fn"] += int(name in expected_effects and name not in actual_effects)
                values["support"] += int(name in expected_effects)
            expected_x = Counter({key: count for key, count in en.items() if key[1] == "X"})
            actual_x = Counter({key: count for key, count in an.items() if key[1] == "X"})
            local["expected_muted_x"] += sum(expected_x.values())
            local["actual_muted_x"] += sum(actual_x.values())
            local["correct_muted_x"] += sum((expected_x & actual_x).values())
            if er != ar or en != an:
                differences.append({
                    "kind": "mismatch", "expected": expected_index, "actual": actual_index,
                    "expected_rhythm": er, "actual_rhythm": ar,
                    "expected_notes": [list(key) + [count] for key, count in sorted(en.items(), key=str)],
                    "actual_notes": [list(key) + [count] for key, count in sorted(an.items(), key=str)],
                })
        totals.update(local)
        exact_measure = not differences and len(expected_events) == len(actual_events)
        totals["exact_measures"] += int(exact_measure)
        if differences:
            cases.append({"measure": index + 1, "differences": differences[:12]})
        measure_rows.append({
            "measure": index + 1,
            "expected_events": len(expected_events), "actual_events": len(actual_events),
            "exact": exact_measure, **local,
        })

    def ratio(numerator: str, denominator: str) -> float:
        return totals[numerator] / totals[denominator] if totals[denominator] else 0.0

    note_precision = ratio("correct_notes", "actual_notes")
    note_recall = ratio("correct_notes", "expected_notes")
    muted_precision = ratio("correct_muted_x", "actual_muted_x")
    muted_recall = ratio("correct_muted_x", "expected_muted_x")
    metrics = {
        "measure_count_expected": len(expected_measures),
        "measure_count_actual": len(actual_measures),
        "measure_exact_accuracy": totals["exact_measures"] / len(expected_measures) if expected_measures else 0.0,
        "event_count_expected": totals["expected_events"],
        "event_count_actual": totals["actual_events"],
        "matched_events": totals["matched_events"],
        "event_exact_accuracy_on_matched": ratio("event_exact", "matched_events"),
        "rhythm_exact_accuracy_on_matched": ratio("rhythm_exact", "matched_events"),
        "notes_exact_accuracy_on_matched": ratio("notes_exact", "matched_events"),
        "note_precision": note_precision,
        "note_recall": note_recall,
        "note_f1": 2 * note_precision * note_recall / (note_precision + note_recall) if note_precision + note_recall else 0.0,
        "muted_x_expected": totals["expected_muted_x"],
        "muted_x_actual": totals["actual_muted_x"],
        "muted_x_precision": muted_precision,
        "muted_x_recall": muted_recall,
        "muted_x_f1": 2 * muted_precision * muted_recall / (muted_precision + muted_recall) if muted_precision + muted_recall else 0.0,
        "missed_events": totals["missed_events"],
        "extra_events": totals["extra_events"],
    }
    metrics["notes_by_string"] = {
        str(string): {
            "expected": values["expected"],
            "actual": values["actual"],
            "precision": values["correct"] / max(1, values["actual"]),
            "recall": values["correct"] / max(1, values["expected"]),
        }
        for string, values in sorted(string_totals.items())
    }
    technique_metrics = {}
    for name, values in sorted(technique_totals.items()):
        precision = values["tp"] / max(1, values["tp"] + values["fp"])
        recall = values["tp"] / max(1, values["tp"] + values["fn"])
        technique_metrics[name] = {
            "support": values["support"], "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    metrics["techniques"] = technique_metrics
    return {
        "schema_version": "1.0",
        "source_gp": gpif["source"],
        "track": {key: gpif[key] for key in ("track_index", "track_name", "tempo_quarter", "string_count", "string_tuning_midi")},
        "metrics": metrics,
        "totals": dict(totals),
        "failure_cases": cases,
        "measures": measure_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pixel OCR IR with exact Guitar Pro 7/8 GPIF semantics")
    parser.add_argument("gp", type=Path)
    parser.add_argument("score_ir", type=Path)
    parser.add_argument("--track-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ground-truth-output", type=Path)
    args = parser.parse_args()
    gpif = load_gpif_score(args.gp, args.track_index)
    score_ir = json.loads(args.score_ir.read_text(encoding="utf-8"))
    result = evaluate(gpif, score_ir)
    if args.ground_truth_output:
        args.ground_truth_output.parent.mkdir(parents=True, exist_ok=True)
        args.ground_truth_output.write_text(json.dumps(gpif, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
