from __future__ import annotations

import argparse
import base64
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys

from guitarocr.paths import JAVA_SOURCE_ROOT, PROJECT_ROOT
from guitarocr.tuxguitar_runtime import (
    java_classpath,
    java_executable,
    javac_executable,
    require_tuxguitar_root,
)


STANDARD_GUITAR_TUNING = [64, 59, 55, 50, 45, 40]


def parse_fraction(value: object) -> Fraction | None:
    if isinstance(value, dict):
        value = value.get("text")
    if not isinstance(value, str) or "/" not in value:
        return None
    numerator, denominator = value.split("/", 1)
    try:
        return Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError):
        return None


def duration_fraction(voice: dict) -> Fraction | None:
    explicit = parse_fraction(voice.get("duration_fraction"))
    if explicit is not None:
        return explicit
    value = voice.get("duration_value")
    if not isinstance(value, int) or value <= 0:
        return None
    result = Fraction(1, value)
    dot = voice.get("dot", "none")
    if dot == "single":
        result *= Fraction(3, 2)
    elif dot == "double":
        result *= Fraction(7, 4)
    division = voice.get("division", "1:1")
    try:
        enters, times = (int(part) for part in division.split(":", 1))
        if enters > 0 and times > 0:
            result *= Fraction(times, enters)
    except (AttributeError, ValueError):
        return None
    return result


def split_rest_duration(value: Fraction) -> list[tuple[int, str, str, Fraction]] | None:
    candidates: list[tuple[int, str, str, Fraction]] = []
    for division, multiplier in (("1:1", Fraction(1)), ("3:2", Fraction(2, 3))):
        for dot, dot_multiplier in (
            ("none", Fraction(1)),
            ("single", Fraction(3, 2)),
        ):
            for duration_value in (1, 2, 4, 8, 16, 32, 64):
                duration = Fraction(1, duration_value) * dot_multiplier * multiplier
                candidates.append((duration_value, dot, division, duration))
    candidates.sort(
        key=lambda item: (
            item[3],
            item[2] == "1:1",
            item[1] == "none",
        ),
        reverse=True,
    )

    @lru_cache(maxsize=None)
    def solve(remaining: Fraction) -> tuple[tuple[int, str, str, Fraction], ...] | None:
        if remaining == 0:
            return ()
        for candidate in candidates:
            if candidate[3] > remaining:
                continue
            suffix = solve(remaining - candidate[3])
            if suffix is not None:
                return (candidate, *suffix)
        return None

    result = solve(value)
    return list(result) if result is not None else None


def issue(report: dict, code: str, message: str, **location: object) -> None:
    report["issue_counts"][code] += 1
    if len(report["issues"]) < 200:
        report["issues"].append({"code": code, "message": message, **location})


def active_voice_indices(score_ir: dict) -> list[int]:
    tracks = score_ir.get("tracks") or []
    if not tracks:
        return []
    voices = {
        int(voice["voice"])
        for measure in tracks[0].get("measures", [])
        for event in measure.get("events", [])
        for voice in event.get("voices", [])
        if voice.get("state") in {"note", "rest"}
        and isinstance(voice.get("voice"), int)
    }
    return sorted(value for value in voices if value in (0, 1))


def ambiguous_note_voice(event: dict) -> int:
    """Choose one active voice for TAB notes whose stem association is ambiguous.

    A common two-voice pattern has an explicit tied note in one voice and a newly
    attacked printed fret in the other.  In that case the unassigned fret belongs
    to the only active voice that does not already own an explicit note.
    """
    active = {
        int(voice["voice"])
        for voice in event.get("voices", [])
        if voice.get("state") == "note" and isinstance(voice.get("voice"), int)
    }
    explicit = {
        int(note["voice"])
        for note in event.get("notes", [])
        if isinstance(note.get("voice"), int)
    }
    missing = active - explicit
    if len(missing) == 1:
        return next(iter(missing))
    if len(active) == 1:
        return next(iter(active))
    return min(active) if active else 0


def resolve_title(score_ir: dict, ir_path: Path, supplied: str | None) -> str:
    if supplied:
        return supplied
    pages = score_ir.get("document", {}).get("pages", [])
    if pages:
        source = pages[0].get("source_pdf") or pages[0].get("image")
        if source:
            return Path(source).stem
    return ir_path.parent.name if ir_path.stem == "document_score_ir" else ir_path.stem


def resolve_tuning(track: dict, supplied: list[int] | None, report: dict) -> list[int]:
    tuning = supplied or track.get("string_tuning_midi")
    string_count = int(track.get("string_count") or 6)
    if tuning is None:
        if string_count != 6:
            raise ValueError(
                f"No tuning is present for a {string_count}-string track; pass --tuning explicitly"
            )
        report["assumptions"].append(
            "string_tuning_midi was absent; used standard guitar E4,B3,G3,D3,A2,E2"
        )
        tuning = STANDARD_GUITAR_TUNING
    tuning = [int(value) for value in tuning]
    if len(tuning) != string_count:
        raise ValueError(f"Tuning has {len(tuning)} pitches but IR declares {string_count} strings")
    return tuning


def build_plan(
    score_ir: dict,
    ir_path: Path,
    plan_path: Path,
    *,
    voice_index: int,
    tempo: int | None,
    tuning: list[int] | None,
    title: str | None,
) -> dict:
    tracks = score_ir.get("tracks") or []
    if not tracks:
        raise ValueError("IR contains no tracks")
    track = tracks[0]
    measures = track.get("measures") or []
    if not measures:
        raise ValueError("IR contains no measures")

    report = {
        "schema_version": "1.0",
        "input_ir": str(ir_path.resolve()),
        "voice": voice_index,
        "assumptions": [],
        "issue_counts": Counter(),
        "issues": [],
        "plan": {},
    }
    actual_tempo = tempo or track.get("tempo_quarter")
    if actual_tempo is None:
        actual_tempo = 120
        report["assumptions"].append("tempo_quarter was absent; used 120 BPM")
    actual_tempo = int(actual_tempo)
    if not 20 <= actual_tempo <= 400:
        raise ValueError("Tempo must be between 20 and 400 BPM")
    actual_tuning = resolve_tuning(track, tuning, report)
    actual_title = resolve_title(score_ir, ir_path, title)
    if track.get("capo") is None:
        report["assumptions"].append("capo was absent; used capo 0")
    elif int(track["capo"]) != 0:
        report["assumptions"].append(
            f"capo {track['capo']} is not serialized by the MVP GP5 writer"
        )

    encoded_title = base64.b64encode(actual_title.encode("utf-8")).decode("ascii")
    lines = [
        "GUITAROCR_PLAN\t2",
        f"META\tTITLE_B64\t{encoded_title}",
        f"META\tTEMPO\t{actual_tempo}",
        "META\tTUNING\t" + ",".join(str(value) for value in actual_tuning),
    ]
    current_signature: tuple[int, int] | None = None
    exported_events = 0
    exported_notes = 0
    exported_rests = 0
    generated_rests = 0
    tied_notes = 0
    pick_stroke_events = 0
    exact_measures = 0

    for measure_index, measure in enumerate(measures):
        signature = measure.get("time_signature")
        if (
            isinstance(signature, list)
            and len(signature) == 2
            and all(isinstance(value, int) and value > 0 for value in signature)
        ):
            current_signature = (signature[0], signature[1])
        elif current_signature is None:
            current_signature = (4, 4)
            issue(
                report,
                "default_time_signature",
                "No current time signature; used 4/4",
                measure=measure.get("number") or measure_index + 1,
            )
        else:
            issue(
                report,
                "carried_time_signature",
                "Missing time signature; carried the preceding value",
                measure=measure.get("number") or measure_index + 1,
            )
        numerator, denominator = current_signature
        capacity = Fraction(numerator, denominator)
        measure_number = measure.get("number") or measure_index + 1
        lines.append(
            f"MEASURE\t{measure_index}\t{int(measure_number)}\t{numerator}\t{denominator}"
        )
        audit_status = (
            measure.get("rhythm_audit", {})
            .get("voices", {})
            .get(f"voice_{voice_index}", {})
            .get("status")
        )
        exact_measures += int(audit_status == "exact")
        selected_by_onset: dict[Fraction, tuple[int, dict, dict, Fraction]] = {}
        for event in measure.get("events", []):
            voices = [
                value for value in event.get("voices", [])
                if int(value.get("voice", -1)) == voice_index
            ]
            if not voices:
                continue
            voice = voices[0]
            state = voice.get("state")
            if state not in {"note", "rest"}:
                continue
            onset = parse_fraction(voice.get("onset"))
            duration = duration_fraction(voice)
            location = {"measure": measure_number, "event_order": event.get("order")}
            if onset is None or duration is None:
                issue(report, "missing_timing", "Skipped event with incomplete timing", **location)
                continue
            if onset < 0 or onset >= capacity or onset + duration > capacity:
                issue(
                    report,
                    "outside_measure",
                    f"Skipped event at {onset} with duration {duration} outside {numerator}/{denominator}",
                    **location,
                )
                continue
            if onset in selected_by_onset:
                previous_order = selected_by_onset[onset][0]
                issue(
                    report,
                    "duplicate_onset",
                    f"Kept event {previous_order}; skipped a second event at onset {onset}",
                    **location,
                )
                continue
            selected_by_onset[onset] = (int(event.get("order", 0)), event, voice, duration)

        prepared_rows: list[dict] = []
        for onset, (event_order, event, voice, duration) in sorted(selected_by_onset.items()):
            state = voice["state"]
            duration_value = int(voice["duration_value"])
            dot = voice.get("dot") or "none"
            division = voice.get("division") or "1:1"
            notes_text = "-"
            technique = event.get("technique_prediction") or {}
            positive = technique.get("positive") or {}
            pick_up = bool(positive.get("pick_up", False))
            pick_down = bool(positive.get("pick_down", False))
            if pick_up and pick_down:
                probabilities = technique.get("probabilities") or {}
                pick_up = float(probabilities.get("pick_up", 0.0)) >= float(
                    probabilities.get("pick_down", 0.0)
                )
                pick_down = not pick_up
                issue(
                    report,
                    "conflicting_pick_stroke",
                    "Both pick directions passed threshold; kept the higher probability",
                    measure=measure_number,
                    event_order=event_order,
                )
            pick_stroke = 1 if pick_up else -1 if pick_down else 0
            pick_stroke_events += int(pick_stroke != 0)
            if state == "note":
                notes_by_string: dict[int, dict] = {}
                fallback_voice = ambiguous_note_voice(event)
                for note in event.get("notes", []):
                    note_voice = note.get("voice")
                    if note_voice not in (None, voice_index):
                        continue
                    if note_voice is None and fallback_voice != voice_index:
                        continue
                    string = note.get("string")
                    fret = note.get("fret")
                    if not isinstance(string, int) or not 1 <= string <= len(actual_tuning):
                        issue(
                            report,
                            "invalid_string",
                            f"Ignored invalid string {string}",
                            measure=measure_number,
                            event_order=event_order,
                        )
                        continue
                    is_dead = bool(note.get("dead") or note.get("muted")) or (
                        isinstance(fret, str) and fret.upper() == "X"
                    )
                    if is_dead:
                        # Dead notes are printed as X. Their sounding fret is not recoverable from
                        # the raster, so GP receives fret zero plus the exact dead-note semantic.
                        fret = int(note.get("underlying_fret", 0))
                    if not isinstance(fret, int) or not 0 <= fret <= 99:
                        issue(
                            report,
                            "unsupported_fret",
                            f"Ignored unsupported fret {fret}",
                            measure=measure_number,
                            event_order=event_order,
                        )
                        continue
                    if note_voice is None:
                        report["issue_counts"]["ambiguous_voice_note_used"] += 1
                    if string in notes_by_string:
                        issue(
                            report,
                            "duplicate_string",
                            f"Kept the first of multiple frets detected on string {string}",
                            measure=measure_number,
                            event_order=event_order,
                        )
                        continue
                    notes_by_string[string] = note
                if not notes_by_string:
                    issue(
                        report,
                        "pitchless_note_event",
                        "Skipped a note event without a resolved TAB pitch",
                        measure=measure_number,
                        event_order=event_order,
                    )
                    continue
                note_parts = []
                for string, note in sorted(notes_by_string.items()):
                    tied = bool(note.get("tie_in"))
                    fret = note.get("fret")
                    dead = bool(note.get("dead") or note.get("muted")) or (
                        isinstance(fret, str) and fret.upper() == "X"
                    )
                    if dead and tied:
                        # GP5/TuxGuitar serializes a tied dead note as a normal
                        # tied fret and silently drops the visible X. Prefer
                        # the printed dead-note state and retain the richer
                        # tie+dead combination in IR for a future GPIF writer.
                        tied = False
                        issue(
                            report,
                            "gp5_dead_tie_conflict",
                            "GP5 cannot retain tie-in and dead-note on one note; kept the X",
                            measure=measure_number,
                            event_order=event_order,
                            string=string,
                        )
                    tied_notes += int(tied)
                    encoded_fret = int(note.get("underlying_fret", 0)) if dead else int(fret)
                    effects = note.get("effects") or {}
                    slide = bool(effects.get("slide", note.get("slide", False)))
                    if dead and slide:
                        # TuxGuitar's GP5 note model treats dead-note and slide as
                        # mutually exclusive setters. Prefer the visible/audible X;
                        # the richer IR keeps both semantics for a future GPIF writer.
                        slide = False
                        issue(
                            report,
                            "gp5_dead_slide_conflict",
                            "GP5 cannot retain slide and dead-note on one note; kept the X",
                            measure=measure_number,
                            event_order=event_order,
                            string=string,
                        )
                    flags = [
                        dead,
                        bool(effects.get("vibrato", note.get("vibrato", False))),
                        slide,
                        bool(effects.get("hammer", note.get("hammer", False))),
                        bool(effects.get("bend", note.get("bend", False))),
                        bool(effects.get("ghost", note.get("ghost", False))),
                        bool(effects.get("accent", note.get("accent", False))),
                        bool(effects.get("palm_mute", note.get("palm_mute", False))),
                        bool(effects.get("staccato", note.get("staccato", False))),
                        bool(effects.get("let_ring", note.get("let_ring", False))),
                        bool(effects.get("tapping", note.get("tapping", False))),
                    ]
                    note_parts.append(
                        ":".join(
                            [str(string), str(encoded_fret), str(int(tied))]
                            + [str(int(value)) for value in flags]
                        )
                    )
                notes_text = ",".join(note_parts)
                exported_notes += len(note_parts)
            else:
                exported_rests += 1
            prepared_rows.append(
                {
                    "onset": onset,
                    "event_order": event_order,
                    "duration_value": duration_value,
                    "dot": dot,
                    "division": division,
                    "state": state,
                    "notes": notes_text,
                    "pick_stroke": pick_stroke,
                    "duration": duration,
                    "generated": False,
                }
            )

        complete_rows: list[dict] = []
        cursor = Fraction(0)
        for row in prepared_rows:
            onset = row["onset"]
            if onset < cursor:
                issue(
                    report,
                    "overlapping_event",
                    f"Skipped event at {onset}; preceding event ends at {cursor}",
                    measure=measure_number,
                    event_order=row["event_order"],
                )
                continue
            if onset > cursor:
                parts = split_rest_duration(onset - cursor)
                if parts is None:
                    raise ValueError(
                        f"Cannot represent the rest gap {onset - cursor} in measure {measure_number}"
                    )
                for duration_value, dot, division, duration in parts:
                    complete_rows.append(
                        {
                            "onset": cursor,
                            "event_order": -100000 - generated_rests,
                            "duration_value": duration_value,
                            "dot": dot,
                            "division": division,
                            "state": "generated_rest",
                            "notes": "-",
                            "pick_stroke": 0,
                            "duration": duration,
                            "generated": True,
                        }
                    )
                    generated_rests += 1
                    cursor += duration
            complete_rows.append(row)
            cursor = onset + row["duration"]
        if cursor < capacity:
            parts = split_rest_duration(capacity - cursor)
            if parts is None:
                raise ValueError(
                    f"Cannot represent the trailing rest {capacity - cursor} in measure {measure_number}"
                )
            for duration_value, dot, division, duration in parts:
                complete_rows.append(
                    {
                        "onset": cursor,
                        "event_order": -100000 - generated_rests,
                        "duration_value": duration_value,
                        "dot": dot,
                        "division": division,
                        "state": "generated_rest",
                        "notes": "-",
                        "pick_stroke": 0,
                        "duration": duration,
                        "generated": True,
                    }
                )
                generated_rests += 1
                cursor += duration

        for row in complete_rows:
            onset = row["onset"]
            lines.append(
                "\t".join(
                    [
                        "EVENT",
                        str(measure_index),
                        str(row["event_order"]),
                        str(onset.numerator),
                        str(onset.denominator),
                        str(row["duration_value"]),
                        str(row["dot"]),
                        str(row["division"]),
                        str(row["state"]),
                        str(row["notes"]),
                        str(row["pick_stroke"]),
                    ]
                )
            )
            exported_events += 1

    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["issue_counts"] = dict(sorted(report["issue_counts"].items()))
    report["plan"] = {
        "title": actual_title,
        "tempo_quarter": actual_tempo,
        "tuning_midi_high_to_low": actual_tuning,
        "measure_count": len(measures),
        "rhythm_exact_measure_count": exact_measures,
        "exported_event_count": exported_events,
        "exported_note_count": exported_notes,
        "exported_rest_count": exported_rests,
        "generated_structural_rest_count": generated_rests,
        "tied_note_count": tied_notes,
        "pick_stroke_event_count": pick_stroke_events,
    }
    return report


def build_all_voice_plan(
    score_ir: dict,
    ir_path: Path,
    plan_path: Path,
    *,
    tempo: int | None,
    tuning: list[int] | None,
    title: str | None,
) -> dict:
    voices = active_voice_indices(score_ir)
    if not voices:
        raise ValueError("IR contains no active note/rest voices")

    voice_reports: list[tuple[int, dict]] = []
    voice_lines: list[tuple[int, list[str]]] = []
    try:
        for voice in voices:
            temporary = plan_path.with_name(f"{plan_path.stem}.voice{voice}.tmp.tsv")
            report = build_plan(
                score_ir,
                ir_path,
                temporary,
                voice_index=voice,
                tempo=tempo,
                tuning=tuning,
                title=title,
            )
            voice_reports.append((voice, report))
            voice_lines.append((voice, temporary.read_text(encoding="utf-8").splitlines()))

        primary_lines = voice_lines[0][1]
        output_lines = ["GUITAROCR_PLAN\t3"]
        output_lines.extend(line for line in primary_lines if line.startswith("META\t"))
        output_lines.extend(line for line in primary_lines if line.startswith("MEASURE\t"))
        for voice, lines in voice_lines:
            for line in lines:
                if not line.startswith("EVENT\t"):
                    continue
                fields = line.split("\t")
                # v3 adds voice index after measure/event-order.
                if voice != voices[0]:
                    # Pick direction is a TGBeat property, not a per-voice one.
                    fields[10] = "0"
                output_lines.append("\t".join([*fields[:3], str(voice), *fields[3:]]))
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    finally:
        for voice in voices:
            plan_path.with_name(f"{plan_path.stem}.voice{voice}.tmp.tsv").unlink(missing_ok=True)

    primary = voice_reports[0][1]
    aggregate_counts: Counter = Counter()
    aggregate_issues: list[dict] = []
    assumptions: list[str] = []
    for voice, report in voice_reports:
        aggregate_counts.update(report["issue_counts"])
        aggregate_issues.extend({**item, "voice": voice} for item in report["issues"])
        for assumption in report["assumptions"]:
            if assumption not in assumptions:
                assumptions.append(assumption)
    per_voice = {str(voice): report["plan"] for voice, report in voice_reports}
    plan = {
        **primary["plan"],
        "active_voices": voices,
        "rhythm_exact_measure_count": min(
            report["plan"]["rhythm_exact_measure_count"] for _, report in voice_reports
        ),
        "rhythm_exact_voice_measure_count": sum(
            report["plan"]["rhythm_exact_measure_count"] for _, report in voice_reports
        ),
        "exported_event_count": sum(
            report["plan"]["exported_event_count"] for _, report in voice_reports
        ),
        "exported_note_count": sum(
            report["plan"]["exported_note_count"] for _, report in voice_reports
        ),
        "exported_rest_count": sum(
            report["plan"]["exported_rest_count"] for _, report in voice_reports
        ),
        "generated_structural_rest_count": sum(
            report["plan"]["generated_structural_rest_count"] for _, report in voice_reports
        ),
        "tied_note_count": sum(
            report["plan"]["tied_note_count"] for _, report in voice_reports
        ),
        "pick_stroke_event_count": primary["plan"]["pick_stroke_event_count"],
        "per_voice": per_voice,
    }
    return {
        "schema_version": "1.0",
        "input_ir": str(ir_path.resolve()),
        "voice": "all",
        "assumptions": assumptions,
        "issue_counts": dict(sorted(aggregate_counts.items())),
        "issues": aggregate_issues[:200],
        "plan": plan,
    }


def compile_writer(root: Path) -> tuple[Path, str]:
    source = JAVA_SOURCE_ROOT / "TuxGuitarIrGp5Writer.java"
    classes = root / "database" / "tmp" / "gp_writer_classes"
    target = classes / "TuxGuitarIrGp5Writer.class"
    if not source.is_file():
        raise FileNotFoundError(source)
    classes.mkdir(parents=True, exist_ok=True)
    classpath = java_classpath(classes)
    if not target.is_file() or target.stat().st_mtime < source.stat().st_mtime:
        command = [str(javac_executable()), "-encoding", "UTF-8", "-cp", classpath, "-d", str(classes), str(source)]
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode:
            raise RuntimeError(f"TuxGuitar writer compilation failed:\n{completed.stdout}{completed.stderr}")
    return classes, classpath


def run_writer(
    root: Path,
    plan: Path,
    output: Path,
    preview: Path | None,
    preview_layout: str = "score_tab",
) -> dict:
    _, classpath = compile_writer(root)
    java = java_executable()
    tuxguitar_root = require_tuxguitar_root()
    output.parent.mkdir(parents=True, exist_ok=True)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(java),
        "-Xmx2g",
        "-Djava.awt.headless=true",
        f"-Dtuxguitar.home.path={tuxguitar_root}",
        "-cp",
        classpath,
        "TuxGuitarIrGp5Writer",
        str(plan),
        str(output),
    ]
    if preview is not None:
        command.append(str(preview))
        command.append(preview_layout)
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode:
        raise RuntimeError(f"TuxGuitar GP5 writer failed:\n{completed.stdout}{completed.stderr}")
    values: dict[str, object] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().lower()] = value.strip()
    values["stdout"] = completed.stdout.strip()
    if completed.stderr.strip():
        values["stderr"] = completed.stderr.strip()
    return values


def export_ir(
    ir_path: Path,
    output: Path,
    *,
    preview: Path | None,
    voice: int | None,
    tempo: int | None,
    tuning: list[int] | None,
    title: str | None,
    preview_layout: str = "score_tab",
) -> dict:
    root = PROJECT_ROOT
    ir_path = ir_path.resolve()
    output = output.resolve()
    score_ir = json.loads(ir_path.read_text(encoding="utf-8"))
    plan = output.with_suffix(".plan.tsv")
    if voice is None:
        report = build_all_voice_plan(
            score_ir,
            ir_path,
            plan,
            tempo=tempo,
            tuning=tuning,
            title=title,
        )
    else:
        report = build_plan(
            score_ir,
            ir_path,
            plan,
            voice_index=voice,
            tempo=tempo,
            tuning=tuning,
            title=title,
        )
    java_result = run_writer(
        root,
        plan,
        output,
        preview.resolve() if preview else None,
        preview_layout,
    )
    report.update(
        {
            "output_gp5": str(output),
            "preview_pdf": str(preview.resolve()) if preview else None,
            "preview_layout": preview_layout if preview else None,
            "plan_file": str(plan),
            "tuxguitar": java_result,
        }
    )
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_file"] = str(report_path)
    return report


def parse_tuning_arg(value: str) -> list[int]:
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Tuning must be comma-separated MIDI pitches") from error


def parse_voice_arg(value: str) -> int | None:
    if value.lower() == "all":
        return None
    if value in {"0", "1"}:
        return int(value)
    raise argparse.ArgumentTypeError("Voice must be all, 0, or 1")


def main() -> None:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description="Export GuitarOCR document_score_ir.json to a minimal, playable GP5 file."
    )
    parser.add_argument("score_ir", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--preview-pdf", type=Path)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--voice",
        type=parse_voice_arg,
        default=None,
        metavar="{all,0,1}",
        help="Export both recognized voices by default; pass 0 or 1 to select one voice.",
    )
    parser.add_argument("--tempo", type=int)
    parser.add_argument("--tuning", type=parse_tuning_arg)
    parser.add_argument("--title")
    parser.add_argument(
        "--preview-layout",
        choices=("score_tab", "tab_only", "score_only"),
        default="score_tab",
        help="notation layout for the optional preview PDF (default: score_tab)",
    )
    args = parser.parse_args()
    output = args.output or root / "output" / "gp" / f"{args.score_ir.parent.name}_ocr.gp5"
    if output.suffix.lower() != ".gp5":
        parser.error("The MVP writer currently outputs .gp5 only")
    preview = None
    if not args.no_preview:
        preview = args.preview_pdf or root / "output" / "pdf" / f"{output.stem}_preview.pdf"
    try:
        report = export_ir(
            args.score_ir,
            output,
            preview=preview,
            voice=args.voice,
            tempo=args.tempo,
            tuning=args.tuning,
            title=args.title,
            preview_layout=args.preview_layout,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
