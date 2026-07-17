from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote, unquote


DURATION_NAMES = {
    1: "w",
    2: "h",
    4: "q",
    8: "e",
    16: "s",
    32: "t",
    64: "f",
}

SLIDE_NAMES = {
    "legatoSlideTo": "sl",
    "shiftSlideTo": "ss",
    "intoFromBelow": "sib",
    "intoFromAbove": "sia",
    "outDownwards": "sod",
    "outUpwards": "sou",
}


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def _truthy_effect(effect: Any, *names: str) -> bool:
    return any(bool(getattr(effect, name, False)) for name in names)


def _duration_dict(duration: Any) -> dict[str, Any]:
    tuplet = getattr(duration, "tuplet", None)
    enters = int(getattr(tuplet, "enters", 1) or 1)
    times = int(getattr(tuplet, "times", 1) or 1)
    return {
        "value": int(getattr(duration, "value", 4) or 4),
        "dotted": bool(getattr(duration, "isDotted", False)),
        "double_dotted": bool(getattr(duration, "isDoubleDotted", False)),
        "tuplet_enters": enters,
        "tuplet_times": times,
    }


def _duration_token(duration: dict[str, Any]) -> str:
    value = int(duration["value"])
    token = DURATION_NAMES.get(value, f"d{value}")
    if duration.get("double_dotted"):
        token += ".."
    elif duration.get("dotted"):
        token += "."
    enters = int(duration.get("tuplet_enters", 1))
    times = int(duration.get("tuplet_times", 1))
    if enters != 1 or times != 1:
        token += f"[{enters}:{times}]"
    return token


def _bend_token(bend: Any) -> str | None:
    if bend is None:
        return None
    value = int(getattr(bend, "value", 0) or 0)
    if not value:
        points = list(getattr(bend, "points", []) or [])
        if points:
            value = max(int(getattr(point, "value", 0) or 0) for point in points) * 25
    bend_type = _enum_name(getattr(bend, "type", "bend"))
    return f"bend:{bend_type}:{value}"


def _harmonic_token(harmonic: Any) -> str | None:
    if harmonic is None:
        return None
    name = harmonic.__class__.__name__.replace("Harmonic", "").replace("Effect", "")
    kind = name.lower() or "natural"
    if kind == "tapped":
        fret = getattr(harmonic, "fret", None)
        if fret is not None:
            return f"harm:tapped:{int(fret)}"
    if kind == "artificial":
        pitch = getattr(harmonic, "pitch", None)
        octave = getattr(harmonic, "octave", None)
        if pitch is not None and octave is not None:
            return (
                f"harm:artificial:{int(pitch.just)}:"
                f"{int(pitch.accidental)}:{int(octave.value)}"
            )
    return "harm:" + kind


def encode_note(note: Any) -> dict[str, Any]:
    effect = getattr(note, "effect", None)
    note_type = _enum_name(getattr(note, "type", "normal"))
    note_effects: list[str] = []
    if note_type == "tie":
        note_effects.append("tie")
    if note_type == "dead":
        note_effects.append("dead")
    if effect is not None:
        boolean_effects = (
            (("vibrato",), "vib"),
            (("hammer",), "hammer"),
            (("ghostNote", "ghost"), "ghost"),
            (("palmMute", "palm_mute"), "pm"),
            (("staccato",), "stacc"),
            (("letRing", "let_ring"), "let"),
            (("leftHandTapped",), "tap"),
            (("accentuatedNote",), "accent"),
            (("heavyAccentuatedNote",), "heavy"),
        )
        for names, token in boolean_effects:
            if _truthy_effect(effect, *names):
                note_effects.append(token)
        for slide in list(getattr(effect, "slides", []) or []):
            slide_name = _enum_name(slide)
            note_effects.append(SLIDE_NAMES.get(slide_name, "slide:" + slide_name))
        bend = _bend_token(getattr(effect, "bend", None))
        if bend:
            note_effects.append(bend)
        harmonic = _harmonic_token(getattr(effect, "harmonic", None))
        if harmonic:
            note_effects.append(harmonic)
        if getattr(effect, "grace", None) is not None:
            note_effects.append("grace")
        if getattr(effect, "trill", None) is not None:
            note_effects.append("trill")
        if getattr(effect, "tremoloPicking", None) is not None:
            note_effects.append("trem")
    fret: int | str = int(getattr(note, "value", 0) or 0)
    if note_type == "dead":
        fret = "x"
    velocity = int(getattr(note, "velocity", 95) or 95)
    # Some legacy GP3/GP4 files use -1 as an unspecified-dynamic sentinel.
    # It is not a playable MIDI velocity and is not a distinct printed symbol.
    if not 0 <= velocity <= 127:
        velocity = 95
    return {
        "string": int(getattr(note, "string", 1) or 1),
        "fret": fret,
        "pitch": int(getattr(note, "realValue", 0) or 0),
        "velocity": velocity,
        "swap_accidentals": bool(getattr(note, "swapAccidentals", False)),
        "effects": list(dict.fromkeys(note_effects)),
    }


def _beat_effects(beat: Any) -> list[str]:
    effect = getattr(beat, "effect", None)
    if effect is None:
        return []
    values: list[str] = []
    pick = _enum_name(getattr(effect, "pickStroke", "none"))
    if pick == "up":
        values.append("pick_up")
    elif pick == "down":
        values.append("pick_down")
    stroke = getattr(effect, "stroke", None)
    stroke_direction = _enum_name(getattr(stroke, "direction", "none"))
    if stroke_direction == "up":
        values.append("stroke_up")
    elif stroke_direction == "down":
        values.append("stroke_down")
    slap = _enum_name(getattr(effect, "slapEffect", "none"))
    if slap not in {"none", "0"}:
        values.append("slap:" + slap)
    if bool(getattr(effect, "fadeIn", False)):
        values.append("fade")
    if bool(getattr(effect, "hasRasgueado", False)):
        values.append("rasg")
    chord = getattr(effect, "chord", None)
    chord_name = str(getattr(chord, "name", "") or "").strip()
    if chord is not None and chord_name:
        values.append("chord:" + quote(chord_name, safe=""))
    mix = getattr(effect, "mixTableChange", None)
    tempo = getattr(mix, "tempo", None)
    tempo_value = int(getattr(tempo, "value", 0) or 0)
    if tempo_value and not bool(getattr(mix, "hideTempo", False)):
        values.append(f"tempo:{tempo_value}")
    text = getattr(beat, "text", None)
    text_value = str(getattr(text, "value", text) or "").strip()
    if text_value:
        values.append("text:" + quote(text_value, safe=""))
    return values


def encode_measure(
    measure: Any,
    previous_time_signature: str | None,
    previous_key_signature: str | None,
    *,
    initial_tempo: int | None = None,
) -> dict[str, Any]:
    header = measure.header
    numerator = int(header.timeSignature.numerator)
    denominator = int(header.timeSignature.denominator.value)
    time_signature = f"{numerator}/{denominator}"
    key_signature = _enum_name(getattr(header, "keySignature", "CMajor"))
    triplet_feel = _enum_name(getattr(header, "tripletFeel", "none"))
    measure_start = int(getattr(measure, "start", getattr(header, "start", 0)) or 0)
    voices = []
    for voice_index, voice in enumerate(measure.voices):
        events = []
        for beat in voice.beats:
            status = _enum_name(getattr(beat, "status", "normal"))
            event = {
                "start": int(getattr(beat, "start", measure_start) or measure_start) - measure_start,
                "duration": _duration_dict(beat.duration),
                "status": status,
                "notes": [encode_note(note) for note in beat.notes],
                "effects": _beat_effects(beat),
            }
            events.append(event)
        # GP8 can render an empty primary voice as a measure rest. Secondary
        # voices containing only placeholder empty beats are not visible and
        # must not become impossible OCR targets.
        if events and (voice_index == 0 or any(event["status"] != "empty" for event in events)):
            voices.append({"voice": voice_index, "events": events})
    bars = []
    if bool(getattr(header, "isRepeatOpen", False)):
        bars.append("repeat_open")
    repeat_close = int(getattr(header, "repeatClose", 0) or 0)
    if repeat_close > 0:
        bars.append("repeat_close")
    if bool(getattr(header, "hasDoubleBar", False)):
        bars.append("double")
    visible_initial_tempo = initial_tempo
    if initial_tempo is not None:
        promoted_tempo = None
        for voice in voices:
            for event in voice["events"]:
                if int(event["start"]) != 0:
                    continue
                for effect in event.get("effects", []):
                    if str(effect).startswith("tempo:"):
                        promoted_tempo = int(str(effect).partition(":")[2])
                        break
                if promoted_tempo is not None:
                    break
            if promoted_tempo is not None:
                break
        if promoted_tempo is not None:
            visible_initial_tempo = promoted_tempo
            for voice in voices:
                for event in voice["events"]:
                    if int(event["start"]) == 0:
                        event["effects"] = [
                            effect for effect in event.get("effects", [])
                            if not str(effect).startswith("tempo:")
                        ]
    return {
        "index": int(measure.number) - 1,
        "number": int(measure.number),
        "time_signature": time_signature,
        "print_time_signature": previous_time_signature is None or time_signature != previous_time_signature,
        "key_signature": key_signature,
        "print_key_signature": previous_key_signature is None or key_signature != previous_key_signature,
        "tempo_quarter": visible_initial_tempo,
        "triplet_feel": triplet_feel if triplet_feel != "none" else None,
        "bars": bars,
        "repeat_count": repeat_close + 1 if repeat_close > 0 else None,
        "alternate_endings": int(getattr(header, "repeatAlternative", 0) or 0) or None,
        "section": str(getattr(getattr(header, "marker", None), "title", "") or "").strip() or None,
        "direction": _enum_name(getattr(header, "direction", None)),
        "from_direction": _enum_name(getattr(header, "fromDirection", None)),
        "voices": voices,
    }


def _note_token(note: dict[str, Any], mode: str) -> str:
    if mode == "notation":
        token = f"p{int(note['pitch'])}"
    elif mode == "both":
        token = f"s{int(note['string'])}f{note['fret']}p{int(note['pitch'])}"
    else:
        token = f"s{int(note['string'])}f{note['fret']}"
    effects = note.get("effects") or []
    velocity = int(note.get("velocity", 95))
    if velocity != 95:
        effects = [*effects, f"vel:{velocity}"]
    # Accidental spelling is visible on a notation staff, but a TAB-only
    # staff prints only the fret value. Do not leak hidden source spelling
    # state into TAB supervision.
    if mode != "tab" and note.get("swap_accidentals"):
        effects = [*effects, "accswap"]
    if effects:
        token += "(" + ",".join(str(effect) for effect in effects) + ")"
    return token


def format_measure_target(measure: dict[str, Any], mode: str) -> str:
    if mode not in {"tab", "notation", "both"}:
        raise ValueError(f"Unsupported display mode: {mode}")
    metadata = []
    if measure.get("print_time_signature"):
        metadata.append("time=" + str(measure["time_signature"]))
    if measure.get("tempo_quarter"):
        metadata.append("tempo=" + str(measure["tempo_quarter"]))
    if mode in {"notation", "both"} and measure.get("print_key_signature"):
        metadata.append("key=" + str(measure["key_signature"]))
    if measure.get("triplet_feel"):
        metadata.append("feel=" + str(measure["triplet_feel"]))
    if measure.get("bars"):
        metadata.append("bar=" + ",".join(str(item) for item in measure["bars"]))
    if measure.get("repeat_count"):
        metadata.append("rep=" + str(measure["repeat_count"]))
    if measure.get("alternate_endings"):
        metadata.append("alt=" + str(measure["alternate_endings"]))
    if measure.get("section"):
        metadata.append("section=" + quote(str(measure["section"]), safe=""))
    if str(measure.get("direction", "none")).lower() not in {"none", "null"}:
        metadata.append("dir=" + quote(str(measure["direction"]), safe=""))
    if str(measure.get("from_direction", "none")).lower() not in {"none", "null"}:
        metadata.append("from=" + quote(str(measure["from_direction"]), safe=""))
    voice_tokens = []
    for voice in measure.get("voices", []):
        events = []
        previous_payload: str | None = None
        for event in voice.get("events", []):
            duration = _duration_token(event["duration"])
            status = str(event.get("status", "normal"))
            if status == "empty":
                payload = "e"
            elif status == "rest":
                payload = "r"
            else:
                notes = list(event.get("notes", []))
                # The source string assignment is invisible in notation-only
                # output.  Canonical pitch order removes an impossible target
                # dependency and makes chord equality independent of the GP
                # parser's internal note ordering.
                if mode == "notation":
                    notes.sort(key=lambda note: (
                        -int(note["pitch"]),
                        tuple(sorted(str(effect) for effect in note.get("effects", []))),
                        int(note.get("velocity", 95)),
                        bool(note.get("swap_accidentals")),
                    ))
                else:
                    notes.sort(key=lambda note: (int(note["string"]), repr(note)))
                payload = ",".join(_note_token(note, mode) for note in notes) or "z"
            beat_effects = [
                value
                for value in (event.get("effects") or [])
                # GP note velocity is playback state, not a printed dynamic
                # mark. Official GP8 omits it in notation, TAB and score+TAB,
                # so it cannot be a supervised OCR target in any mode. Keep
                # parser/export support for legacy M2 strings, but never emit
                # hidden velocity state from newly generated labels.
                if not str(value).startswith("dyn:")
            ]
            if beat_effects:
                payload += "<" + ",".join(str(value) for value in beat_effects) + ">"
            expanded_payload = payload
            if previous_payload is not None and expanded_payload == previous_payload:
                payload = "^"
            events.append(f"@{int(event['start'])}:{duration}:{payload}")
            previous_payload = expanded_payload
        voice_tokens.append(f"V{int(voice['voice'])}" + "{" + " ".join(events) + "}")
    prefix = "M2"
    if metadata:
        prefix += " " + " ".join(metadata)
    return prefix + " | " + " || ".join(voice_tokens)


def format_previous_measure_context(target: str, mode: str) -> str:
    """Compact a previous M2 measure for autoregressive document recognition.

    Only the last event of each voice and printed continuity metadata are kept.
    This gives the decoder enough evidence for ties and voice continuity without
    doubling the sequence length with an entire preceding measure.
    """
    measure = parse_measure_target(target)
    context = {
        "time_signature": measure.get("time_signature"),
        "print_time_signature": bool(measure.get("print_time_signature")),
        "tempo_quarter": measure.get("tempo_quarter"),
        "key_signature": measure.get("key_signature"),
        "print_key_signature": bool(measure.get("print_key_signature")),
        "triplet_feel": measure.get("triplet_feel"),
        "bars": [],
        "repeat_count": None,
        "alternate_endings": None,
        "section": None,
        "direction": None,
        "from_direction": None,
        "voices": [],
    }
    for voice in measure.get("voices", []):
        events = voice.get("events") or []
        if not events:
            continue
        last = max(events, key=lambda event: int(event["start"]))
        context["voices"].append({"voice": int(voice["voice"]), "events": [last]})
    return format_measure_target(context, mode).replace("M2", "C2", 1)


_DURATION_PATTERN = re.compile(
    r"^(?P<base>w|h|q|e|s|t|f|d\d+)(?P<dots>\.\.|\.)?"
    r"(?:\[(?P<enters>\d+):(?P<times>\d+)\])?$"
)
_NOTE_PATTERN = re.compile(
    r"^(?:s(?P<string>\d+)f(?P<fret>x|-?\d+))?"
    r"(?:p(?P<pitch>-?\d+))?(?:\((?P<effects>.*)\))?$"
)


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    values = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character in "(<[":
            depth += 1
        elif character in ")>]":
            depth = max(0, depth - 1)
        elif character == delimiter and depth == 0:
            values.append(text[start:index])
            start = index + 1
    values.append(text[start:])
    return [value for value in values if value]


def parse_duration_token(token: str) -> dict[str, Any]:
    match = _DURATION_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"Invalid M2 duration token: {token!r}")
    base = match.group("base")
    reverse = {value: key for key, value in DURATION_NAMES.items()}
    value = int(base[1:]) if base.startswith("d") else reverse[base]
    dots = match.group("dots") or ""
    return {
        "value": value,
        "dotted": dots == ".",
        "double_dotted": dots == "..",
        "tuplet_enters": int(match.group("enters") or 1),
        "tuplet_times": int(match.group("times") or 1),
    }


def _parse_note_token(token: str) -> dict[str, Any]:
    match = _NOTE_PATTERN.fullmatch(token)
    if match is None or not (match.group("string") or match.group("pitch")):
        raise ValueError(f"Invalid M2 note token: {token!r}")
    effects = _split_top_level(match.group("effects") or "")
    velocity = 95
    swap_accidentals = False
    musical_effects = []
    for effect in effects:
        if effect.startswith("vel:"):
            velocity = int(effect.partition(":")[2])
        elif effect == "accswap":
            swap_accidentals = True
        else:
            musical_effects.append(effect)
    fret_text = match.group("fret")
    result: dict[str, Any] = {
        "effects": musical_effects,
        "velocity": velocity,
        "swap_accidentals": swap_accidentals,
    }
    if match.group("string"):
        result["string"] = int(match.group("string"))
        result["fret"] = "x" if fret_text == "x" else int(fret_text)
    if match.group("pitch"):
        result["pitch"] = int(match.group("pitch"))
    return result


def parse_measure_target(text: str) -> dict[str, Any]:
    """Parse one strict M2 target into the canonical measure dictionary."""

    prefix, separator, voice_text = text.strip().partition("|")
    if not separator or not prefix.strip().startswith("M2"):
        raise ValueError("M2 target must start with 'M2' and contain '|'")
    metadata: dict[str, str] = {}
    for token in prefix.strip().split()[1:]:
        key, equals, value = token.partition("=")
        if not equals:
            raise ValueError(f"Invalid M2 metadata token: {token!r}")
        metadata[key] = value
    measure: dict[str, Any] = {
        "time_signature": metadata.get("time"),
        "print_time_signature": "time" in metadata,
        "tempo_quarter": int(metadata["tempo"]) if "tempo" in metadata else None,
        "key_signature": metadata.get("key"),
        "print_key_signature": "key" in metadata,
        "triplet_feel": metadata.get("feel"),
        "bars": metadata.get("bar", "").split(",") if metadata.get("bar") else [],
        "repeat_count": int(metadata["rep"]) if "rep" in metadata else None,
        "alternate_endings": int(metadata["alt"]) if "alt" in metadata else None,
        "section": unquote(metadata["section"]) if "section" in metadata else None,
        "direction": unquote(metadata["dir"]) if "dir" in metadata else None,
        "from_direction": unquote(metadata["from"]) if "from" in metadata else None,
        "voices": [],
    }
    voice_parts = [part.strip() for part in voice_text.split("||") if part.strip()]
    for voice_part in voice_parts:
        match = re.fullmatch(r"V(?P<voice>\d+)\{(?P<events>.*)\}", voice_part)
        if match is None:
            raise ValueError(f"Invalid M2 voice: {voice_part!r}")
        events = []
        for event_text in match.group("events").split():
            if not event_text.startswith("@"):
                raise ValueError(f"Invalid M2 event: {event_text!r}")
            start_text, separator, remainder = event_text[1:].partition(":")
            if not separator:
                raise ValueError(f"Invalid M2 event start: {event_text!r}")
            bracket_end = remainder.find("]")
            duration_end = bracket_end + 1 if bracket_end >= 0 else remainder.find(":")
            if duration_end <= 0 or duration_end >= len(remainder) or remainder[duration_end] != ":":
                raise ValueError(f"Invalid M2 event duration: {event_text!r}")
            duration_text = remainder[:duration_end]
            payload = remainder[duration_end + 1:]
            if payload == "^":
                if not events:
                    raise ValueError("M2 repeated payload '^' cannot be the first event")
                previous = deepcopy(events[-1])
                previous["start"] = int(start_text)
                previous["duration"] = parse_duration_token(duration_text)
                events.append(previous)
                continue
            beat_effects = []
            if payload.endswith(">") and "<" in payload:
                payload, _opening, effect_text = payload.rpartition("<")
                beat_effects = _split_top_level(effect_text[:-1])
            if payload == "e":
                status = "empty"
                notes = []
            elif payload == "r":
                status = "rest"
                notes = []
            elif payload == "z":
                status = "normal"
                notes = []
            else:
                status = "normal"
                notes = [_parse_note_token(note) for note in _split_top_level(payload)]
            events.append({
                "start": int(start_text),
                "duration": parse_duration_token(duration_text),
                "status": status,
                "notes": notes,
                "effects": beat_effects,
            })
        measure["voices"].append({"voice": int(match.group("voice")), "events": events})
    return measure


def _track_rank(track: Any) -> tuple[int, int, int, int]:
    note_count = 0
    event_count = 0
    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                if _enum_name(getattr(beat, "status", "normal")) != "empty":
                    event_count += 1
                note_count += len(beat.notes)
    string_count = len(track.strings)
    guitar_bonus = 5000 if string_count == 6 else max(0, 3000 - abs(string_count - 6) * 700)
    playable_bonus = 0 if track.isPercussionTrack else 100000
    return playable_bonus + guitar_bonus + note_count * 3 + event_count, note_count, event_count, string_count


def select_target_track(song: Any) -> tuple[int, Any]:
    ranked = [(_track_rank(track), -index, index, track) for index, track in enumerate(song.tracks)]
    if not ranked:
        raise ValueError("Song has no tracks")
    _rank, _negative_index, index, track = max(ranked, key=lambda item: (item[0], item[1]))
    if track.isPercussionTrack:
        raise ValueError("Song has no non-percussion target track")
    return index, track


def song_sequence_payload(song: Any, track_index: int, source_path: Path, source_hash: str) -> dict[str, Any]:
    track = song.tracks[track_index]
    measures = []
    previous_signature = None
    previous_key_signature = None
    counters: Counter[str] = Counter()
    maximum_fret = 0
    note_count = 0
    event_count = 0
    multi_voice_measures = 0
    for measure in track.measures:
        encoded = encode_measure(
            measure,
            previous_signature,
            previous_key_signature,
            initial_tempo=int(getattr(song, "tempo", 120) or 120) if not measures else None,
        )
        previous_signature = encoded["time_signature"]
        previous_key_signature = encoded["key_signature"]
        if len(encoded["voices"]) > 1:
            multi_voice_measures += 1
        for voice in encoded["voices"]:
            event_count += len(voice["events"])
            for event in voice["events"]:
                notes = event.get("notes") or []
                if event.get("status") == "normal" and notes:
                    # MIDI velocity is not visible in GP8's printed output.
                    # Normalize it so source-only playback data cannot leak
                    # into the image-to-sequence target.
                    for note in notes:
                        note["velocity"] = 95
                duration = event["duration"]
                if duration["dotted"] or duration["double_dotted"]:
                    counters["dotted"] += 1
                if duration["tuplet_enters"] != 1 or duration["tuplet_times"] != 1:
                    counters["tuplet"] += 1
                if event["status"] == "rest":
                    counters["rest"] += 1
                if event["status"] == "empty":
                    counters["empty"] += 1
                for effect in event.get("effects", []):
                    counters[str(effect).split(":", 1)[0]] += 1
                for note in notes:
                    note_count += 1
                    if isinstance(note["fret"], int):
                        maximum_fret = max(maximum_fret, note["fret"])
                    for effect in note["effects"]:
                        counters[str(effect).split(":", 1)[0]] += 1
        encoded["targets"] = {
            mode: format_measure_target(encoded, mode)
            for mode in ("tab", "notation", "both")
        }
        measures.append(encoded)
    tuning = [int(string.value) for string in track.strings]
    diversity_tags = (
        "bend", "hammer", "slide", "sl", "ss", "sib", "sia", "sod", "sou",
        "pm", "tie", "dead", "dotted", "tuplet", "rest", "harm", "grace",
        "trill", "trem", "tap", "ghost", "accent", "heavy", "let", "stacc",
        "pick_up", "pick_down", "stroke_up", "stroke_down", "slap", "fade",
        "rasg", "chord", "tempo", "text", "dyn",
    )
    tags = sorted(key for key in diversity_tags if counters[key])
    if multi_voice_measures:
        tags.append("multi_voice")
    return {
        "schema_version": "2.0",
        "source_id": source_hash[:16],
        "sha256": source_hash,
        "source_path": str(source_path.resolve()),
        "source_format": source_path.suffix.lower().lstrip("."),
        "song": {
            "title": str(getattr(song, "title", "") or source_path.stem),
            "artist": str(getattr(song, "artist", "") or ""),
            "tempo_quarter": int(getattr(song, "tempo", 120) or 120),
        },
        "track": {
            "index": track_index,
            "number": int(getattr(track, "number", track_index + 1)),
            "name": str(getattr(track, "name", "") or ""),
            "string_count": len(track.strings),
            "tuning_midi_high_to_low": tuning,
            "capo": int(getattr(track, "offset", 0) or 0),
        },
        "statistics": {
            "measure_count": len(measures),
            "event_count": event_count,
            "note_count": note_count,
            "multi_voice_measure_count": multi_voice_measures,
            "maximum_fret": maximum_fret,
            "technique_counts": dict(sorted(counters.items())),
            "tags": tags,
        },
        "measures": measures,
    }


def parse_song(path: Path) -> tuple[Any, str]:
    import guitarpro

    last_error: Exception | None = None
    for encoding in ("utf-8", "gbk", "cp936", "cp1252", "latin-1"):
        try:
            song = guitarpro.parse(str(path), encoding=encoding)
            _repair_tied_note_values(song)
            return song, encoding
        except Exception as error:  # pragma: no cover - corpus dependent
            last_error = error
    raise RuntimeError(f"Could not parse {path}: {last_error!r}")


def _repair_tied_note_values(song: Any) -> None:
    """Resolve tie frets by identity-safe chronological traversal.

    PyGuitarPro's GP3/4/5 reader uses ``list.index(beat)`` while a beat is
    being parsed. Beats are attrs classes with structural equality, so two
    visually identical tie beats can compare equal and the reader can attach
    the later tie to an older fret. Guitar Pro itself uses the immediately
    preceding note on the same string. Recompute that state explicitly before
    any source label or round-trip metric reads ``note.realValue``.
    """

    for track in getattr(song, "tracks", []):
        previous_by_voice: dict[int, dict[int, int]] = {}
        for measure in getattr(track, "measures", []):
            for voice_index, voice in enumerate(getattr(measure, "voices", [])):
                previous = previous_by_voice.setdefault(voice_index, {})
                for beat in getattr(voice, "beats", []):
                    for note in getattr(beat, "notes", []):
                        note_type = _enum_name(getattr(note, "type", "normal"))
                        string = int(getattr(note, "string", 0) or 0)
                        if note_type == "tie" and string in previous:
                            note.value = previous[string]
                        value = int(getattr(note, "value", -1) or 0)
                        if note_type != "dead" and string > 0 and value >= 0:
                            previous[string] = value


def analyze_source(path: Path) -> dict[str, Any]:
    song, encoding = parse_song(path)
    track_index, _track = select_target_track(song)
    source_hash = sha256(path.read_bytes()).hexdigest()
    payload = song_sequence_payload(song, track_index, path, source_hash)
    payload["source_encoding"] = encoding
    return payload


def prepare_single_track_gp5(path: Path, output: Path, mode: str) -> dict[str, Any]:
    import guitarpro
    from guitarpro import models as gm

    song, encoding = parse_song(path)
    track_index, track = select_target_track(song)
    source_hash = sha256(path.read_bytes()).hexdigest()
    payload = song_sequence_payload(song, track_index, path, source_hash)
    song.tracks = [track]
    track.number = 1
    track.settings = gm.TrackSettings(
        tablature=mode in {"tab", "both"},
        notation=mode in {"notation", "both"},
        diagramsAreBelow=False,
        showRhythm=True,
        forceHorizontal=False,
        forceChannels=False,
        diagramList=True,
        diagramsInScore=True,
        autoLetRing=False,
        autoBrush=False,
        extendRhythmic=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    guitarpro.write(song, str(output), version=(5, 1, 0), encoding="utf-8")
    payload["source_encoding"] = encoding
    payload["prepared_gp5"] = str(output.resolve())
    return payload
