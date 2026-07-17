from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

from guitarocr.data.gp_measure_sequence import parse_measure_target


DEFAULT_TUNING = (64, 59, 55, 50, 45, 40)


def _duration(value: dict[str, Any], gm: Any) -> Any:
    duration = gm.Duration(value=int(value["value"]))
    duration.isDotted = bool(value.get("dotted"))
    duration.isDoubleDotted = bool(value.get("double_dotted"))
    enters = int(value.get("tuplet_enters", 1))
    times = int(value.get("tuplet_times", 1))
    duration.tuplet = gm.Tuplet(enters=enters, times=times)
    return duration


def _enum_member(enum_type: Any, name: str | None, default: Any) -> Any:
    if not name:
        return default
    normalized = str(name).replace(" ", "").lower()
    for key, value in vars(enum_type).items():
        if key.startswith("_"):
            continue
        value_name = str(getattr(value, "name", key)).replace(" ", "").lower()
        if key.replace(" ", "").lower() == normalized or value_name == normalized:
            return value
    return default


def _choose_position(pitch: int, tuning: list[int], used_strings: set[int]) -> tuple[int, int]:
    candidates = []
    for string, open_pitch in enumerate(tuning, start=1):
        fret = pitch - open_pitch
        if 0 <= fret <= 24:
            collision = 1 if string in used_strings else 0
            candidates.append((collision, abs(fret - 5), fret, string))
    if not candidates:
        return 1, max(0, min(24, pitch - tuning[0]))
    _collision, _position_cost, fret, string = min(candidates)
    return string, fret


def _assign_positions(
    notes: list[dict[str, Any]],
    tuning: list[int],
    previous: dict[int, tuple[int, int]],
    reservations: dict[int, int],
) -> list[tuple[int, int]]:
    """Assign a collision-free guitar position to a notation-only chord."""

    candidates: list[list[tuple[float, int, int]]] = []
    for note in notes:
        if "string" in note:
            string = int(note["string"])
            fret_value = note.get("fret", 0)
            if fret_value == "x" and "pitch" in note:
                fret = max(0, int(note["pitch"]) - tuning[string - 1])
            else:
                fret = 0 if fret_value == "x" else int(fret_value)
            candidates.append([(0.0, string, fret)])
            continue
        pitch = int(note["pitch"])
        tie = "tie" in (note.get("effects") or [])
        values = []
        for string, open_pitch in enumerate(tuning, start=1):
            fret = pitch - open_pitch
            if not 0 <= fret <= 36:
                continue
            previous_pitch, previous_fret = previous.get(string, (-10_000, -1))
            tie_match = tie and previous_pitch == pitch
            reservation_owner = reservations.get(string)
            reserved_match = tie and reservation_owner == pitch
            cost = abs(fret - 5) + (0 if not tie or tie_match else 1_000)
            if reservation_owner is not None and (
                reservation_owner != pitch or not tie
            ):
                cost += 2_000
            if tie and reservations and not reserved_match:
                cost += 2_000
            if tie_match:
                cost -= 1_000
                fret = previous_fret
            if reserved_match:
                cost -= 4_000
                if previous_pitch == pitch:
                    fret = previous_fret
            values.append((float(cost), string, fret))
        if not values:
            string, fret = _choose_position(pitch, tuning, set())
            values = [(10_000.0, string, fret)]
        candidates.append(sorted(values))

    order = sorted(range(len(notes)), key=lambda index: (len(candidates[index]), index))
    best: tuple[float, list[tuple[int, int]]] | None = None
    assigned: list[tuple[int, int] | None] = [None] * len(notes)

    def search(position: int, used_strings: set[int], cost: float) -> None:
        nonlocal best
        if best is not None and cost >= best[0]:
            return
        if position == len(order):
            best = (cost, [value for value in assigned if value is not None])
            return
        note_index = order[position]
        for candidate_cost, string, fret in candidates[note_index]:
            if string in used_strings:
                continue
            assigned[note_index] = (string, fret)
            search(position + 1, used_strings | {string}, cost + candidate_cost)
            assigned[note_index] = None

    search(0, set(), 0.0)
    if best is not None:
        return best[1]

    # Malformed source chords can exceed the string count.  Preserve every
    # note deterministically even though GP5 cannot make such a chord fully
    # playable, instead of silently dropping a note.
    used: set[int] = set()
    fallback = []
    for values in candidates:
        value = next((item for item in values if item[1] not in used), values[0])
        used.add(value[1])
        fallback.append((value[1], value[2]))
    return fallback


def _tie_reservation_note_ids(
    measures: list[dict[str, Any]],
) -> dict[int, set[int]]:
    """Mark the prior note that each notation-only tie must continue."""

    result: dict[int, set[int]] = {0: set(), 1: set()}
    # A chord may contain the same written pitch on multiple strings. Keep a
    # small stack of live anchors per pitch instead of collapsing them to the
    # last note; otherwise the second tied unison is assigned to an unrelated
    # string and PyGuitarPro resolves it to the wrong pitch on readback.
    previous_by_voice: dict[int, dict[int, list[int]]] = {0: {}, 1: {}}
    for measure in measures:
        for voice in measure.get("voices", []):
            voice_index = int(voice["voice"])
            previous_by_pitch = previous_by_voice.setdefault(voice_index, {})
            for event in voice.get("events", []):
                grouped: dict[int, list[dict[str, Any]]] = {}
                for note in event.get("notes", []):
                    if "pitch" in note:
                        grouped.setdefault(int(note["pitch"]), []).append(note)
                for pitch, notes in grouped.items():
                    anchors = list(previous_by_pitch.get(pitch, []))
                    tie_count = sum(
                        "tie" in (note.get("effects") or []) for note in notes
                    )
                    if tie_count:
                        result.setdefault(voice_index, set()).update(
                            anchors[-tie_count:]
                        )
                    # Ties replace the matched anchors on their strings while
                    # older unmatched unisons may still be sounding. Cap at
                    # the physical string count to keep malformed files sane.
                    retained = anchors[:-tie_count] if tie_count else anchors
                    previous_by_pitch[pitch] = (
                        retained + [id(note) for note in notes]
                    )[-6:]
    return result


def _plan_notation_voice_positions(
    measures: list[dict[str, Any]],
    voice_index: int,
    tuning: list[int],
    *,
    beam_width: int = 256,
) -> dict[int, tuple[int, int]]:
    """Globally assign hidden strings while preserving every visible tie."""

    events: list[list[dict[str, Any]]] = []
    for measure in measures:
        voice = next(
            (
                value for value in measure.get("voices", [])
                if int(value["voice"]) == voice_index
            ),
            None,
        )
        if voice is not None:
            events.extend(list(event.get("notes", [])) for event in voice["events"])
    if not events:
        return {}
    if not any(
        "tie" in (note.get("effects") or [])
        for notes in events for note in notes
    ):
        return {}

    # At each event boundary, retain enough copies of a pitch when its next
    # occurrence is a tie.  If the next occurrence is a normal note, that
    # note can establish a new anchor and no reservation is required yet.
    required_after: list[Counter[int]] = [Counter() for _ in events]
    next_requirement: Counter[int] = Counter()
    for event_index in range(len(events) - 1, -1, -1):
        required_after[event_index] = next_requirement.copy()
        grouped: dict[int, list[dict[str, Any]]] = {}
        for note in events[event_index]:
            if "pitch" in note:
                grouped.setdefault(int(note["pitch"]), []).append(note)
        for pitch, notes in grouped.items():
            tie_count = sum(
                "tie" in (note.get("effects") or []) for note in notes
            )
            next_requirement[pitch] = tie_count if tie_count == len(notes) else 0
            if not next_requirement[pitch]:
                del next_requirement[pitch]

    # Each retained node stores only a back-pointer and this event's chosen
    # positions; paths are reconstructed once after the final event.
    nodes: list[tuple[int, list[tuple[int, tuple[int, int]]]]] = [(-1, [])]
    initial_state: tuple[int | None, ...] = tuple(None for _ in tuning)
    beams: dict[tuple[int | None, ...], tuple[float, int]] = {
        initial_state: (0.0, 0)
    }

    for event_index, notes in enumerate(events):
        candidates_by_state: dict[
            tuple[int | None, ...],
            tuple[float, int, list[tuple[int, tuple[int, int]]]],
        ] = {}
        for state, (base_cost, parent_node) in beams.items():
            note_candidates: list[list[tuple[float, int, int]]] = []
            valid = True
            for note in notes:
                pitch = int(note["pitch"])
                tie = "tie" in (note.get("effects") or [])
                values = []
                for string, open_pitch in enumerate(tuning, start=1):
                    fret = pitch - open_pitch
                    if not 0 <= fret <= 36:
                        continue
                    if tie and state[string - 1] != pitch:
                        continue
                    values.append((abs(fret - 5) * 0.01, string, fret))
                if not values:
                    valid = False
                    break
                note_candidates.append(values)
            if not valid:
                continue

            order = sorted(
                range(len(notes)), key=lambda index: (len(note_candidates[index]), index)
            )
            positions: list[tuple[int, int] | None] = [None] * len(notes)

            def enumerate_assignments(
                order_index: int, used: set[int], local_cost: float
            ) -> None:
                if order_index == len(order):
                    new_state = list(state)
                    chosen = []
                    for note, position in zip(notes, positions):
                        assert position is not None
                        string, fret = position
                        pitch = int(note["pitch"])
                        new_state[string - 1] = pitch
                        chosen.append((id(note), (string, fret)))
                    # Only pitches whose next occurrence is a tie affect a
                    # future decision. Keeping every ordinary pitch in the
                    # beam creates hundreds of musically equivalent states
                    # and makes dense notation-only songs needlessly slow.
                    required_pitches = set(required_after[event_index])
                    new_state = [
                        value if value in required_pitches else None
                        for value in new_state
                    ]
                    state_tuple = tuple(new_state)
                    counts = Counter(value for value in state_tuple if value is not None)
                    if any(
                        counts[pitch] < count
                        for pitch, count in required_after[event_index].items()
                    ):
                        return
                    total_cost = base_cost + local_cost
                    previous = candidates_by_state.get(state_tuple)
                    if previous is None or total_cost < previous[0]:
                        candidates_by_state[state_tuple] = (
                            total_cost, parent_node, chosen
                        )
                    return
                note_index = order[order_index]
                for cost, string, fret in note_candidates[note_index]:
                    if string in used:
                        continue
                    positions[note_index] = (string, fret)
                    enumerate_assignments(
                        order_index + 1, used | {string}, local_cost + cost
                    )
                    positions[note_index] = None

            enumerate_assignments(0, set(), 0.0)

        if not candidates_by_state:
            return {}
        ranked = sorted(
            candidates_by_state.items(), key=lambda item: (item[1][0], repr(item[0]))
        )[:beam_width]
        beams = {}
        for state, (cost, parent_node, chosen) in ranked:
            nodes.append((parent_node, chosen))
            beams[state] = (cost, len(nodes) - 1)

    _cost, node_index = min(beams.values(), key=lambda value: value[0])
    plan: dict[int, tuple[int, int]] = {}
    while node_index > 0:
        parent_node, chosen = nodes[node_index]
        plan.update(chosen)
        node_index = parent_node
    return plan


def _bend(effect_text: str, gm: Any) -> Any:
    _prefix, _separator, tail = effect_text.partition(":")
    kind_text, _separator, value_text = tail.partition(":")
    value = int(value_text or 100)
    kind = _enum_member(gm.BendType, kind_text, gm.BendType.bend)
    point_value = max(0, round(value / 25))
    points = [
        gm.BendPoint(position=0, value=0),
        gm.BendPoint(position=6, value=point_value),
        gm.BendPoint(position=12, value=point_value),
    ]
    return gm.BendEffect(type=kind, value=value, points=points)


def _apply_note_effects(note: Any, effects: list[str], gm: Any) -> None:
    slide_map = {
        "sl": "legatoSlideTo",
        "ss": "shiftSlideTo",
        "sib": "intoFromBelow",
        "sia": "intoFromAbove",
        "sod": "outDownwards",
        "sou": "outUpwards",
    }
    for effect in effects:
        if effect == "tie":
            note.type = gm.NoteType.tie
        elif effect == "dead":
            note.type = gm.NoteType.dead
        elif effect == "vib":
            note.effect.vibrato = True
        elif effect == "hammer":
            note.effect.hammer = True
        elif effect == "ghost":
            note.effect.ghostNote = True
        elif effect == "pm":
            note.effect.palmMute = True
        elif effect == "stacc":
            note.effect.staccato = True
        elif effect == "let":
            note.effect.letRing = True
        elif effect == "tap":
            # Legacy M2 used a note-level alias. GP3-5 stores tapping as a
            # beat slap-effect, which canonical v2 serializes as
            # ``slap:tapping``.
            note.beat.effect.slapEffect = gm.SlapEffect.tapping
        elif effect == "accent":
            note.effect.accentuatedNote = True
        elif effect == "heavy":
            note.effect.heavyAccentuatedNote = True
        elif effect in slide_map:
            slide = _enum_member(gm.SlideType, slide_map[effect], None)
            if slide is not None:
                note.effect.slides.append(slide)
        elif effect.startswith("slide:"):
            slide = _enum_member(gm.SlideType, effect.partition(":")[2], None)
            if slide is not None:
                note.effect.slides.append(slide)
        elif effect.startswith("bend:"):
            note.effect.bend = _bend(effect, gm)
        elif effect == "harm:natural":
            note.effect.harmonic = gm.NaturalHarmonic()
        elif effect.startswith("harm:"):
            harmonic_parts = effect.split(":")
            harmonic_name = harmonic_parts[1]
            harmonic_types = {
                "artificial": gm.ArtificialHarmonic,
                "pinch": gm.PinchHarmonic,
                "tapped": gm.TappedHarmonic,
                "semi": gm.SemiHarmonic,
            }
            harmonic_type = harmonic_types.get(harmonic_name)
            if harmonic_type is not None:
                if harmonic_name == "tapped":
                    # GP5 requires this byte. Older M2 used only
                    # ``harm:tapped``; a musically conventional +12 fret
                    # fallback keeps those targets writable.
                    harmonic_fret = (
                        int(harmonic_parts[2])
                        if len(harmonic_parts) >= 3
                        else min(255, max(0, int(note.value) + 12))
                    )
                    note.effect.harmonic = gm.TappedHarmonic(fret=harmonic_fret)
                elif harmonic_name == "artificial" and len(harmonic_parts) >= 5:
                    octave_value = int(harmonic_parts[4])
                    octave = next(
                        (
                            value for key, value in vars(gm.Octave).items()
                            if not key.startswith("_")
                            and getattr(value, "value", None) == octave_value
                        ),
                        gm.Octave.ottava,
                    )
                    note.effect.harmonic = gm.ArtificialHarmonic(
                        pitch=gm.PitchClass(
                            int(harmonic_parts[2]), int(harmonic_parts[3])
                        ),
                        octave=octave,
                    )
                else:
                    note.effect.harmonic = harmonic_type()
        elif effect == "grace":
            note.effect.grace = gm.GraceEffect(fret=max(0, int(note.value)))
        elif effect == "trill":
            note.effect.trill = gm.TrillEffect(
                fret=max(0, int(note.value) + 1), duration=gm.Duration(value=16)
            )
        elif effect == "trem":
            note.effect.tremoloPicking = gm.TremoloPickingEffect(
                duration=gm.Duration(value=16)
            )


def _note(
    note_data: dict[str, Any], beat: Any, tuning: list[int],
    position: tuple[int, int], gm: Any, default_velocity: int = 95,
) -> Any:
    string, fret = position
    note = gm.Note(
        beat=beat,
        string=string,
        value=fret,
        velocity=(
            int(default_velocity)
            if int(note_data.get("velocity", 95)) == 95
            else int(note_data["velocity"])
        ),
        swapAccidentals=bool(note_data.get("swap_accidentals")),
        type=gm.NoteType.normal,
    )
    effects = list(note_data.get("effects") or [])
    if note_data.get("fret") == "x" and "dead" not in effects:
        effects.append("dead")
    _apply_note_effects(note, effects, gm)
    return note


def _apply_beat_effects(beat: Any, effects: list[str], string_count: int, gm: Any) -> None:
    for effect in effects:
        if effect == "pick_up":
            beat.effect.pickStroke = gm.BeatStrokeDirection.up
        elif effect == "pick_down":
            beat.effect.pickStroke = gm.BeatStrokeDirection.down
        elif effect == "stroke_up":
            beat.effect.stroke = gm.BeatStroke(direction=gm.BeatStrokeDirection.up, value=8)
        elif effect == "stroke_down":
            beat.effect.stroke = gm.BeatStroke(direction=gm.BeatStrokeDirection.down, value=8)
        elif effect.startswith("slap:"):
            beat.effect.slapEffect = _enum_member(
                gm.SlapEffect, effect.partition(":")[2], gm.SlapEffect.none
            )
        elif effect == "fade":
            beat.effect.fadeIn = True
        elif effect == "rasg":
            beat.effect.hasRasgueado = True
        elif effect.startswith("chord:"):
            name = unquote(effect.partition(":")[2])
            beat.effect.chord = gm.Chord(
                length=string_count,
                name=name,
                firstFret=1,
                strings=[-1] * string_count,
                omissions=[False] * 7,
                show=True,
                newFormat=True,
            )
        elif effect.startswith("tempo:"):
            tempo = int(effect.partition(":")[2])
            beat.effect.mixTableChange = gm.MixTableChange(
                tempo=gm.MixTableItem(value=tempo, duration=0), hideTempo=False
            )
        elif effect.startswith("text:"):
            beat.text = unquote(effect.partition(":")[2])


def _display_settings(mode: str, gm: Any) -> Any:
    return gm.TrackSettings(
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


def targets_to_song(
    targets: Iterable[str],
    *,
    mode: str = "both",
    title: str = "Guitar OCR",
    artist: str = "",
    tuning: Iterable[int] = DEFAULT_TUNING,
    capo: int = 0,
) -> Any:
    import guitarpro
    from guitarpro import models as gm

    del guitarpro  # imported here to keep corpus-only utilities lightweight
    if mode not in {"tab", "notation", "both"}:
        raise ValueError(f"Unsupported display mode: {mode}")
    tuning_values = [int(value) for value in tuning]
    measures = [parse_measure_target(target) for target in targets if target.strip()]
    if not measures:
        raise ValueError("At least one M2 measure target is required")

    song = gm.Song(title=title, artist=artist)
    song.measureHeaders = []
    song.tracks = []
    first_tempo = next(
        (int(measure["tempo_quarter"]) for measure in measures if measure.get("tempo_quarter")),
        120,
    )
    song.tempo = first_tempo
    track = gm.Track(song, number=1, name="Guitar OCR", measures=[], strings=[])
    track.offset = int(capo)
    track.settings = _display_settings(mode, gm)
    track.channel.instrument = 25
    track.strings = [
        gm.GuitarString(number=index, value=value)
        for index, value in enumerate(tuning_values, start=1)
    ]
    song.tracks.append(track)

    current_time = "4/4"
    current_key = gm.KeySignature.CMajor
    start = gm.Duration.quarterTime
    previous_positions: dict[int, dict[int, tuple[int, int]]] = {0: {}, 1: {}}
    reserved_strings: dict[int, dict[int, int]] = {0: {}, 1: {}}
    current_velocity: dict[int, int] = {0: 95, 1: 95}
    reservation_note_ids = _tie_reservation_note_ids(measures)
    notation_position_plans = (
        {
            voice_index: _plan_notation_voice_positions(
                measures, voice_index, tuning_values
            )
            for voice_index in range(2)
        }
        if mode == "notation" else {}
    )
    for index, measure_data in enumerate(measures, start=1):
        if measure_data.get("time_signature"):
            current_time = str(measure_data["time_signature"])
        numerator_text, _slash, denominator_text = current_time.partition("/")
        numerator = int(numerator_text or 4)
        denominator = int(denominator_text or 4)
        if measure_data.get("key_signature"):
            current_key = _enum_member(
                gm.KeySignature, str(measure_data["key_signature"]), current_key
            )
        header = gm.MeasureHeader(number=index, start=start)
        header.timeSignature = gm.TimeSignature(
            numerator=numerator, denominator=gm.Duration(value=denominator)
        )
        header.keySignature = current_key
        header.hasDoubleBar = "double" in measure_data["bars"]
        header.isRepeatOpen = "repeat_open" in measure_data["bars"]
        if "repeat_close" in measure_data["bars"]:
            header.repeatClose = max(1, int(measure_data.get("repeat_count") or 2) - 1)
        header.repeatAlternative = int(measure_data.get("alternate_endings") or 0)
        if measure_data.get("section"):
            header.marker = gm.Marker(title=str(measure_data["section"]))
        header.tripletFeel = _enum_member(
            gm.TripletFeel, measure_data.get("triplet_feel"), gm.TripletFeel.none
        )
        header.direction = (
            gm.DirectionSign(str(measure_data["direction"]))
            if measure_data.get("direction") else None
        )
        header.fromDirection = (
            gm.DirectionSign(str(measure_data["from_direction"]))
            if measure_data.get("from_direction") else None
        )
        song.measureHeaders.append(header)

        measure = gm.Measure(track, header, voices=[])
        voices_by_index = {
            int(voice["voice"]): voice for voice in measure_data.get("voices", [])
        }
        for voice_index in range(2):
            voice = gm.Voice(measure, beats=[])
            voice_data = voices_by_index.get(voice_index)
            if voice_data is not None:
                for event in voice_data["events"]:
                    for effect in event.get("effects") or []:
                        if str(effect).startswith("dyn:"):
                            current_velocity[voice_index] = int(
                                str(effect).partition(":")[2]
                            )
                    status = str(event.get("status", "normal"))
                    beat = gm.Beat(
                        voice=voice,
                        duration=_duration(event["duration"], gm),
                        start=start + int(event.get("start", 0)),
                        status={
                            "empty": gm.BeatStatus.empty,
                            "rest": gm.BeatStatus.rest,
                        }.get(status, gm.BeatStatus.normal),
                    )
                    note_values = list(event.get("notes", []))
                    planned_positions = [
                        notation_position_plans.get(voice_index, {}).get(id(note))
                        for note in note_values
                    ]
                    if all(position is not None for position in planned_positions):
                        positions = [
                            position for position in planned_positions
                            if position is not None
                        ]
                    else:
                        positions = _assign_positions(
                            note_values,
                            tuning_values,
                            previous_positions[voice_index],
                            reserved_strings[voice_index],
                        )
                    beat.notes = [
                        _note(
                            note_data,
                            beat,
                            tuning_values,
                            position,
                            gm,
                            current_velocity[voice_index],
                        )
                        for note_data, position in zip(note_values, positions)
                    ]
                    for note_data, note, position in zip(
                        note_values, beat.notes, positions
                    ):
                        if "dead" in (note_data.get("effects") or []):
                            continue
                        string, fret = position
                        pitch = int(
                            note_data.get("pitch", tuning_values[string - 1] + fret)
                        )
                        previous_positions[voice_index][string] = (pitch, fret)
                        is_tie = "tie" in (note_data.get("effects") or [])
                        continues = id(note_data) in reservation_note_ids.get(
                            voice_index, set()
                        )
                        if is_tie and not continues:
                            reserved_strings[voice_index].pop(string, None)
                        if continues:
                            reserved_strings[voice_index][string] = pitch
                    _apply_beat_effects(
                        beat, list(event.get("effects") or []), len(tuning_values), gm
                    )
                    voice.beats.append(beat)
            measure.voices.append(voice)
        track.measures.append(measure)
        start += numerator * gm.Duration.quarterTime * 4 // denominator

    song.key = song.measureHeaders[0].keySignature
    return song


def write_targets_gp5(
    targets: Iterable[str],
    output: str | Path,
    *,
    mode: str = "both",
    title: str = "Guitar OCR",
    artist: str = "",
    tuning: Iterable[int] = DEFAULT_TUNING,
    capo: int = 0,
) -> Path:
    import guitarpro

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    song = targets_to_song(
        targets, mode=mode, title=title, artist=artist, tuning=tuning, capo=capo
    )
    guitarpro.write(song, str(output_path), version=(5, 1, 0), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one-M2-measure-per-line text to GP5.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("tab", "notation", "both"), default="both")
    parser.add_argument("--title", default="Guitar OCR")
    parser.add_argument("--artist", default="")
    parser.add_argument("--tuning", default=",".join(str(value) for value in DEFAULT_TUNING))
    parser.add_argument("--capo", type=int, default=0)
    args = parser.parse_args()
    targets = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines()]
    tuning = [int(value) for value in args.tuning.split(",") if value.strip()]
    write_targets_gp5(
        targets,
        args.output,
        mode=args.mode,
        title=args.title,
        artist=args.artist,
        tuning=tuning,
        capo=args.capo,
    )


if __name__ == "__main__":
    main()
