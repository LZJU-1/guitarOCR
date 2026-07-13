from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


NOTE_VALUES = {
    "Whole": 1,
    "Half": 2,
    "Quarter": 4,
    "Eighth": 8,
    "16th": 16,
    "32nd": 32,
    "64th": 64,
    "128th": 128,
}


def _index(parent: ET.Element | None) -> dict[str, ET.Element]:
    if parent is None:
        return {}
    return {item.get("id", ""): item for item in parent}


def _refs(element: ET.Element | None, tag: str) -> list[str]:
    text = element.findtext(tag, "") if element is not None else ""
    return [value for value in text.split() if value != "-1"]


def _property(element: ET.Element, name: str) -> ET.Element | None:
    return element.find(f"./Properties/Property[@name='{name}']")


def _property_text(element: ET.Element, name: str, child: str) -> str | None:
    value = _property(element, name)
    return value.findtext(child) if value is not None else None


def _enabled_property(element: ET.Element, name: str) -> bool:
    value = _property(element, name)
    return value is not None and value.find("Enable") is not None


def _rhythm(rhythm: ET.Element) -> tuple[int, str, str, Fraction]:
    label = rhythm.findtext("NoteValue", "Quarter")
    if label not in NOTE_VALUES:
        raise ValueError(f"Unsupported GPIF rhythm value: {label}")
    value = NOTE_VALUES[label]
    dot_element = rhythm.find("AugmentationDot")
    dot_count = int(dot_element.get("count", "0")) if dot_element is not None else 0
    dot = "none" if dot_count == 0 else "single" if dot_count == 1 else "double"
    dot_factor = Fraction(1) if dot_count == 0 else Fraction(3, 2) if dot_count == 1 else Fraction(7, 4)

    enters = times = 1
    tuplet = rhythm.find("PrimaryTuplet")
    if tuplet is not None:
        enters = int(tuplet.get("num", tuplet.get("enters", "1")))
        times = int(tuplet.get("den", tuplet.get("times", "1")))
    else:
        tuplet = rhythm.find("Tuplet")
        if tuplet is not None:
            enters = int(tuplet.get("num", tuplet.get("enters", "1")))
            times = int(tuplet.get("den", tuplet.get("times", "1")))
    division = f"{enters}:{times}"
    duration = Fraction(1, value) * dot_factor * Fraction(times, enters)
    return value, dot, division, duration


def _note(note: ET.Element, string_count: int) -> dict:
    string_value = _property_text(note, "String", "String")
    fret_value = _property_text(note, "Fret", "Fret")
    midi_value = _property_text(note, "Midi", "Number")
    if string_value is None:
        raise ValueError(f"GPIF note {note.get('id')} has no string")
    gpif_string = int(string_value)
    tied = note.find("Tie")
    muted = _enabled_property(note, "Muted")
    slide = _property_text(note, "Slide", "Flags")
    bend = _property(note, "Bend")
    effects = {
        "muted": muted,
        "palm_mute": _enabled_property(note, "PalmMuted"),
        "slide": slide is not None and int(slide) != 0,
        "slide_flags": int(slide) if slide is not None else 0,
        "bend": bend is not None,
        "hammer": _enabled_property(note, "HopoOrigin") or _enabled_property(note, "HopoDestination"),
        "vibrato": note.find("Vibrato") is not None,
        "harmonic": _property(note, "HarmonicType") is not None,
        "ghost": note.find("AntiAccent") is not None,
        "accent": note.find("Accent") is not None,
    }
    fret: int | None = int(fret_value) if fret_value is not None else None
    return {
        "string": string_count - gpif_string,
        "fret": fret,
        "printed_fret": "X" if muted else fret,
        "midi": int(midi_value) if midi_value is not None else None,
        "tie_in": tied is not None and tied.get("destination") == "true",
        "tie_out": tied is not None and tied.get("origin") == "true",
        "effects": effects,
    }


def _tempo(root: ET.Element) -> int | None:
    for automation in root.findall("./MasterTrack/Automations/Automation"):
        if automation.findtext("Type") == "Tempo":
            raw = automation.findtext("Value", "").split()
            if raw:
                return round(float(raw[0]))
    return None


def load_gpif_score(path: Path, track_index: int = 0) -> dict:
    """Load the exact first-voice score semantics from a Guitar Pro 7/8 .gp archive."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        try:
            data = archive.read("Content/score.gpif")
        except KeyError as error:
            raise ValueError(f"{path} is not a Guitar Pro archive containing Content/score.gpif") from error
    root = ET.fromstring(data)
    tracks = list(root.find("Tracks") or [])
    if not 0 <= track_index < len(tracks):
        raise IndexError(f"Track index {track_index} is outside the {len(tracks)} GPIF tracks")
    track = tracks[track_index]
    pitches = track.findtext(".//Staves/Staff/Properties/Property[@name='Tuning']/Pitches", "")
    tuning_low_to_high = [int(value) for value in pitches.split()]
    if not tuning_low_to_high:
        raise ValueError(f"Track {track_index} has no GPIF tuning")
    tuning = list(reversed(tuning_low_to_high))
    string_count = len(tuning)

    bars = _index(root.find("Bars"))
    voices = _index(root.find("Voices"))
    beats = _index(root.find("Beats"))
    notes = _index(root.find("Notes"))
    rhythms = _index(root.find("Rhythms"))
    measures: list[dict] = []
    carried_time = [4, 4]
    for measure_index, master_bar in enumerate(list(root.find("MasterBars") or [])):
        raw_time = master_bar.findtext("Time")
        if raw_time and "/" in raw_time:
            carried_time = [int(value) for value in raw_time.split("/", 1)]
        bar_refs = _refs(master_bar, "Bars")
        if track_index >= len(bar_refs):
            raise ValueError(f"Measure {measure_index + 1} has no bar for track {track_index}")
        bar = bars[bar_refs[track_index]]
        event_rows: list[dict] = []
        for voice_number, voice_ref in enumerate(_refs(bar, "Voices")):
            voice = voices[voice_ref]
            onset = Fraction(0)
            for order, beat_ref in enumerate(_refs(voice, "Beats")):
                beat = beats[beat_ref]
                rhythm_ref = beat.find("Rhythm")
                if rhythm_ref is None or rhythm_ref.get("ref") not in rhythms:
                    raise ValueError(f"Beat {beat_ref} has no known rhythm")
                value, dot, division, duration = _rhythm(rhythms[rhythm_ref.get("ref", "")])
                parsed_notes = [_note(notes[value], string_count) for value in _refs(beat, "Notes")]
                pick_direction = _property_text(beat, "PickStroke", "Direction")
                event_rows.append({
                    "order": order,
                    "voice": voice_number,
                    "onset": {"text": f"{onset.numerator}/{onset.denominator}"},
                    "duration_fraction": {"text": f"{duration.numerator}/{duration.denominator}"},
                    "state": "note" if parsed_notes else "rest",
                    "duration_value": value,
                    "dot": dot,
                    "division": division,
                    "notes": parsed_notes,
                    "effects": {
                        "pick_up": pick_direction == "Up",
                        "pick_down": pick_direction == "Down",
                    },
                    "free_text": beat.findtext("FreeText"),
                })
                onset += duration
        if not event_rows:
            capacity = Fraction(*carried_time)
            duration_value = carried_time[1] // carried_time[0] if carried_time[1] % carried_time[0] == 0 else 1
            event_rows.append({
                "order": 0,
                "voice": 0,
                "onset": {"text": "0/1"},
                "duration_fraction": {"text": f"{capacity.numerator}/{capacity.denominator}"},
                "state": "rest",
                "duration_value": duration_value,
                "dot": "none",
                "division": "1:1",
                "notes": [],
                "implicit_full_measure_rest": True,
            })
        section = master_bar.find("Section")
        measures.append({
            "number": measure_index + 1,
            "time_signature": list(carried_time),
            "section": ({
                "letter": section.findtext("Letter", "").strip(),
                "text": section.findtext("Text", "").strip(),
            } if section is not None else None),
            "events": event_rows,
        })
    return {
        "schema_version": "1.0",
        "source": str(path.resolve()),
        "source_format": "gpif",
        "track_index": track_index,
        "track_name": track.findtext("Name", f"Track {track_index + 1}"),
        "tempo_quarter": _tempo(root),
        "string_count": string_count,
        "string_tuning_midi": tuning,
        "measures": measures,
    }
