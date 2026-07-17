from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from guitarocr.data.gpif import load_gpif_score
from guitarocr.export.export_score_ir_to_gp import export_ir


def _score_ir_measures(measures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt exact GPIF events to the document Score IR consumed by the GP5 writer."""

    result = []
    for measure in measures:
        events = []
        for source_event in measure.get("events", []):
            voice_index = int(source_event.get("voice", 0))
            voice = {
                "voice": voice_index,
                "state": source_event.get("state", "rest"),
                "onset": source_event.get("onset"),
                "duration_fraction": source_event.get("duration_fraction"),
                "duration_value": source_event.get("duration_value", 4),
                "dot": source_event.get("dot", "none"),
                "division": source_event.get("division", "1:1"),
            }
            notes = []
            for source_note in source_event.get("notes", []):
                note = dict(source_note)
                note["voice"] = voice_index
                effects = dict(note.get("effects") or {})
                note["effects"] = effects
                note["dead"] = bool(effects.get("muted"))
                notes.append(note)
            beat_effects = source_event.get("effects") or {}
            events.append(
                {
                    "order": int(source_event.get("order", len(events))),
                    "voices": [voice],
                    "notes": notes,
                    "technique_prediction": {
                        "positive": {
                            "pick_up": bool(beat_effects.get("pick_up")),
                            "pick_down": bool(beat_effects.get("pick_down")),
                        }
                    },
                }
            )
        result.append(
            {
                "number": int(measure.get("number", len(result) + 1)),
                "time_signature": measure.get("time_signature"),
                "section": measure.get("section"),
                "events": events,
            }
        )
    return result


def _track_index(path: Path, requested_index: int | None, requested_name: str | None) -> int:
    if requested_index is not None:
        return requested_index
    if requested_name:
        # load_gpif_score validates indices but does not expose the track list.
        # Probe sequentially so this stays coupled to the same parser used for
        # the exact semantics below.
        index = 0
        while True:
            try:
                score = load_gpif_score(path, index)
            except IndexError:
                break
            if str(score.get("track_name", "")).casefold() == requested_name.casefold():
                return index
            index += 1
        raise ValueError(f"Track {requested_name!r} was not found in {path}")
    return 0


def prepare(
    source: Path,
    output_dir: Path,
    *,
    track_index: int | None = None,
    track_name: str | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_index = _track_index(source, track_index, track_name)
    ground_truth = load_gpif_score(source, selected_index)
    track = {
        "track_index": 0,
        "name": ground_truth["track_name"],
        "tempo_quarter": ground_truth["tempo_quarter"],
        "string_count": ground_truth["string_count"],
        "string_tuning_midi": ground_truth["string_tuning_midi"],
        "capo": 0,
        "measures": _score_ir_measures(ground_truth["measures"]),
    }
    score_ir = {
        "schema_version": "1.0",
        "document": {
            "layout": "tab_only",
            "page_count": 0,
            "source_gp": str(source),
            "source_track_index": selected_index,
            "source_track_name": ground_truth["track_name"],
        },
        "tracks": [track],
        "scope": {
            "training_eligible": False,
            "purpose": "independent checkpoint end-to-end acceptance",
        },
    }
    ir_path = output_dir / "ground_truth_score_ir.json"
    ir_path.write_text(
        json.dumps(score_ir, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ground_truth_path = output_dir / "ground_truth_gpif.json"
    ground_truth_path.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    gp5_path = output_dir / "GT_SOURCE.gp5"
    export_report = export_ir(
        ir_path,
        gp5_path,
        preview=None,
        voice=0,
        tempo=int(ground_truth["tempo_quarter"]),
        tuning=[int(value) for value in ground_truth["string_tuning_midi"]],
        # The display-mode preparation helper is intentionally pinned to the
        # worker's GP5 parser. Keep this intermediate title ASCII-safe; the
        # original source name remains in the provenance JSON.
        title=str(ground_truth["track_name"] or f"track_{selected_index}"),
        preview_layout="tab_only",
    )
    result = {
        "source_gp": str(source),
        "source_track_index": selected_index,
        "source_track_name": ground_truth["track_name"],
        "tempo_quarter": ground_truth["tempo_quarter"],
        "string_tuning_midi": ground_truth["string_tuning_midi"],
        "measure_count": len(ground_truth["measures"]),
        "ground_truth_gpif": str(ground_truth_path),
        "ground_truth_score_ir": str(ir_path),
        "gt_source_gp5": str(gp5_path),
        "export_report": export_report,
    }
    (output_dir / "ground_truth_prepare_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one GP7/8 GPIF guitar track and materialize a tab-ready GP5."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--track-index", type=int)
    parser.add_argument("--track-name")
    args = parser.parse_args()
    result = prepare(
        args.source,
        args.output_dir,
        track_index=args.track_index,
        track_name=args.track_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
