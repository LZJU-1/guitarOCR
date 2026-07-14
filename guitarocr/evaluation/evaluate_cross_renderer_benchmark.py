from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

from guitarocr.evaluation.evaluate_gpif_ir import evaluate
from guitarocr.paths import DATABASE_ROOT, PROJECT_ROOT


SUPPORTED_LAYOUTS = {"tab_only", "score_tab"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def reference_from_song_label(label: dict) -> dict:
    measures = []
    for measure in label["measures"]:
        events = []
        onset_by_voice: dict[int, Fraction] = defaultdict(Fraction)
        for beat in measure.get("beats", []):
            pick_stroke = int(beat.get("pick_stroke", 0) or 0)
            event_voices = []
            event_notes = []
            for voice in beat.get("voices", []):
                voice_index = int(voice["index"])
                duration = voice["duration"]
                value = int(duration["value"])
                dot = (
                    "double" if duration.get("double_dotted") else
                    "single" if duration.get("dotted") else "none"
                )
                enters = int(duration.get("division_enters", 1))
                times = int(duration.get("division_times", 1))
                amount = Fraction(1, value) * Fraction(times, enters)
                if dot == "single":
                    amount *= Fraction(3, 2)
                elif dot == "double":
                    amount *= Fraction(7, 4)
                notes = []
                for note in voice.get("notes", []):
                    effects = dict(note.get("effects") or {})
                    dead = bool(effects.get("dead"))
                    copied_note = {
                        "string": int(note["string"]),
                        "fret": int(note["fret"]),
                        "printed_fret": "X" if dead else int(note["fret"]),
                        "voice": voice_index,
                        "tie_in": bool(note.get("tied")),
                        "tie_out": False,
                        "effects": effects,
                    }
                    notes.append(copied_note)
                    event_notes.append(copied_note)
                onset = onset_by_voice[voice_index]
                event_voices.append({
                    "voice": voice_index,
                    "onset": {"text": f"{onset.numerator}/{onset.denominator}"},
                    "duration_fraction": {
                        "text": f"{amount.numerator}/{amount.denominator}"
                    },
                    "state": "rest" if voice.get("rest") else "note",
                    "duration_value": value,
                    "dot": dot,
                    "division": f"{enters}:{times}",
                    "notes": notes,
                    "effects": {
                        "pick_up": pick_stroke == 1,
                        "pick_down": pick_stroke == -1,
                    },
                })
                onset_by_voice[voice_index] += amount
            events.append({
                "order": len(events),
                "voices": event_voices,
                "notes": event_notes,
                "effects": {
                    "pick_up": pick_stroke == 1,
                    "pick_down": pick_stroke == -1,
                },
            })
        time_signature = measure.get("time_signature") or {}
        measures.append({
            "number": int(measure["number"]),
            "time_signature": [
                int(time_signature.get("numerator", 4)),
                int(time_signature.get("denominator", 4)),
            ],
            "section": None,
            "events": events,
        })
    target = label["target_track"]
    tuning = [
        int(item["midi_pitch"])
        for item in sorted(target.get("tuning", []), key=lambda item: int(item["string"]))
    ]
    return {
        "schema_version": "1.0",
        "source": label.get("original_path", label["id"]),
        "source_format": "tuxguitar_dataset_label",
        "track_index": int(target.get("number", 1)) - 1,
        "track_name": target.get("name", "Guitar"),
        "tempo_quarter": next(
            (int(measure["tempo_quarter"]) for measure in label["measures"]
             if measure.get("tempo_quarter")),
            None,
        ),
        "string_count": int(target["string_count"]),
        "string_tuning_midi": tuning,
        "measures": measures,
    }


def run_sample(record: dict, output: Path, force: bool) -> dict:
    sample_root = output / "samples" / record["sample_id"]
    score_ir = sample_root / "inference" / "document_score_ir.json"
    result_path = sample_root / "evaluation.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    if record["layout"] not in SUPPORTED_LAYOUTS:
        result = {
            "sample_id": record["sample_id"],
            "renderer": record["renderer"],
            "layout": record["layout"],
            "status": "unsupported_layout",
            "reason": "score_only pitch-to-string reconstruction is not implemented",
        }
    else:
        sample_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "guitarocr.cli.pdf_to_gp",
            str(PROJECT_ROOT / record["pdf"]),
            "-o",
            str(sample_root / "PRE.gp5"),
            "--work-dir",
            str(sample_root / "inference"),
            "--layout",
            record["layout"],
            "--force-pdf-render",
            "--no-preview",
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode and not score_ir.is_file():
            result = {
                "sample_id": record["sample_id"],
                "renderer": record["renderer"],
                "layout": record["layout"],
                "status": "inference_failed",
                "command": command,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        else:
            reference_label = json.loads(
                (PROJECT_ROOT / record["reference_label"]).read_text(encoding="utf-8")
            )
            reference = reference_from_song_label(reference_label)
            predicted = json.loads(score_ir.read_text(encoding="utf-8"))
            evaluation = evaluate(reference, predicted)
            result = {
                "sample_id": record["sample_id"],
                "source_id": record["source_id"],
                "renderer": record["renderer"],
                "layout": record["layout"],
                "status": "success",
                "gp5_export_status": "success" if completed.returncode == 0 else "failed",
                "gp5_export_error": completed.stderr[-4000:] if completed.returncode else None,
                "metrics": evaluation["metrics"],
                "totals": evaluation["totals"],
                "failure_cases": evaluation["failure_cases"][:20],
                "score_ir": str(score_ir),
                "output_gp5": (
                    str(sample_root / "PRE.gp5")
                    if (sample_root / "PRE.gp5").is_file() else None
                ),
            }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def aggregate(results: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for result in results:
        groups[(result["renderer"], result["layout"])].append(result)
    rows = {}
    for (renderer, layout), values in sorted(groups.items()):
        successful = [value for value in values if value["status"] == "success"]
        totals = Counter()
        measure_count = 0
        actual_measures = 0
        for value in successful:
            totals.update(value["totals"])
            metrics = value["metrics"]
            measure_count += int(metrics["measure_count_expected"])
            actual_measures += int(metrics["measure_count_actual"])
        matched = totals["matched_events"]
        expected = totals["expected_events"]
        actual = totals["actual_events"]
        rows[f"{renderer}/{layout}"] = {
            "samples": len(values),
            "success": len(successful),
            "failed": len(values) - len(successful),
            "statuses": dict(Counter(value["status"] for value in values)),
            "measure_count_expected": measure_count,
            "measure_count_actual": actual_measures,
            "measure_count_match": measure_count == actual_measures if successful else False,
            "event_precision": matched / actual if actual else 0.0,
            "event_recall": matched / expected if expected else 0.0,
            "core_event_exact": totals["event_exact"] / matched if matched else 0.0,
            "rhythm_exact": totals["rhythm_exact"] / matched if matched else 0.0,
            "notes_exact": totals["notes_exact"] / matched if matched else 0.0,
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pixel-only OCR on the cross-renderer benchmark before any fine-tuning."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATABASE_ROOT / "cross_renderer_benchmark" / "benchmark.jsonl",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--renderers")
    parser.add_argument("--layouts")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output = (args.output or manifest.parent / "evaluation").resolve()
    renderers = set(args.renderers.split(",")) if args.renderers else None
    layouts = set(args.layouts.split(",")) if args.layouts else None
    records = [row for row in read_jsonl(manifest) if row["status"] == "ready"]
    if renderers:
        records = [row for row in records if row["renderer"] in renderers]
    if layouts:
        records = [row for row in records if row["layout"] in layouts]
    if args.limit:
        records = records[: args.limit]
    results = []
    for index, record in enumerate(records, start=1):
        print(f"[{index}/{len(records)}] {record['sample_id']}", flush=True)
        results.append(run_sample(record, output, args.force))
    summary = {
        "schema_version": "1.0",
        "manifest": str(manifest),
        "training_data_used": False,
        "records": len(records),
        "successful": sum(row["status"] == "success" for row in results),
        "groups": aggregate(results),
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
