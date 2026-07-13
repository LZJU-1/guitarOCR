from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from guitarocr.data.build_score_rhythm_dataset import TECHNIQUE_CLASSES
from guitarocr.data.gpif import load_gpif_score
from guitarocr.paths import DATABASE_ROOT


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event_effects(event: dict) -> dict[str, bool]:
    event_values = event.get("effects") or {}
    values = {name: bool(event_values.get(name, False)) for name in TECHNIQUE_CLASSES}
    for note in event.get("notes", []):
        effects = note.get("effects") or {}
        for name in TECHNIQUE_CLASSES:
            values[name] = values[name] or bool(effects.get(name, False))
        values["dead"] = values["dead"] or bool(
            note.get("dead") or note.get("muted")
            or str(note.get("printed_fret", note.get("fret", ""))).upper() == "X"
        )
    return values


def augment(database: Path, source_id: str) -> dict:
    sources = {record["id"]: record for record in read_jsonl(database / "manifests" / "sources.jsonl")}
    if source_id not in sources:
        raise KeyError(f"Unknown source id: {source_id}")
    source = sources[source_id]
    gp_path = database / source["source_gp"]
    if gp_path.suffix.lower() != ".gp":
        raise ValueError("Exact GPIF augmentation currently supports Guitar Pro 7/8 .gp ZIP files only")
    track_index = max(0, int(source.get("target_track_number", 1)) - 1)
    gpif = load_gpif_score(gp_path, track_index)

    records: list[dict] = []
    for split in ("train", "validation", "test"):
        for record in read_jsonl(database / "rhythm_events" / "manifests" / f"{split}.jsonl"):
            if record["source_id"] == source_id:
                records.append(record)
    labels_by_measure: dict[int, list[tuple[Path, dict]]] = defaultdict(list)
    for record in records:
        path = database / record["label"]
        label = json.loads(path.read_text(encoding="utf-8"))
        labels_by_measure[int(label["measure_number"])].append((path, label))

    before = Counter()
    after = Counter()
    changed_labels = 0
    for measure in gpif["measures"]:
        number = int(measure["number"])
        expected_events = [event for event in measure.get("events", []) if int(event.get("voice", 0)) == 0]
        labels = sorted(labels_by_measure.get(number, []), key=lambda item: int(item[1]["precise_start"]))
        if len(labels) != len(expected_events):
            raise ValueError(
                f"Measure {number}: GPIF has {len(expected_events)} primary events, labels have {len(labels)}"
            )
        for (path, label), expected in zip(labels, expected_events):
            primary = label["voices"][0]
            old = {name: bool(primary.get("effects", {}).get(name, False)) for name in TECHNIQUE_CLASSES}
            new = event_effects(expected)
            before.update(name for name, value in old.items() if value)
            after.update(name for name, value in new.items() if value)
            if old == new:
                continue
            primary["effects"] = new
            temporary = path.with_suffix(f".json.tmp.{os.getpid()}")
            temporary.write_text(json.dumps(label, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
            changed_labels += 1
    return {
        "source_id": source_id,
        "gp": str(gp_path),
        "events": len(records),
        "changed_labels": changed_labels,
        "before": dict(sorted(before.items())),
        "after": dict(sorted(after.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay exact GP7/8 GPIF techniques onto rendered event labels")
    parser.add_argument("source_id")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    args = parser.parse_args()
    print(json.dumps(augment(args.database.resolve(), args.source_id), ensure_ascii=False))


if __name__ == "__main__":
    main()
