from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from guitarocr.paths import DATABASE_ROOT


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mix source-disjoint GP8 tie presence/count labels with the existing "
            "TuxGuitar tie dataset. GP8 records omit vertical-bin supervision."
        )
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--rhythm-task-root", default="gp8_score_rhythm_events")
    parser.add_argument("--tux-task-root", default="tie_events")
    parser.add_argument("--output-task-root", default="gp8_tie_events")
    args = parser.parse_args()
    database = args.database.resolve()
    output = database / args.output_task_root / "manifests"
    summary = {"schema_version": "1.0", "splits": {}}

    for split in ("train", "validation", "test"):
        records = []
        counts = Counter()
        for record in read_jsonl(
            database / args.tux_task_root / "manifests" / f"{split}.jsonl"
        ):
            copied = dict(record)
            copied["renderer_domain"] = "tuxguitar"
            copied["y_supervision"] = True
            records.append(copied)
            counts["tuxguitar_events"] += 1
            counts["tuxguitar_tie_events"] += int(bool(copied["tie_present"]))

        for record in read_jsonl(
            database / args.rhythm_task_root / "manifests" / f"{split}.jsonl"
        ):
            if record.get("renderer_domain") != "guitarpro8":
                continue
            copied = {
                "schema_version": "1.0",
                "sample_id": f"gp8tie_{record['sample_id']}",
                "source_id": record["source_id"],
                "split": split,
                "renderer_domain": "guitarpro8",
                "image": record["image"],
                "tie_present": bool(record.get("tie_present", False)),
                "tied_note_count": int(record.get("tied_note_count", 0)),
                "score_note_count": int(record.get("score_note_count", 0)),
                "attacked_note_count": int(record.get("attacked_note_count", 0)),
                "tied_notes": record.get("tied_notes", []),
                "tied_y_bins": [],
                "y_supervision": False,
                "alignment_method": record.get("alignment_method"),
            }
            records.append(copied)
            counts["guitarpro8_events"] += 1
            counts["guitarpro8_tie_events"] += int(copied["tie_present"])
            counts["guitarpro8_tie_notes"] += copied["tied_note_count"]

        records.sort(key=lambda item: item["sample_id"])
        write_jsonl(output / f"{split}.jsonl", records)
        summary["splits"][split] = {"events": len(records), **dict(counts)}

    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
