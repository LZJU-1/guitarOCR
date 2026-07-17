from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from guitarocr.data.gp_measure_sequence import format_measure_target, parse_measure_target
from guitarocr.data.measure_sequence_constraints import validate_measure_target


def validate(dataset: Path) -> dict[str, Any]:
    labels = sorted((dataset / "labels").glob("*.json"))
    totals: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    source_hashes: set[str] = set()
    for path in labels:
        label = json.loads(path.read_text(encoding="utf-8"))
        source_id = str(label["source_id"])
        source_hash = str(label["sha256"])
        totals["sources"] += 1
        if source_id in source_ids or source_hash in source_hashes:
            totals["duplicate_sources"] += 1
        source_ids.add(source_id)
        source_hashes.add(source_hash)
        tuning = [int(value) for value in label["track"]["tuning_midi_high_to_low"]]
        totals["measures"] += len(label["measures"])
        for measure in label["measures"]:
            for mode, target in measure["targets"].items():
                totals["targets"] += 1
                try:
                    parsed = parse_measure_target(target)
                    canonical = format_measure_target(parsed, mode)
                except Exception as error:
                    totals["syntax_failures"] += 1
                    if len(failures) < 50:
                        failures.append({
                            "source_id": source_id,
                            "measure": measure["number"],
                            "mode": mode,
                            "error": f"syntax:{error}",
                        })
                    continue
                if canonical != target:
                    totals["canonical_failures"] += 1
                    if len(failures) < 50:
                        failures.append({
                            "source_id": source_id,
                            "measure": measure["number"],
                            "mode": mode,
                            "error": "non_canonical_target",
                            "target": target,
                            "canonical": canonical,
                        })
                _parsed, errors = validate_measure_target(
                    target,
                    mode,
                    tuning=tuning,
                    string_count=len(tuning),
                )
                if errors:
                    totals["constraint_failures"] += 1
                    if len(failures) < 50:
                        failures.append({
                            "source_id": source_id,
                            "measure": measure["number"],
                            "mode": mode,
                            "error": errors,
                        })

    split_sources: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        manifest = dataset / "manifests" / f"{split}.jsonl"
        if not manifest.is_file():
            continue
        split_sources[split] = {
            json.loads(line)["source_id"]
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line
        }
    leakage = {
        f"{left}:{right}": sorted(split_sources[left] & split_sources[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        if left in split_sources and right in split_sources
        and split_sources[left] & split_sources[right]
    }
    totals["split_leakage_sources"] = sum(len(values) for values in leakage.values())
    return {
        "schema_version": "1.0",
        "dataset": str(dataset.resolve()),
        "totals": dict(totals),
        "split_sources": {
            split: len(values) for split, values in sorted(split_sources.items())
        },
        "split_leakage": leakage,
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate every M2 label and source-disjoint manifest."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("database/gp8_measure_sequence_v1")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/m2_dataset_validation.json")
    )
    args = parser.parse_args()
    report = validate(args.dataset.resolve())
    value = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(value, encoding="utf-8")
    print(value, end="")
    totals = report["totals"]
    if any(
        int(totals.get(name, 0))
        for name in (
            "duplicate_sources",
            "syntax_failures",
            "canonical_failures",
            "constraint_failures",
            "split_leakage_sources",
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
