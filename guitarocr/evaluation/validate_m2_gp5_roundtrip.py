from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from guitarocr.data.gp_measure_sequence import (
    format_measure_target,
    parse_measure_target,
    parse_song,
    select_target_track,
    song_sequence_payload,
)
from guitarocr.export.measure_sequence_to_gp5 import write_targets_gp5


def _canonical(target: str, mode: str) -> str:
    return format_measure_target(parse_measure_target(target), mode)


_ARTIFICIAL_DEFAULT = re.compile(r"harm:artificial:-?\d+:-?\d+:-?\d+")


def _roundtrip_equivalent(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    # Some source GP3-5 files contain an artificial-harmonic marker without
    # its optional pitch-class fields. The GP5 writer must materialize default
    # bytes, which the reader then exposes. The printed symbol and all known
    # source semantics are unchanged, so compare that case as equivalent.
    return expected == _ARTIFICIAL_DEFAULT.sub("harm:artificial", actual)


def validate(
    manifest: Path,
    output: Path,
    modes: set[str],
    source_ids: set[str] | None = None,
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["mode"] in modes and (
                source_ids is None or row["source_id"] in source_ids
            ):
                groups[(row["source_id"], row["mode"])].append(row)

    output.mkdir(parents=True, exist_ok=True)
    overall: Counter[str] = Counter()
    by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    failures = []
    for index, ((source_id, mode), rows) in enumerate(
        sorted(groups.items()), start=1
    ):
        rows.sort(key=lambda row: int(row["measure_index"]))
        targets = [_canonical(row["target"], mode) for row in rows]
        try:
            label = json.loads(
                Path(rows[0]["label_json"]).read_text(encoding="utf-8")
            )
            gp5 = output / f"{source_id}_{mode}.gp5"
            write_targets_gp5(
                targets,
                gp5,
                mode=mode,
                title=label["song"]["title"],
                artist=label["song"]["artist"],
                tuning=label["track"]["tuning_midi_high_to_low"],
                capo=label["track"]["capo"],
            )
            song, source_hash = parse_song(gp5)
            track_index, _track = select_target_track(song)
            payload = song_sequence_payload(
                song, track_index, gp5, source_hash
            )
            actual = [
                _canonical(measure["targets"][mode], mode)
                for measure in payload["measures"]
            ]
        except Exception as exc:
            for counter in (overall, by_mode[mode]):
                counter["songs"] += 1
                counter["expected_measures"] += len(targets)
                counter["errors"] += 1
            failures.append(
                {
                    "source_id": source_id,
                    "mode": mode,
                    "expected_count": len(targets),
                    "actual_count": 0,
                    "exact": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"[{index}/{len(groups)}] {source_id} {mode} ERROR: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        exact = sum(
            _roundtrip_equivalent(expected, predicted)
            for expected, predicted in zip(targets, actual)
        )
        for counter in (overall, by_mode[mode]):
            counter["songs"] += 1
            counter["expected_measures"] += len(targets)
            counter["actual_measures"] += len(actual)
            counter["exact_measures"] += exact
        if exact != len(targets) or len(actual) != len(targets):
            examples = [
                {"measure": measure, "expected": expected, "actual": predicted}
                for measure, (expected, predicted) in enumerate(
                    zip(targets, actual), start=1
                )
                if not _roundtrip_equivalent(expected, predicted)
            ][:3]
            failures.append({
                "source_id": source_id,
                "mode": mode,
                "expected_count": len(targets),
                "actual_count": len(actual),
                "exact": exact,
                "examples": examples,
            })
        print(f"[{index}/{len(groups)}] {source_id} {mode} {exact}/{len(targets)}", flush=True)
    return {
        "schema_version": "1.0",
        "manifest": str(manifest.resolve()),
        "summary": dict(overall),
        "by_mode": {mode: dict(value) for mode, value in sorted(by_mode.items())},
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write canonical M2 to GP5 and compare its parsed semantics."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("database/gp8_measure_sequence_v1/manifests/test.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lab/m2_gp5_roundtrip_test"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/m2_gp5_full_roundtrip_test.json"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("tab", "notation", "both"),
        default=["tab", "notation", "both"],
    )
    parser.add_argument(
        "--source-id",
        action="append",
        help="Limit validation to one or more source ids (repeatable).",
    )
    args = parser.parse_args()
    report = validate(
        args.manifest.resolve(),
        args.output.resolve(),
        set(args.modes),
        set(args.source_id) if args.source_id else None,
    )
    value = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(value, encoding="utf-8")
    print(value, end="")
    if report["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
