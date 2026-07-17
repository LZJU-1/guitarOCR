from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from guitarocr.data.gp_measure_sequence import (
    format_measure_target,
    parse_measure_target,
    parse_song,
    song_sequence_payload,
)
from guitarocr.evaluation.measure_sequence_metrics import MeasureSequenceMetrics


def _targets_from_gp5(path: Path, mode: str) -> tuple[list[str], dict[str, Any]]:
    song, source_hash = parse_song(path)
    payload = song_sequence_payload(song, 0, path, source_hash)
    return [measure["targets"][mode] for measure in payload["measures"]], payload


def _load_prediction(path: Path, mode: str) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values.append(format_measure_target(parse_measure_target(line), mode))
    return values


def _paired_metrics(expected: list[str], predicted: list[str], mode: str) -> dict[str, Any]:
    metrics = MeasureSequenceMetrics()
    paired = min(len(expected), len(predicted))
    for index in range(paired):
        metrics.update(expected[index], predicted[index], mode)
    result = metrics.result()
    result.update(
        {
            "expected_measure_count": len(expected),
            "predicted_measure_count": len(predicted),
            "paired_measure_count": paired,
            "missing_measure_count": max(0, len(expected) - len(predicted)),
            "extra_measure_count": max(0, len(predicted) - len(expected)),
            "measure_count_exact": len(expected) == len(predicted),
        }
    )
    return result


def evaluate(
    gt_source_gp5: Path,
    prediction_m2: Path,
    pre_gp5: Path,
    output: Path,
    *,
    mode: str = "tab",
) -> dict[str, Any]:
    expected_targets, expected_payload = _targets_from_gp5(gt_source_gp5, mode)
    predicted_targets = _load_prediction(prediction_m2, mode)
    readback_targets, readback_payload = _targets_from_gp5(pre_gp5, mode)
    result = {
        "schema_version": "1.0",
        "mode": mode,
        "gt_source_gp5": str(gt_source_gp5.resolve()),
        "prediction_m2": str(prediction_m2.resolve()),
        "pre_gp5": str(pre_gp5.resolve()),
        "source_to_prediction": _paired_metrics(
            expected_targets, predicted_targets, mode
        ),
        "prediction_to_gp5_readback": _paired_metrics(
            predicted_targets, readback_targets, mode
        ),
        "source": {
            "song": expected_payload["song"],
            "track": expected_payload["track"],
            "statistics": expected_payload["statistics"],
        },
        "readback": {
            "song": readback_payload["song"],
            "track": readback_payload["track"],
            "statistics": readback_payload["statistics"],
        },
        "measure_differences": [],
    }
    for index in range(min(len(expected_targets), len(predicted_targets))):
        if expected_targets[index] != predicted_targets[index]:
            result["measure_differences"].append(
                {
                    "measure_number": index + 1,
                    "expected": expected_targets[index],
                    "predicted": predicted_targets[index],
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full-document M2 OCR with its paired GT GP5 and GP5 readback."
    )
    parser.add_argument("--gt-source-gp5", type=Path, required=True)
    parser.add_argument("--prediction-m2", type=Path, required=True)
    parser.add_argument("--pre-gp5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("tab", "notation", "both"), default="tab")
    args = parser.parse_args()
    result = evaluate(
        args.gt_source_gp5,
        args.prediction_m2,
        args.pre_gp5,
        args.output,
        mode=args.mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
