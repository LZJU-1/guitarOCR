from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from guitarocr.data.build_gp8_measure_sequence_dataset import recognition_prompt
from guitarocr.data.measure_sequence_constraints import validate_measure_target
from guitarocr.evaluation.measure_sequence_metrics import MeasureSequenceMetrics


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_rows(manifest: Path, maximum: int | None, seed: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    stable = lambda value: sha256(f"{seed}:{value}".encode("utf-8")).digest()
    if not maximum:
        return sorted(rows, key=lambda row: stable(row["id"]))

    modes = sorted({row["mode"] for row in rows})
    quotas = {mode: maximum // len(modes) for mode in modes}
    for mode in modes[: maximum % len(modes)]:
        quotas[mode] += 1
    selected = []
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        mode_selected: dict[str, dict[str, Any]] = {}

        # Reserve a small, source-diverse slice for every represented semantic
        # class before filling the quota. A purely uniform sample can contain
        # almost no bends, harmonics or multi-voice measures and therefore
        # cannot support the per-technique release gate.
        tags = sorted({
            str(tag)
            for row in mode_rows
            for tag in row.get("semantic_tags", [])
        })
        coverage_per_tag = min(20, max(4, quotas[mode] // 50))
        for tag in tags:
            candidates = sorted(
                (row for row in mode_rows if tag in row.get("semantic_tags", [])),
                key=lambda row: stable(f"technique:{mode}:{tag}:{row['id']}"),
            )
            tag_sources: set[str] = set()
            for row in candidates:
                if (
                    len(mode_selected) >= quotas[mode]
                    or len(tag_sources) >= coverage_per_tag
                ):
                    break
                if row["id"] in mode_selected or row["source_id"] in tag_sources:
                    continue
                mode_selected[row["id"]] = row
                tag_sources.add(row["source_id"])

        by_source: dict[str, list[dict[str, Any]]] = {}
        for row in mode_rows:
            if row["id"] not in mode_selected:
                by_source.setdefault(row["source_id"], []).append(row)
        for values in by_source.values():
            values.sort(key=lambda row: stable(row["id"]))
        source_ids = sorted(by_source, key=lambda value: stable(f"{mode}:{value}"))
        while len(mode_selected) < quotas[mode]:
            progressed = False
            for source_id in source_ids:
                if not by_source[source_id]:
                    continue
                row = by_source[source_id].pop()
                mode_selected[row["id"]] = row
                progressed = True
                if len(mode_selected) >= quotas[mode]:
                    break
            if not progressed:
                break
        selected.extend(mode_selected.values())
    return sorted(selected, key=lambda row: stable(row["id"]))


def _artifact_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    identity: dict[str, Any] = {"path": str(resolved)}
    if resolved.is_file():
        stat = resolved.stat()
        identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        return identity
    if not resolved.is_dir():
        identity["missing"] = True
        return identity
    artifacts = []
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "model.safetensors",
    ):
        candidate = resolved / name
        if candidate.is_file():
            stat = candidate.stat()
            artifacts.append({
                "name": name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    identity["artifacts"] = artifacts
    return identity


def _run_signature(args: argparse.Namespace, rows: list[dict[str, Any]]) -> str:
    manifest = args.manifest.resolve()
    manifest_stat = manifest.stat()
    value = {
        "schema": 1,
        "manifest": {
            "path": str(manifest),
            "size": manifest_stat.st_size,
            "mtime_ns": manifest_stat.st_mtime_ns,
        },
        "selected_ids": [row["id"] for row in rows],
        "model": _artifact_identity(args.model),
        "adapter": _artifact_identity(args.adapter),
        "max_new_tokens": args.max_new_tokens,
        "maximum_attempts": args.maximum_attempts,
        "image_ablation": args.image_ablation,
        "seed": args.seed,
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _inference_images(
    rows: list[dict[str, Any]], image_ablation: str
) -> dict[str, str]:
    if image_ablation == "none":
        return {row["id"]: row["image"] for row in rows}
    if image_ablation != "shuffled":
        raise ValueError(f"Unsupported image ablation: {image_ablation}")

    result: dict[str, str] = {}
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        if len(mode_rows) < 2:
            raise ValueError(f"Need at least two {mode} rows for shuffled images")
        for index, row in enumerate(mode_rows):
            replacement = None
            for offset in range(1, len(mode_rows)):
                candidate = mode_rows[(index + offset) % len(mode_rows)]
                if candidate["source_id"] != row["source_id"]:
                    replacement = candidate
                    break
            if replacement is None:
                raise ValueError(
                    f"Need at least two independent {mode} sources for shuffled images"
                )
            result[row["id"]] = replacement["image"]
    return result


def _extract_m2(text: str) -> str:
    value = text.strip()
    start = value.find("M2")
    if start >= 0:
        value = value[start:]
    for marker in ("<|endoftext|>", "<|user|>", "<|assistant|>"):
        value = value.partition(marker)[0]
    return value.strip().strip("`").strip()


def run_inference(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = _load_rows(args.manifest, args.max_samples, args.seed)
    inference_images = _inference_images(rows, args.image_ablation)
    signature = _run_signature(args, rows)
    completed: dict[str, dict[str, Any]] = {}
    if args.predictions.is_file() and not args.resume:
        args.predictions.unlink()
    if args.predictions.is_file():
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                if record.get("run_signature") != signature:
                    raise ValueError(
                        "Existing predictions were produced by a different "
                        "manifest/model/adapter selection. Remove the file or "
                        "run without --resume to overwrite it."
                    )
                completed[record["id"]] = record

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model = model.to(args.device).eval()
    label_cache: dict[str, dict[str, Any]] = {}
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            if row["id"] in completed:
                continue
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "url": inference_images[row["id"]]},
                    {
                        "type": "text",
                        "text": recognition_prompt(
                            row["mode"], row.get("previous_context")
                        ),
                    },
                ],
            }]
            label_path = str(row.get("label_json") or "")
            if label_path and label_path not in label_cache:
                label_cache[label_path] = json.loads(
                    Path(label_path).read_text(encoding="utf-8")
                )
            track = label_cache.get(label_path, {}).get("track", {})
            tuning = track.get("tuning_midi_high_to_low")
            string_count = track.get("string_count")
            raw = ""
            predicted = ""
            constraint_errors: list[str] = []
            for attempt in range(1, args.maximum_attempts + 1):
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                inputs.pop("token_type_ids", None)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                raw = processor.decode(
                    generated[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=False,
                )
                predicted = _extract_m2(raw)
                _parsed, constraint_errors = validate_measure_target(
                    predicted,
                    row["mode"],
                    tuning=tuning,
                    string_count=string_count,
                )
                if not constraint_errors or attempt >= args.maximum_attempts:
                    break
                messages.extend([
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": predicted}],
                    },
                    {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": (
                                "Correct the M2 using the same image. Constraint errors: "
                                + "; ".join(constraint_errors[:8])
                                + ". Return only one corrected M2 fragment."
                            ),
                        }],
                    },
                ])
            record = {
                "run_signature": signature,
                "id": row["id"],
                "source_id": row["source_id"],
                "mode": row["mode"],
                "image": row["image"],
                "inference_image": inference_images[row["id"]],
                "image_ablation": args.image_ablation,
                "expected": row["target"],
                "predicted": predicted,
                "raw": raw,
                "tuning": tuning,
                "string_count": string_count,
                "recognition_attempts": attempt,
                "constraint_errors": constraint_errors,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed[row["id"]] = record
            print(f"[{index}/{len(rows)}] {row['id']}", flush=True)


def evaluate(predictions: Path, metrics_path: Path) -> dict[str, Any]:
    metrics = MeasureSequenceMetrics()
    by_mode: dict[str, MeasureSequenceMetrics] = {}
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        metrics.update(
            row["expected"],
            row["predicted"],
            row["mode"],
            tuning=row.get("tuning"),
            string_count=row.get("string_count"),
        )
        by_mode.setdefault(row["mode"], MeasureSequenceMetrics()).update(
            row["expected"],
            row["predicted"],
            row["mode"],
            tuning=row.get("tuning"),
            string_count=row.get("string_count"),
        )
    result = {
        "overall": metrics.result(),
        "by_mode": {mode: value.result() for mode, value in sorted(by_mode.items())},
    }
    _write_json(metrics_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer and score GLM-OCR M2 measure sequences.")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("database/gp8_measure_sequence_v2/manifests/test.jsonl"),
    )
    parser.add_argument("--model", type=Path, default=Path("tools/models/GLM-OCR"))
    parser.add_argument(
        "--adapter", type=Path,
        default=Path("output/glm_ocr_measure_sequence_v2_lora"),
    )
    parser.add_argument(
        "--predictions", type=Path,
        default=Path("reports/glm_ocr_measure_sequence_test_predictions.jsonl"),
    )
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("reports/glm_ocr_measure_sequence_test_metrics.json"),
    )
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--image-ablation", choices=("none", "shuffled"), default="none",
        help="Replace each image with another held-out image of the same layout.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume only when every existing record has the same run signature.",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if not args.evaluate_only:
        run_inference(args)
    result = evaluate(args.predictions, args.metrics)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
