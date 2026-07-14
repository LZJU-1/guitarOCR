from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import shutil
from typing import Any


MERGED_NAME = re.compile(
    r"^(?:tab|notation|both)_[0-9a-f]{14}_(?P<song>.+)_page\d+\.[^.]+$",
    re.IGNORECASE,
)


def source_group(file_name: str) -> str:
    match = MERGED_NAME.match(Path(file_name).name)
    if not match:
        raise ValueError(f"Cannot recover source song from merged COCO image name: {file_name}")
    return match.group("song")


def source_family(group: str) -> str:
    return group.split("_", 1)[0].lower()


def _load_split(dataset_dir: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    annotation_path = dataset_dir / "annotations" / f"instance_{split}.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    records: list[dict[str, Any]] = []
    for image in payload.get("images", []):
        file_name = str(image["file_name"])
        source_path = dataset_dir / "images" / file_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing merged COCO image: {source_path}")
        records.append(
            {
                "source_split": split,
                "source_path": source_path,
                "group": source_group(file_name),
                "image": image,
                "annotations": annotations_by_image.get(int(image["id"]), []),
            }
        )
    return payload, records


def _choose_validation_groups(groups: set[str], val_ratio: float, seed: int) -> set[str]:
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    by_family: dict[str, list[str]] = defaultdict(list)
    for group in sorted(groups):
        by_family[source_family(group)].append(group)

    selected: set[str] = set()
    for family_index, family in enumerate(sorted(by_family)):
        family_groups = by_family[family]
        rng = random.Random(seed + family_index * 1_000_003)
        rng.shuffle(family_groups)
        if len(family_groups) <= 1:
            count = 0
        else:
            count = max(1, min(len(family_groups) - 1, round(len(family_groups) * val_ratio)))
        selected.update(family_groups[:count])
    return selected


def _clean_output(output_dir: Path) -> None:
    for relative in ("images", "images_mask", "annotations"):
        target = output_dir / relative
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for name in ("source_disjoint_summary.json", "dataset_stats.json", "label.txt"):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def _write_split(
    records: list[dict[str, Any]],
    output_dir: Path,
    split: str,
    template: dict[str, Any],
) -> dict[str, Any]:
    image_dir = output_dir / "images"
    mask_dir = output_dir / "images_mask"
    annotation_dir = output_dir / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    output_images: list[dict[str, Any]] = []
    output_annotations: list[dict[str, Any]] = []
    category_names = {int(item["id"]): str(item["name"]) for item in template.get("categories", [])}
    category_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()

    for image_id, record in enumerate(sorted(records, key=lambda item: item["image"]["file_name"]), start=1):
        image = dict(record["image"])
        image["id"] = image_id
        image["source_group"] = record["group"]
        file_name = str(image["file_name"])
        shutil.copy2(record["source_path"], image_dir / file_name)
        shutil.copy2(record["source_path"], mask_dir / file_name)
        output_images.append(image)
        group_counts[record["group"]] += 1

        for annotation in record["annotations"]:
            copied = dict(annotation)
            copied["id"] = len(output_annotations) + 1
            copied["image_id"] = image_id
            output_annotations.append(copied)
            category_counts[category_names.get(int(copied["category_id"]), "unknown")] += 1

    output_payload = {
        key: template[key]
        for key in ("info", "licenses")
        if key in template
    }
    output_payload.update(
        {
            "images": output_images,
            "annotations": output_annotations,
            "categories": template.get("categories", []),
        }
    )
    annotation_path = annotation_dir / f"instance_{split}.json"
    annotation_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return {
        "songs": len(group_counts),
        "images": len(output_images),
        "annotations": len(output_annotations),
        "families": dict(sorted(Counter(source_family(group) for group in group_counts).items())),
        "category_counts": dict(sorted(category_counts.items())),
        "annotation_file": str(annotation_path),
    }


def build_source_disjoint_coco(
    input_dir: Path,
    output_dir: Path,
    *,
    val_ratio: float = 0.1,
    seed: int = 20260716,
    overwrite: bool = False,
) -> dict[str, Any]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output COCO directories must be different.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        _clean_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_payload, train_records = _load_split(input_dir, "train")
    val_payload, val_records = _load_split(input_dir, "val")
    if train_payload.get("categories") != val_payload.get("categories"):
        raise ValueError("Train and validation COCO categories do not match.")
    records = train_records + val_records
    groups = {str(record["group"]) for record in records}
    validation_groups = _choose_validation_groups(groups, val_ratio, seed)
    train = [record for record in records if record["group"] not in validation_groups]
    val = [record for record in records if record["group"] in validation_groups]

    summary = {
        "schema_version": "1.0",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "val_ratio": val_ratio,
        "source_song_count": len(groups),
        "train": _write_split(train, output_dir, "train", train_payload),
        "val": _write_split(val, output_dir, "val", train_payload),
    }
    train_groups = {record["group"] for record in train}
    val_groups = {record["group"] for record in val}
    overlap = sorted(train_groups & val_groups)
    summary["source_overlap_count"] = len(overlap)
    summary["source_overlap"] = overlap
    category_names = [str(item["name"]) for item in train_payload.get("categories", [])]
    (output_dir / "label.txt").write_text("\n".join(category_names) + "\n", encoding="utf-8")
    summary_path = output_dir / "source_disjoint_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-split a merged Guitar Pro layout COCO dataset by source song across all display modes."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build_source_disjoint_coco(
        args.input_dir,
        args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
