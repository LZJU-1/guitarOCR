from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from guitarocr.paths import DATABASE_ROOT


INPUT_WIDTH = 512
INPUT_HEIGHT = 128
CLASSES = [*(f"digit_{digit}" for digit in range(10)), "dead_x"]
# The original page split happened to place every dead-X source in training.
# These detector-only source swaps preserve split sizes while making X visible
# in held-out evaluation. No source is ever shared across splits.
SPLIT_OVERRIDES = {
    "feb2735ca835c9cb": "test",
    "0d3a153380b36c1d": "train",
    "eadd7f4740e2f6af": "validation",
    "2f48f92446035eba": "train",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_generated_files(root: Path) -> None:
    expected = (root.parent / "dataset").resolve()
    if root.resolve() != expected:
        raise ValueError(f"Refusing to clean unexpected detector dataset path: {root}")
    for split in ("train", "validation", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for directory, pattern in ((image_dir, "*.png"), (label_dir, "*.json")):
            for path in directory.glob(pattern):
                if path.is_file():
                    path.unlink()


def tile_measure(
    page: Image.Image,
    staff_bbox: list[float],
    symbols: list[dict],
) -> list[tuple[Image.Image, list[dict], dict]]:
    x, y, width, height = staff_bbox
    pad_x = max(3.0, height * 0.04)
    pad_y = max(6.0, height * 0.08)
    left = max(0, math.floor(x - pad_x))
    top = max(0, math.floor(y - pad_y))
    right = min(page.width, math.ceil(x + width + pad_x))
    bottom = min(page.height, math.ceil(y + height + pad_y))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid measure crop: {(left, top, right, bottom)}")

    crop = page.crop((left, top, right, bottom)).convert("L")
    # Preserve symbol height. Very wide measures are tiled horizontally instead
    # of being squeezed until their digits become only a few pixels tall.
    scale = INPUT_HEIGHT / crop.height
    resized_width = max(1, round(crop.width * scale))
    resized_height = INPUT_HEIGHT
    resized = crop.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    if resized_width <= INPUT_WIDTH:
        tile_starts = [0]
        canvas_offset_x = (INPUT_WIDTH - resized_width) // 2
    else:
        stride = INPUT_WIDTH - 64
        tile_starts = list(range(0, max(1, resized_width - INPUT_WIDTH + 1), stride))
        last_start = resized_width - INPUT_WIDTH
        if tile_starts[-1] != last_start:
            tile_starts.append(last_start)
        canvas_offset_x = 0

    results: list[tuple[Image.Image, list[dict], dict]] = []
    for tile_start in tile_starts:
        output = Image.new("L", (INPUT_WIDTH, INPUT_HEIGHT), 255)
        if resized_width <= INPUT_WIDTH:
            output.paste(resized, (canvas_offset_x, 0))
        else:
            output.paste(resized.crop((tile_start, 0, tile_start + INPUT_WIDTH, INPUT_HEIGHT)), (0, 0))

        output_symbols: list[dict] = []
        for symbol in symbols:
            sx, sy, sw, sh = symbol["bbox"]
            transformed_x = (sx - left) * scale + canvas_offset_x - tile_start
            transformed_y = (sy - top) * scale
            center_x = (symbol["center"][0] - left) * scale + canvas_offset_x - tile_start
            center_y = (symbol["center"][1] - top) * scale
            # Symbols in the 64-pixel overlap are intentionally present in both
            # tiles. Inference merges those duplicates after mapping to page space.
            if not (0 <= center_x < INPUT_WIDTH):
                continue
            copied = dict(symbol)
            copied["bbox"] = [transformed_x, transformed_y, sw * scale, sh * scale]
            copied["center"] = [center_x, center_y]
            output_symbols.append(copied)

        transform = {
            "page_crop_xyxy": [left, top, right, bottom],
            "scale": scale,
            "offset": [canvas_offset_x, 0],
            "tile_start_x": tile_start,
            "tile_count": len(tile_starts),
            "resized_size": [resized_width, resized_height],
            "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
        }
        results.append((output, output_symbols, transform))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build measure crops for the TuxGuitar TAB symbol detector.")
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    args = parser.parse_args()

    database = args.database.resolve()
    detector_root = database / "tab_detector"
    dataset_root = detector_root / "dataset"
    manifest_root = detector_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    clean_generated_files(dataset_root)

    source_records = read_jsonl(database / "manifests" / "sources.jsonl")
    source_splits = {
        record["id"]: SPLIT_OVERRIDES.get(record["id"], record["split"])
        for record in source_records
    }
    page_labels = sorted((database / "labels" / "pages" / "tab_only").glob("*/*.json"))
    page_labels.extend(sorted((database / "labels" / "pages" / "score_tab_symbols").glob("*/*.json")))
    if not page_labels:
        raise FileNotFoundError("No page-level TAB labels found. Run build_tuxguitar_page_annotations.ps1 first.")

    records_by_split: dict[str, list[dict]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    source_counts: dict[str, set[str]] = defaultdict(set)
    for page_label_path in page_labels:
        page_label = json.loads(page_label_path.read_text(encoding="utf-8"))
        source_id = page_label["source_id"]
        layout = page_label.get("layout", "tab_only")
        split = source_splits[source_id]
        source_counts[split].add(source_id)
        page_image_path = database / page_label["image"]
        with Image.open(page_image_path) as opened:
            page_image = opened.convert("L")

        for measure in page_label["measures"]:
            base_id = (
                f"{source_id}_{layout}_p{page_label['page_index']:03d}_"
                f"m{measure['measure_number']:03d}"
            )
            tiles = tile_measure(page_image, measure["tab_staff"]["bbox"], measure["symbols"])
            for tile_index, (image, symbols, transform) in enumerate(tiles):
                sample_id = f"{base_id}_w{tile_index:02d}"
                image_path = dataset_root / split / "images" / f"{sample_id}.png"
                label_path = dataset_root / split / "labels" / f"{sample_id}.json"
                # These generated crops are disposable training data. Low PNG
                # compression is substantially faster than the optimizer and
                # leaves pixel values unchanged.
                temporary_image = image_path.with_suffix(f".png.tmp.{os.getpid()}")
                image.save(temporary_image, format="PNG", compress_level=1)
                temporary_image.replace(image_path)
                for symbol in symbols:
                    class_counts[symbol["class"]] += 1

                label = {
                    "schema_version": "1.1",
                    "sample_id": sample_id,
                    "source_id": source_id,
                    "layout": layout,
                    "split": split,
                    "page_index": page_label["page_index"],
                    "measure_index": measure["measure_index"],
                    "measure_number": measure["measure_number"],
                    "tile_index": tile_index,
                    "source_page_image": page_label["image"],
                    "image_size": [INPUT_WIDTH, INPUT_HEIGHT],
                    "transform": transform,
                    "symbols": symbols,
                }
                label_text = json.dumps(label, ensure_ascii=False, indent=2) + "\n"
                temporary_label = label_path.with_suffix(f".json.tmp.{os.getpid()}")
                temporary_label.write_text(label_text, encoding="utf-8")
                temporary_label.replace(label_path)
                records_by_split[split].append(
                    {
                        "sample_id": sample_id,
                        "source_id": source_id,
                        "layout": layout,
                        "split": split,
                        "image": image_path.relative_to(database).as_posix(),
                        "label": label_path.relative_to(database).as_posix(),
                        "symbol_count": len(symbols),
                    }
                )

    for split in ("train", "validation", "test"):
        manifest_path = manifest_root / f"{split}.jsonl"
        manifest_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records_by_split[split]
            ),
            encoding="utf-8",
        )

    (manifest_root / "classes.json").write_text(
        json.dumps({"classes": CLASSES, "class_to_index": {name: i for i, name in enumerate(CLASSES)}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "1.1",
        "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
        "class_count": len(CLASSES),
        "classes": CLASSES,
        "source_split_overrides": SPLIT_OVERRIDES,
        "splits": {
            split: {
                "sources": len(source_counts[split]),
                "tiles": len(records_by_split[split]),
                "symbols": sum(record["symbol_count"] for record in records_by_split[split]),
            }
            for split in ("train", "validation", "test")
        },
        "class_counts": dict(sorted(class_counts.items())),
    }
    (manifest_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
