from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


IMAGE_SIZE = 64
SPLIT_COUNTS = {"train": 400, "validation": 80, "test": 80}


SEMANTICS = {
    **{
        f"digit_{digit}": {
            "category": "digit",
            "meaning": f"Numeric glyph {digit}; used in TAB frets, time signatures, tempo, and tuplets.",
            "limitation": "Multi-digit values are assembled later from adjacent digit instances.",
        }
        for digit in range(10)
    },
    "dead_x": {
        "category": "note",
        "meaning": "X-shaped dead/muted note mark in TAB or score notation.",
        "limitation": "TAB versus score meaning is resolved from staff context.",
    },
    "notehead_filled": {
        "category": "notehead",
        "meaning": "Filled notehead used by quarter and shorter note values.",
        "limitation": "Exact duration requires stem, beam, flag, dot, and tuplet relationships.",
    },
    "notehead_open": {
        "category": "notehead",
        "meaning": "Open notehead used by whole and half notes.",
        "limitation": "Whole versus half duration is determined by presence of a stem.",
    },
    "notehead_harmonic": {
        "category": "notehead",
        "meaning": "Diamond harmonic notehead.",
        "limitation": "Natural/artificial harmonic detail requires neighboring text/effect context.",
    },
    "rest_block": {
        "category": "rest",
        "meaning": "Rectangular whole-or-half rest primitive.",
        "limitation": "The two durations have the same primitive shape; staff-relative vertical position resolves them.",
    },
    "rest_quarter": {"category": "rest", "meaning": "Quarter rest.", "limitation": ""},
    "rest_eighth": {"category": "rest", "meaning": "Eighth rest.", "limitation": ""},
    "rest_sixteenth": {"category": "rest", "meaning": "Sixteenth rest.", "limitation": ""},
    "rest_thirty_second": {"category": "rest", "meaning": "Thirty-second rest.", "limitation": ""},
    "rest_sixty_fourth": {"category": "rest", "meaning": "Sixty-fourth rest.", "limitation": ""},
    "accidental_sharp": {"category": "accidental", "meaning": "Sharp accidental.", "limitation": "Pitch scope is resolved in the event parser."},
    "accidental_flat": {"category": "accidental", "meaning": "Flat accidental.", "limitation": "Pitch scope is resolved in the event parser."},
    "accidental_natural": {"category": "accidental", "meaning": "Natural accidental.", "limitation": "Pitch scope is resolved in the event parser."},
    "clef_treble": {"category": "clef", "meaning": "Treble clef.", "limitation": ""},
    "clef_bass": {"category": "clef", "meaning": "Bass clef.", "limitation": ""},
    "clef_c": {
        "category": "clef",
        "meaning": "Generic C-clef primitive.",
        "limitation": "Alto versus tenor meaning is determined by its vertical position on the staff, not by isolated shape.",
    },
    "clef_neutral": {"category": "clef", "meaning": "Neutral/percussion clef.", "limitation": "Percussion tracks are excluded from the first page dataset."},
    "dot": {
        "category": "dot",
        "meaning": "Generic dot primitive.",
        "limitation": "Augmentation, repeat, and staccato meanings are distinguished by spatial context.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic synthetic atomic-symbol dataset.")
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--train-per-class", type=int, default=SPLIT_COUNTS["train"])
    parser.add_argument("--validation-per-class", type=int, default=SPLIT_COUNTS["validation"])
    parser.add_argument("--test-per-class", type=int, default=SPLIT_COUNTS["test"])
    return parser.parse_args()


def class_name(template: Path) -> str:
    return template.stem.split("__", 1)[0]


def target_extent(name: str, rng: random.Random) -> int:
    if name.startswith("digit_") or name == "dead_x":
        return rng.randint(22, 38)
    if name.startswith("clef_"):
        return rng.randint(38, 56)
    if name == "dot":
        return rng.randint(7, 15)
    if name.startswith("accidental_"):
        return rng.randint(28, 47)
    if name.startswith("rest_"):
        return rng.randint(30, 50)
    return rng.randint(24, 42)


def make_ink_mask(template: Image.Image, extent: int, rng: random.Random) -> Image.Image:
    grayscale = template.convert("L")
    ink = Image.eval(grayscale, lambda value: 255 - value)
    bbox = ink.getbbox()
    if not bbox:
        raise ValueError("Empty symbol template")
    ink = ink.crop(bbox)
    scale = extent / max(ink.width, ink.height)
    width = max(2, round(ink.width * scale * rng.uniform(0.92, 1.08)))
    height = max(2, round(ink.height * scale * rng.uniform(0.94, 1.06)))
    ink = ink.resize((width, height), Image.Resampling.LANCZOS)
    if rng.random() < 0.22:
        ink = ink.filter(ImageFilter.MaxFilter(3))
    angle = rng.uniform(-5.0, 5.0)
    return ink.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=0)


def draw_staff_background(canvas: Image.Image, name: str, rng: random.Random) -> list[int]:
    if rng.random() >= 0.72:
        return []
    draw = ImageDraw.Draw(canvas)
    line_count = 6 if name.startswith("digit_") or name == "dead_x" else 5
    spacing = rng.randint(7, 11)
    center = IMAGE_SIZE // 2 + rng.randint(-5, 5)
    first = round(center - ((line_count - 1) * spacing / 2))
    color = rng.randint(55, 145)
    width = 1 if rng.random() < 0.82 else 2
    y_values = []
    for index in range(line_count):
        y = first + index * spacing
        y_values.append(y)
        draw.line((0, y, IMAGE_SIZE - 1, y), fill=color, width=width)
    return y_values


def synthesize(template: Image.Image, name: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=rng.randint(244, 255))
    staff_lines = draw_staff_background(canvas, name, rng)
    mask = make_ink_mask(template, target_extent(name, rng), rng)

    x = (IMAGE_SIZE - mask.width) // 2 + rng.randint(-8, 8)
    y = (IMAGE_SIZE - mask.height) // 2 + rng.randint(-8, 8)
    x = max(-mask.width // 5, min(IMAGE_SIZE - (4 * mask.width // 5), x))
    y = max(-mask.height // 5, min(IMAGE_SIZE - (4 * mask.height // 5), y))

    # TuxGuitar erases the TAB line immediately behind a printed fret label.
    if staff_lines and (name.startswith("digit_") or name == "dead_x"):
        knockout = ImageDraw.Draw(canvas)
        knockout.rectangle((x - 1, y - 1, x + mask.width + 1, y + mask.height + 1), fill=rng.randint(247, 255))

    ink_value = rng.randint(0, 45)
    ink_layer = Image.new("L", canvas.size, color=ink_value)
    positioned_mask = Image.new("L", canvas.size, color=0)
    positioned_mask.paste(mask, (x, y))
    canvas = Image.composite(ink_layer, canvas, positioned_mask)

    if rng.random() < 0.55:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.12, 0.75)))
    array = np.asarray(canvas, dtype=np.float32)
    if rng.random() < 0.65:
        array += np.random.default_rng(seed ^ 0x5A17).normal(0.0, rng.uniform(0.5, 5.5), array.shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")


def main() -> None:
    args = parse_args()
    templates = sorted(args.templates.glob("*.png"))
    if not templates:
        raise SystemExit(f"No PNG templates found in {args.templates}")

    grouped: dict[str, list[Path]] = {}
    for template in templates:
        grouped.setdefault(class_name(template), []).append(template)
    missing_semantics = sorted(set(grouped) - set(SEMANTICS))
    if missing_semantics:
        raise SystemExit(f"Missing semantic descriptions: {missing_semantics}")

    output_resolved = args.output.resolve()
    symbol_root = args.templates.resolve().parent
    if output_resolved.parent != symbol_root or output_resolved.name != "dataset":
        raise SystemExit(f"Refusing to build outside the expected symbol_cnn/dataset directory: {output_resolved}")
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    classes = sorted(grouped)
    class_to_index = {name: index for index, name in enumerate(classes)}
    counts = {
        "train": args.train_per_class,
        "validation": args.validation_per_class,
        "test": args.test_per_class,
    }
    manifest_rows: list[dict[str, object]] = []
    split_offsets = {"train": 0, "validation": 1_000_000, "test": 2_000_000}

    loaded_templates = {path: Image.open(path).convert("L") for path in templates}
    for split, count in counts.items():
        for name in classes:
            destination = args.output / split / name
            destination.mkdir(parents=True)
            variants = grouped[name]
            for sample_index in range(count):
                seed = args.seed + split_offsets[split] + class_to_index[name] * 10_000 + sample_index
                template_path = variants[sample_index % len(variants)]
                image = synthesize(loaded_templates[template_path], name, seed)
                filename = f"{sample_index:05d}.png"
                output_path = destination / filename
                image.save(output_path, compress_level=3)
                manifest_rows.append(
                    {
                        "relative_path": output_path.relative_to(args.output).as_posix(),
                        "split": split,
                        "class_index": class_to_index[name],
                        "class_name": name,
                        "template": template_path.name,
                        "seed": seed,
                    }
                )

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    class_records = []
    for name in classes:
        record = {"index": class_to_index[name], "name": name, **SEMANTICS[name]}
        record["templates"] = [path.name for path in grouped[name]]
        class_records.append(record)
    (args.output / "classes.json").write_text(
        json.dumps(class_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "schema_version": "1.0",
        "seed": args.seed,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "class_count": len(classes),
        "template_count": len(templates),
        "split_counts_per_class": counts,
        "total_images": len(manifest_rows),
        "source": "TuxGuitar 2.0.1 vector painters plus deterministic print-like augmentation",
        "scope": "atomic symbol classification; not page detection or relationship parsing",
    }
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
