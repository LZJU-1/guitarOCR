from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from guitarocr.data.musescore_layout import LAYOUTS, convert_mscx_layout
from guitarocr.paths import DATABASE_ROOT, PROJECT_ROOT


RENDERERS = ("tuxguitar", "musescore", "guitarpro")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def find_musescore(explicit: Path | None) -> Path | None:
    candidates = [
        explicit,
        Path(os.environ["GUITAROCR_MUSESCORE_EXE"])
        if os.environ.get("GUITAROCR_MUSESCORE_EXE") else None,
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe"),
    ]
    command = shutil.which("MuseScore4") or shutil.which("mscore")
    if command:
        candidates.append(Path(command))
    return next((path.resolve() for path in candidates if path and path.is_file()), None)


def select_sources(database: Path, count: int) -> tuple[Path, list[dict]]:
    candidates = [
        database / "v2" / "manifests" / "sources.jsonl",
        database / "manifests" / "sources.jsonl",
    ]
    manifest = next((path for path in candidates if path.is_file()), None)
    if manifest is None:
        raise FileNotFoundError("No source manifest was found; build the GuitarOCR database first")
    source_root = manifest.parent.parent
    eligible = []
    for row in read_jsonl(manifest):
        source = source_root / row["source_gp"]
        label = source_root / row["label_json"]
        if (
            row.get("split") == "test"
            and int(row.get("track_count", 0)) == 1
            and row.get("source_format") in {"gp3", "gp4", "gp5"}
            and source.is_file()
            and label.is_file()
        ):
            metadata = json.loads(label.read_text(encoding="utf-8"))
            strings = int(metadata.get("target_track", {}).get("string_count", 0))
            if 4 <= strings <= 8:
                eligible.append({**row, "_source": source, "_label": label, "_strings": strings})
    by_format: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_format[row["source_format"]].append(row)
    for rows in by_format.values():
        rows.sort(
            key=lambda row: (
                int(row.get("effect_note_count", 0)),
                int(row.get("multi_digit_fret_count", 0)),
                int(row.get("note_count", 0)),
            ),
            reverse=True,
        )
    selected = []
    while len(selected) < count and any(by_format.values()):
        for source_format in ("gp3", "gp4", "gp5"):
            if by_format[source_format] and len(selected) < count:
                selected.append(by_format[source_format].pop(0))
    if not selected:
        raise RuntimeError("No single-track source-disjoint test songs are available")
    return source_root, selected


def run(command: list[str], *, timeout: int = 300) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}{completed.stderr}"
        )


def render_tuxguitar(source: Path, output: Path, layout: str, overwrite: bool) -> None:
    if output.is_file() and output.stat().st_size and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        "-m",
        "guitarocr.export.render_gp_to_pdf",
        str(source),
        str(output),
        "--layout",
        layout,
    ])


def render_musescore(
    executable: Path,
    source: Path,
    work_root: Path,
    output: Path,
    layout: str,
    overwrite: bool,
) -> None:
    if output.is_file() and output.stat().st_size and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    base = work_root / f"{source.stem}_imported.mscx"
    converted = work_root / f"{source.stem}_{layout}.mscx"
    if overwrite or not base.is_file():
        run([str(executable), "-o", str(base), str(source)])
    convert_mscx_layout(base, converted, layout)
    run([str(executable), "-o", str(output), str(converted)])


def pdf_record(
    *,
    source: dict,
    renderer: str,
    layout: str,
    pdf: Path,
    generation: str,
    error: str | None = None,
) -> dict:
    ready = pdf.is_file() and pdf.stat().st_size > 0
    return {
        "schema_version": "1.0",
        "sample_id": f"{source['id']}_{renderer}_{layout}",
        "source_id": source["id"],
        "source_split": source["split"],
        "source_format": source["source_format"],
        "source_gp": manifest_path(source["_source"]),
        "reference_label": manifest_path(source["_label"]),
        "target_track_number": int(source["target_track_number"]),
        "string_count": int(source["_strings"]),
        "measure_count": int(source["measure_count"]),
        "renderer": renderer,
        "layout": layout,
        "generation": generation,
        "status": "ready" if ready else "missing",
        "pdf": manifest_path(pdf),
        "pdf_sha256": sha256(pdf) if ready else None,
        "error": error,
        "training_eligible": False,
        "source_disjoint_assertion": "source split is test; benchmark artifacts must not enter training",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a source-disjoint multi-renderer GuitarOCR benchmark."
    )
    parser.add_argument("--database", type=Path, default=DATABASE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-count", type=int, default=12)
    parser.add_argument("--renderers", default=",".join(RENDERERS))
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    parser.add_argument("--musescore-executable", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    database = args.database.resolve()
    output = (args.output or database / "cross_renderer_benchmark").resolve()
    renderers = tuple(value.strip() for value in args.renderers.split(",") if value.strip())
    layouts = tuple(value.strip() for value in args.layouts.split(",") if value.strip())
    invalid_renderers = sorted(set(renderers) - set(RENDERERS))
    invalid_layouts = sorted(set(layouts) - set(LAYOUTS))
    if invalid_renderers or invalid_layouts:
        parser.error(f"Invalid renderers={invalid_renderers}, layouts={invalid_layouts}")
    _, sources = select_sources(database, args.source_count)
    muse = find_musescore(args.musescore_executable)
    records = []
    for source in sources:
        for renderer in renderers:
            for layout in layouts:
                pdf = output / "raw" / renderer / layout / f"{source['id']}.pdf"
                error = None
                generation = "manual_export"
                try:
                    if renderer == "tuxguitar":
                        generation = "automatic_tuxguitar"
                        render_tuxguitar(source["_source"], pdf, layout, args.overwrite)
                    elif renderer == "musescore" and muse is not None:
                        generation = "automatic_musescore_mscx"
                        render_musescore(
                            muse,
                            source["_source"],
                            output / "work" / "musescore" / source["id"],
                            pdf,
                            layout,
                            args.overwrite,
                        )
                except Exception as exception:  # Keep a complete queue even if one renderer fails.
                    error = str(exception)
                records.append(pdf_record(
                    source=source,
                    renderer=renderer,
                    layout=layout,
                    pdf=pdf,
                    generation=generation,
                    error=error,
                ))
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "benchmark.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    manual = [row for row in records if row["status"] != "ready"]
    (output / "manual_export_queue.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in manual),
        encoding="utf-8",
    )
    counts = Counter((row["renderer"], row["layout"], row["status"]) for row in records)
    summary = {
        "schema_version": "1.0",
        "source_manifest_split": "test",
        "source_count": len(sources),
        "source_ids": [row["id"] for row in sources],
        "musescore_executable": str(muse) if muse else None,
        "records": len(records),
        "ready": sum(row["status"] == "ready" for row in records),
        "missing": len(manual),
        "counts": {
            f"{renderer}/{layout}/{status}": value
            for (renderer, layout, status), value in sorted(counts.items())
        },
        "manifest": str(manifest),
        "manual_export_queue": str(output / "manual_export_queue.jsonl"),
        "training_eligible": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
