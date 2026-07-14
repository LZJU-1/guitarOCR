from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from guitarocr.paths import JAVA_SOURCE_ROOT, PROJECT_ROOT
from guitarocr.tuxguitar_runtime import (
    java_classpath,
    java_executable,
    javac_executable,
    require_tuxguitar_root,
)


def main() -> None:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description="Render a GP/GPX/GP3-5 song as a TuxGuitar PDF.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--layout",
        choices=("score_tab", "tab_only", "score_only"),
        default="score_tab",
        help="notation layout to render (default: score_tab)",
    )
    args = parser.parse_args()
    source = JAVA_SOURCE_ROOT / "TuxGuitarPdfRenderer.java"
    classes = root / "database" / "tmp" / "gp_pdf_renderer_classes"
    classes.mkdir(parents=True, exist_ok=True)
    classpath = java_classpath(classes)
    target = classes / "TuxGuitarPdfRenderer.class"
    if not target.is_file() or target.stat().st_mtime < source.stat().st_mtime:
        compiled = subprocess.run(
            [str(javac_executable()), "-encoding", "UTF-8", "-cp", classpath, "-d", str(classes), str(source)],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if compiled.returncode:
            print(compiled.stdout + compiled.stderr, file=sys.stderr)
            raise SystemExit(compiled.returncode)
    java = java_executable()
    tuxguitar_root = require_tuxguitar_root()
    completed = subprocess.run(
        [
            str(java),
            "-Xmx3g",
            "-Djava.awt.headless=true",
            f"-Dtuxguitar.home.path={tuxguitar_root}",
            "-cp",
            classpath,
            "TuxGuitarPdfRenderer",
            str(args.input.resolve()),
            str(args.output.resolve()),
            args.layout,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout_encoding = sys.stdout.encoding or "utf-8"
    print(
        completed.stdout.encode(stdout_encoding, errors="replace").decode(stdout_encoding),
        end="",
    )
    if completed.returncode:
        stderr_encoding = sys.stderr.encoding or "utf-8"
        print(
            completed.stderr.encode(stderr_encoding, errors="replace").decode(stderr_encoding),
            end="",
            file=sys.stderr,
        )
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
