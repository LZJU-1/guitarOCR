from __future__ import annotations

import os
from pathlib import Path
import shutil

from guitarocr.paths import TUXGUITAR_ROOT


def require_tuxguitar_root() -> Path:
    root = TUXGUITAR_ROOT.resolve()
    if not (root / "lib" / "tuxguitar.jar").is_file():
        raise FileNotFoundError(
            "TuxGuitar runtime was not found. Run scripts/setup_windows.ps1 or set "
            f"GUITAROCR_TUXGUITAR_ROOT. Checked: {root}"
        )
    return root


def java_classpath(classes: Path) -> str:
    root = require_tuxguitar_root()
    return os.pathsep.join(
        [
            str(classes),
            str(root / "lib" / "*"),
            str(root / "share" / "plugins" / "*"),
            str(root / "share"),
            str(root / "dist"),
        ]
    )


def java_executable() -> Path:
    root = require_tuxguitar_root()
    for name in ("java.exe", "java"):
        bundled = root / "jre" / "bin" / name
        if bundled.is_file():
            return bundled
    configured = os.environ.get("GUITAROCR_JAVA")
    located = configured or shutil.which("java")
    if located:
        return Path(located)
    raise FileNotFoundError("Java was not found in TuxGuitar or PATH.")


def javac_executable() -> Path:
    configured = os.environ.get("GUITAROCR_JAVAC")
    located = configured or shutil.which("javac")
    if located:
        return Path(located)
    raise FileNotFoundError(
        "javac was not found. Install JDK 17+ or set GUITAROCR_JAVAC to javac."
    )
