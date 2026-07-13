from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import PIL
import numpy
import torch
import torchvision

from guitarocr.paths import DATABASE_ROOT, PROJECT_ROOT, TUXGUITAR_ROOT, WEIGHTS_ROOT
from guitarocr.pipeline.pdf_page_renderer import find_poppler_program
from guitarocr.tuxguitar_runtime import javac_executable, require_tuxguitar_root


REQUIRED_WEIGHTS = (
    "atomic_symbol_cnn.pt",
    "rhythm_context_cnn.pt",
    "score_event_locator.pt",
    "tab_symbol_detector.pt",
    "tie_context_cnn.pt",
)


def checked(name: str, operation) -> dict:
    try:
        value = operation()
        return {"name": name, "ok": True, "value": str(value)}
    except Exception as exc:  # The command is an environment diagnostic.
        return {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def require_weights() -> Path:
    missing = [filename for filename in REQUIRED_WEIGHTS if not (WEIGHTS_ROOT / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing models under {WEIGHTS_ROOT}: {', '.join(missing)}")
    return WEIGHTS_ROOT


def main() -> None:
    DATABASE_ROOT.mkdir(parents=True, exist_ok=True)
    checks = [
        checked("weights", require_weights),
        checked("pdftoppm", lambda: find_poppler_program("pdftoppm")),
        checked("pdfinfo", lambda: find_poppler_program("pdfinfo")),
        checked("tuxguitar", require_tuxguitar_root),
        checked("javac", javac_executable),
    ]
    report = {
        "ok": all(item["ok"] for item in checks),
        "project_root": str(PROJECT_ROOT),
        "weights_root": str(WEIGHTS_ROOT),
        "database_root": str(DATABASE_ROOT),
        "tuxguitar_root": str(TUXGUITAR_ROOT),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "opencv": cv2.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "checks": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
