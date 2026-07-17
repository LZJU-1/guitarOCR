from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from guitarocr.paths import PROJECT_ROOT


SUPPORTED_GP8_VERSION = "8.1.2.37"
SUPPORTED_GP8_EXE_SIZE = 33_815_040
SUPPORTED_GP8_EXE_SHA256 = "F9607B932DD0F0DF6D37603CB548AFEBB39CFA6DF178068C2C8B5C7E1A2F5657"
SUPPORTED_INJECT_DLL_SHA256 = "7959120FF051F46A81C3E68CE3C118F43DA35FA2E5CC654F31071F1E3B6DF639"
DISPLAY_MODES = {"tab", "notation", "both"}


@dataclass(frozen=True)
class GuitarProDatagenRuntime:
    root: Path
    python: Path
    executable: Path
    inject_dll: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _configured_root(root: Path | None = None) -> Path:
    if root is not None:
        return root.expanduser().resolve()
    configured = os.environ.get("GUITAROCR_GP8_DATAGEN_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (PROJECT_ROOT / "guitar-hero-main").resolve()


def _configured_python(root: Path, python: Path | None = None) -> Path:
    if python is not None:
        return python.expanduser().resolve()
    configured = os.environ.get("GUITAROCR_GP8_PYTHON")
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return Path(sys.executable).resolve()


def require_guitarpro_datagen_runtime(
    root: Path | None = None,
    *,
    python: Path | None = None,
    verify_hashes: bool = True,
) -> GuitarProDatagenRuntime:
    if os.name != "nt":
        raise RuntimeError("The Guitar Pro 8.1.2.37 export worker is Windows-only.")

    resolved_root = _configured_root(root)
    executable = resolved_root / "datagen" / "vendor" / "gp8_runtime" / "GuitarPro.exe"
    inject_dll = resolved_root / "datagen" / "gt2pdf" / "bin" / "gt2pdf_inject.dll"
    cli = resolved_root / "datagen" / "gt2pdf" / "cli.py"
    resolved_python = _configured_python(resolved_root, python)
    missing = [path for path in (executable, inject_dll, cli, resolved_python) if not path.is_file()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The local Guitar Pro datagen runtime is incomplete. Missing:\n"
            f"{joined}\nSet GUITAROCR_GP8_DATAGEN_ROOT and GUITAROCR_GP8_PYTHON if needed."
        )

    if executable.stat().st_size != SUPPORTED_GP8_EXE_SIZE:
        raise RuntimeError(
            f"Unsupported GuitarPro.exe size: {executable.stat().st_size} bytes. "
            f"This adapter is pinned to Guitar Pro {SUPPORTED_GP8_VERSION} "
            f"({SUPPORTED_GP8_EXE_SIZE} bytes)."
        )
    if verify_hashes:
        executable_hash = _sha256(executable)
        dll_hash = _sha256(inject_dll)
        if executable_hash != SUPPORTED_GP8_EXE_SHA256:
            raise RuntimeError(
                f"GuitarPro.exe SHA-256 mismatch: {executable_hash}. "
                f"Expected the project-bundled Guitar Pro {SUPPORTED_GP8_VERSION} binary."
            )
        if dll_hash != SUPPORTED_INJECT_DLL_SHA256:
            raise RuntimeError(
                f"gt2pdf_inject.dll SHA-256 mismatch: {dll_hash}. "
                "The injector and Guitar Pro build must stay paired."
            )

    return GuitarProDatagenRuntime(
        root=resolved_root,
        python=resolved_python,
        executable=executable,
        inject_dll=inject_dll,
    )


def export_gp_with_guitarpro(
    input_path: Path,
    output_pdf: Path,
    *,
    layout_json: Path,
    ready_timeout: int = 90,
    root: Path | None = None,
    python: Path | None = None,
    verify_hashes: bool = True,
) -> subprocess.CompletedProcess[str]:
    runtime = require_guitarpro_datagen_runtime(
        root,
        python=python,
        verify_hashes=verify_hashes,
    )
    source = input_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input GP file does not exist: {source}")
    pdf = output_pdf.expanduser().resolve()
    layout = layout_json.expanduser().resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    layout.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(runtime.python),
        "-m",
        "datagen.gt2pdf.cli",
        "convert",
        str(source),
        str(pdf),
        "--layout-json",
        str(layout),
        "--ready-timeout",
        str(max(1, ready_timeout)),
    ]
    return subprocess.run(
        command,
        cwd=runtime.root,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def prepare_gp5_display_mode(
    input_path: Path,
    output_path: Path,
    display_mode: str,
    *,
    root: Path | None = None,
    python: Path | None = None,
    verify_hashes: bool = True,
) -> subprocess.CompletedProcess[str]:
    mode = display_mode.strip().lower()
    if mode not in DISPLAY_MODES:
        raise ValueError(f"display_mode must be one of {sorted(DISPLAY_MODES)}, got {display_mode!r}")
    runtime = require_guitarpro_datagen_runtime(
        root,
        python=python,
        verify_hashes=verify_hashes,
    )
    source = input_path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input GP file does not exist: {source}")
    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    program = (
        "import sys, guitarpro; "
        "from datagen.gp5 import parse_gp_song, set_display_mode; "
        "song,encoding=parse_gp_song(sys.argv[1]); "
        "set_display_mode(song, sys.argv[3]); "
        "guitarpro.write(song, sys.argv[2], version=(5,1,0), encoding=encoding)"
    )
    return subprocess.run(
        [str(runtime.python), "-c", program, str(source), str(output), mode],
        cwd=runtime.root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
