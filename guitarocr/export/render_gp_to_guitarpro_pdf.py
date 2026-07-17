from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from guitarocr.guitarpro_runtime import (
    SUPPORTED_GP8_VERSION,
    export_gp_with_guitarpro,
    prepare_gp5_display_mode,
    require_guitarpro_datagen_runtime,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render one GP file with the project-bundled Guitar Pro 8 worker and export native layout JSON. "
            "The worker terminates existing GuitarPro.exe processes before it starts."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--layout-json",
        type=Path,
        help="output native layout JSON (default: PDF name with .layout.json suffix)",
    )
    parser.add_argument("--datagen-root", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--ready-timeout", type=int, default=90)
    parser.add_argument(
        "--display-mode",
        choices=("tab", "notation", "both"),
        help=(
            "rewrite a GP3/GP4/GP5 input to the requested Guitar Pro display mode "
            "before the official GP8 PDF export"
        ),
    )
    parser.add_argument(
        "--allow-unverified-runtime",
        action="store_true",
        help="skip SHA-256 verification; unsafe for a build not matched to the injector",
    )
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    layout_json = (
        args.layout_json.expanduser().resolve()
        if args.layout_json
        else output.with_suffix(".layout.json")
    )
    runtime = require_guitarpro_datagen_runtime(
        args.datagen_root,
        python=args.python,
        verify_hashes=not args.allow_unverified_runtime,
    )
    print(f"Using Guitar Pro {SUPPORTED_GP8_VERSION}: {runtime.executable}")
    with tempfile.TemporaryDirectory(prefix="guitarocr_gp8_display_") as temporary:
        export_input = args.input
        if args.display_mode:
            export_input = Path(temporary) / f"prepared_{args.display_mode}.gp5"
            prepared = prepare_gp5_display_mode(
                args.input,
                export_input,
                args.display_mode,
                root=runtime.root,
                python=runtime.python,
                verify_hashes=False,
            )
            if prepared.returncode:
                detail = (prepared.stdout or "") + (prepared.stderr or "")
                raise SystemExit(
                    f"Could not prepare Guitar Pro display mode {args.display_mode!r}: {detail}"
                )
        completed = export_gp_with_guitarpro(
            export_input,
            output,
            layout_json=layout_json,
            ready_timeout=args.ready_timeout,
            root=runtime.root,
            python=runtime.python,
            verify_hashes=False,
        )
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"PDF: {output}")
    print(f"Layout: {layout_json}")


if __name__ == "__main__":
    main()
