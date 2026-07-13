from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from guitarocr.pipeline.layout_classifier import classify_notation_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a page as score_tab, tab_only, or score_only")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    with Image.open(args.image) as opened:
        result = classify_notation_layout(opened.convert("L"))
    result["image"] = str(args.image.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
