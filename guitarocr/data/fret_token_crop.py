from __future__ import annotations

from PIL import Image

from guitarocr.models.fret_token_model import INPUT_HEIGHT, INPUT_WIDTH


def crop_fret_token(
    page: Image.Image,
    event_x: float,
    string_y: float,
    spacing: float,
) -> Image.Image:
    crop_width = max(20, round(spacing * 2.5))
    crop_height = max(12, round(spacing * 1.05))
    left = round(event_x - crop_width / 2)
    top = round(string_y - crop_height / 2)
    source = page.convert("L")
    output = Image.new("L", (crop_width, crop_height), 255)
    source_box = (
        max(0, left),
        max(0, top),
        min(source.width, left + crop_width),
        min(source.height, top + crop_height),
    )
    if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
        output.paste(source.crop(source_box), (source_box[0] - left, source_box[1] - top))
    return output.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.LANCZOS)
