"""Turn stroke data into a grayscale image.

Quick, Draw! "simplified" strokes are already scaled to a 0-255 box. We draw
them at 4x resolution and downsample, which gives smooth (anti-aliased) lines.
This is the only place strokes become pixels: training, evaluation and the
future drawing UI should all use it.
"""
import numpy as np
from PIL import Image, ImageDraw

from .config import IMG_SIZE


def render(strokes, size=IMG_SIZE):
    """strokes: list of [xs, ys] in 0..255. Returns uint8 array (size, size), ink = 255."""
    big = size * 4
    scale = (big - 1) / 255.0
    img = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(img)
    for xs, ys in strokes:
        pts = [(x * scale, y * scale) for x, y in zip(xs, ys)]
        if len(pts) == 1:
            pts = pts * 2
        draw.line(pts, fill=255, width=8, joint="curve")
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.uint8)
