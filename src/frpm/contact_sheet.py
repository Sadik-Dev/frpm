"""High-resolution contact sheets (identity / expression / pose) with small,
tasteful text labels - no ugly filenames burned onto faces - suitable to
hand directly to an image-generation model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .utils import ensure_dir, imread_unicode

logger = logging.getLogger("frpm.contact_sheet")

CELL_SIZE = 512
LABEL_HEIGHT = 40
PADDING = 8
BG_COLOR = (24, 24, 24)
LABEL_FG = (255, 255, 255)

_FONT_CANDIDATES = [
    "C:\\Windows\\Fonts\\segoeui.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _cell_from_path(path: str, label: str, cell_size: int = CELL_SIZE) -> Image.Image:
    import cv2

    cell = Image.new("RGB", (cell_size, cell_size + LABEL_HEIGHT), BG_COLOR)
    img_bgr = imread_unicode(path)
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((cell_size, cell_size), Image.LANCZOS)  # letterbox, no distortion
        ox, oy = (cell_size - pil_img.width) // 2, (cell_size - pil_img.height) // 2
        cell.paste(pil_img, (ox, oy))
    else:
        logger.warning("Could not load image for contact sheet cell: %s", path)
    if label:
        draw = ImageDraw.Draw(cell)
        font = _load_font(22)
        text_w = draw.textlength(label, font=font)
        tx = max(4.0, (cell_size - text_w) / 2)
        draw.text((tx, cell_size + LABEL_HEIGHT / 2 - 11), label, font=font, fill=LABEL_FG)
    return cell


def build_grid(
    items: list[tuple[str, str]],
    out_path: Path,
    columns: int = 4,
    cell_size: int = CELL_SIZE,
    title: Optional[str] = None,
) -> Optional[Path]:
    """``items`` is a list of ``(image_path, label)``. Writes one grid image
    to ``out_path``; returns ``None`` (and writes nothing) if ``items`` is
    empty."""
    if not items:
        return None
    columns = max(1, min(columns, len(items)))
    rows = (len(items) + columns - 1) // columns
    title_h = 60 if title else 0
    sheet = Image.new("RGB", (columns * cell_size, rows * (cell_size + LABEL_HEIGHT) + title_h), BG_COLOR)
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((PADDING, 14), title, font=_load_font(32), fill=LABEL_FG)
    for idx, (path, label) in enumerate(items):
        r, c = divmod(idx, columns)
        sheet.paste(_cell_from_path(path, label, cell_size), (c * cell_size, title_h + r * (cell_size + LABEL_HEIGHT)))
    ensure_dir(out_path.parent)
    sheet.save(out_path, quality=95)
    return out_path
