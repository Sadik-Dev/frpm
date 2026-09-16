"""Crop generation: face reference crops (square, head+hair+neck), portrait
crops (4:5, head+shoulders+chest where framing allows), and body crops.

Never distorts aspect ratio: uses edge-reflect padding for small overflows
and refuses to fabricate missing content (returns ``None``) when the source
frame simply doesn't show enough of the person.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .models import BBox


def _crop_with_reflect_pad(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    pad_left, pad_top = max(0, -x1), max(0, -y1)
    pad_right, pad_bottom = max(0, x2 - w), max(0, y2 - h)
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return crop
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
    return crop


def make_face_crop(image: np.ndarray, bbox: BBox, out_size: int = 1024, expand: float = 2.6) -> np.ndarray:
    """Square crop around head+hair+neck. ``expand`` > 1 because the raw
    detector bbox is tight to facial features, not the whole head."""
    import cv2

    cx, cy = bbox.center
    side = max(bbox.w, bbox.h) * expand
    cy -= bbox.h * 0.15  # nudge up to capture hair/forehead rather than centering on facial features
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    crop = _crop_with_reflect_pad(image, x1, y1, x2, y2)
    if crop.size == 0:
        return crop
    interp = cv2.INTER_AREA if crop.shape[0] > out_size else cv2.INTER_CUBIC
    return cv2.resize(crop, (out_size, out_size), interpolation=interp)


def make_portrait_crop(
    image: np.ndarray,
    bbox: BBox,
    out_width: int = 1024,
    aspect_w: int = 4,
    aspect_h: int = 5,
    expand_w: float = 2.4,
    min_available_ratio: float = 0.7,
) -> Optional[np.ndarray]:
    """4:5 head+shoulders+chest crop, or ``None`` if the source frame
    doesn't show enough of the person below/around the face to make one
    honestly (per spec: never hallucinate missing body area)."""
    import cv2

    h, _w = image.shape[:2]
    cx, _cy = bbox.center
    crop_w = bbox.w * expand_w
    crop_h = crop_w * aspect_h / aspect_w
    top = bbox.y - bbox.h * 0.9
    bottom = top + crop_h
    x1, x2 = cx - crop_w / 2, cx + crop_w / 2

    available_bottom = min(bottom, h)
    available_height = available_bottom - max(top, 0)
    if crop_h <= 0 or available_height / crop_h < min_available_ratio:
        return None

    crop = _crop_with_reflect_pad(image, int(round(x1)), int(round(top)), int(round(x2)), int(round(bottom)))
    if crop.size == 0:
        return None
    out_h = int(round(out_width * aspect_h / aspect_w))
    interp = cv2.INTER_AREA if crop.shape[0] > out_h else cv2.INTER_CUBIC
    return cv2.resize(crop, (out_width, out_h), interpolation=interp)


def make_body_crop(image: np.ndarray, bbox: BBox, out_width: int = 1024) -> np.ndarray:
    """Looser crop for upper/half/full body references: wide horizontal
    context, full frame height, still centered on the person."""
    import cv2

    h, w = image.shape[:2]
    cx, _cy = bbox.center
    half_w = min(w, bbox.w * 4.0) / 2
    x1, x2 = int(round(cx - half_w)), int(round(cx + half_w))
    crop = _crop_with_reflect_pad(image, x1, 0, x2, h)
    if crop.size == 0:
        return crop
    scale = out_width / crop.shape[1]
    out_h = max(1, int(round(crop.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(crop, (out_width, out_h), interpolation=interp)
