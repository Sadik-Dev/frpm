"""LoRA caption generation.

Per the spec: caption CHANGEABLE attributes (clothing colour, expression,
gaze direction) and deliberately avoid over-captioning permanent identity
traits - the LoRA should learn the person's face naturally from the trigger
token alone.

Scene/setting captions (indoor vs outdoor, "car interior", etc.) would need
a dedicated scene-classification model, which is out of scope here (it adds
license/dependency surface for a "nice to have" fragment, and a wrong guess
actively hurts training more than an absent one helps) - so this covers
clothing colour + expression + gaze only. Review/edit captions before
training if you need scene-level detail.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .models import Candidate

EXPRESSION_PHRASES: dict[str, Optional[str]] = {
    "neutral": None,
    "slight_smile": "slight smile",
    "confident_smile": "confident smile",
    "big_smile": "big smile",
    "laughing": "laughing",
    "serious": "serious expression",
    "focused": "focused expression",
    "surprised": "surprised expression",
    "mouth_open": "mouth open",
    "talking": "talking",
    "looking_down": "looking down",
    "looking_up": "looking up",
    "eyes_left": "looking left",
    "eyes_right": "looking right",
    "eyes_closed": "eyes closed",
    "raised_eyebrows": "raised eyebrows",
    "frowning": "frowning",
}

_COLOR_PALETTE = {
    "black": (10, 10, 10),
    "white": (245, 245, 245),
    "gray": (128, 128, 128),
    "red": (200, 30, 30),
    "blue": (30, 60, 200),
    "green": (40, 140, 60),
    "yellow": (220, 200, 40),
    "orange": (230, 130, 30),
    "purple": (130, 50, 160),
    "pink": (230, 140, 170),
    "brown": (110, 70, 40),
}


def guess_clothing_color(image: np.ndarray, bbox) -> Optional[str]:
    """Very rough dominant-colour guess of the region just below the face
    (shoulders/torso), used only as an optional caption fragment."""
    h, w = image.shape[:2]
    y1 = min(h, bbox.y + int(bbox.h * 1.3))
    y2 = min(h, y1 + int(bbox.h * 1.2))
    x1 = max(0, bbox.x - int(bbox.w * 0.3))
    x2 = min(w, bbox.x + bbox.w + int(bbox.w * 0.3))
    if y2 <= y1 or x2 <= x1:
        return None
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return None
    mean_bgr = region.reshape(-1, 3).mean(axis=0)
    mean_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
    best_name, best_dist = None, float("inf")
    for name, rgb in _COLOR_PALETTE.items():
        dist = float(np.linalg.norm(mean_rgb - np.array(rgb)))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def build_caption(candidate: Candidate, trigger_token: str, image: Optional[np.ndarray] = None) -> str:
    parts = [trigger_token]
    if candidate.expressions:
        for tag in candidate.expressions.tags:
            phrase = EXPRESSION_PHRASES.get(tag)
            if phrase:
                parts.append(phrase)
    if image is not None:
        color = guess_clothing_color(image, candidate.bbox)
        if color:
            parts.append(f"{color} shirt")
    return ", ".join(parts)
