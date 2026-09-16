"""Practical expression tagging from MediaPipe face blendshapes.

Deliberately avoids generic emotion labels (happy/sad/angry) in favour of
image-generation-useful, multi-label tags. An image can carry several tags
at once, e.g. ``["slight_smile", "eyes_right"]``.
"""
from __future__ import annotations

from .config import ExpressionThresholds
from .models import ExpressionRecord


def _avg(blendshapes: dict[str, float], *names: str) -> float:
    vals = [blendshapes.get(n, 0.0) for n in names]
    return sum(vals) / len(vals) if vals else 0.0


def classify_expressions(
    blendshapes: dict[str, float], thresholds: ExpressionThresholds | None = None
) -> ExpressionRecord:
    t = thresholds or ExpressionThresholds()
    tags: list[str] = []

    smile = _avg(blendshapes, "mouthSmileLeft", "mouthSmileRight")
    jaw_open = blendshapes.get("jawOpen", 0.0)
    eye_blink = _avg(blendshapes, "eyeBlinkLeft", "eyeBlinkRight")
    brow_up = _avg(blendshapes, "browOuterUpLeft", "browOuterUpRight", "browInnerUp")
    brow_down = _avg(blendshapes, "browDownLeft", "browDownRight")
    mouth_frown = _avg(blendshapes, "mouthFrownLeft", "mouthFrownRight")
    eye_squint = _avg(blendshapes, "eyeSquintLeft", "eyeSquintRight")
    eye_wide = _avg(blendshapes, "eyeWideLeft", "eyeWideRight")
    looking_left = _avg(blendshapes, "eyeLookOutLeft", "eyeLookInRight")
    looking_right = _avg(blendshapes, "eyeLookOutRight", "eyeLookInLeft")
    looking_down = _avg(blendshapes, "eyeLookDownLeft", "eyeLookDownRight")
    looking_up = _avg(blendshapes, "eyeLookUpLeft", "eyeLookUpRight")
    mouth_press = _avg(blendshapes, "mouthPressLeft", "mouthPressRight")
    cheek_squint = _avg(blendshapes, "cheekSquintLeft", "cheekSquintRight")

    # --- smile family: strongest tier wins (mutually exclusive) ---
    if smile >= t.smile_big and jaw_open >= t.jaw_open_talking:
        tags.append("laughing")
    elif smile >= t.smile_big:
        tags.append("big_smile")
    elif smile >= t.smile_confident:
        tags.append("confident_smile")
    elif smile >= t.smile_slight:
        tags.append("slight_smile")

    # --- mouth / jaw state ---
    if jaw_open >= t.jaw_open_big:
        tags.append("mouth_open")
    elif t.jaw_open_talking <= jaw_open < t.jaw_open_big and smile < t.smile_confident:
        tags.append("talking")

    # --- eyes open/closed ---
    if eye_blink >= t.eyes_closed:
        tags.append("eyes_closed")

    # --- gaze direction (screen-space; see README for convention) ---
    if looking_left >= t.eyes_direction and looking_left > looking_right:
        tags.append("eyes_left")
    elif looking_right >= t.eyes_direction and looking_right > looking_left:
        tags.append("eyes_right")
    if looking_down >= t.eyes_direction and looking_down > looking_up:
        tags.append("looking_down")
    elif looking_up >= t.eyes_direction and looking_up > looking_down:
        tags.append("looking_up")

    # --- brows ---
    if brow_up >= t.brow_raise:
        tags.append("raised_eyebrows")
    if brow_down >= t.frown or mouth_frown >= t.frown:
        tags.append("frowning")

    # --- surprised: raised brows + open mouth + wide eyes, and NOT smiling ---
    surprise_signal = (brow_up + jaw_open + eye_wide) / 3.0
    if surprise_signal >= t.surprise_combo and smile < t.smile_slight:
        tags.append("surprised")

    # --- focused ---
    is_focused = (
        mouth_press >= t.focused_squint or eye_squint >= t.focused_squint or cheek_squint >= t.focused_squint
    ) and smile < t.smile_slight
    if is_focused:
        tags.append("focused")

    # --- serious / neutral baseline (only if nothing more specific fired for the mouth) ---
    calm_mouth = smile < t.smile_slight and jaw_open < t.jaw_open_talking and mouth_frown < t.frown
    if calm_mouth and brow_down < t.frown:
        tags.append("serious" if is_focused else "neutral")

    if not tags:
        tags.append("neutral")

    return ExpressionRecord(tags=tags, blendshapes=dict(blendshapes))
