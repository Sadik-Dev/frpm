"""Facial landmarks, blendshapes, and head-pose transform via MediaPipe
Face Landmarker (Apache-2.0).

We run the landmarker on a padded square crop around each already-detected
(YuNet) face rather than the full frame. This gives a guaranteed 1:1
correspondence between our ``Candidate`` objects and landmark results (no
detection-matching heuristics needed) and lets us skip faces we don't care
about (e.g. bystanders, post identity-selection).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .models import BBox


def padded_square_crop(image: np.ndarray, bbox: BBox, pad_ratio: float = 0.7) -> tuple[np.ndarray, int, int]:
    """Returns (crop, offset_x, offset_y) - a square crop around bbox padded
    by ``pad_ratio`` on each side, clamped to image bounds."""
    h, w = image.shape[:2]
    cx, cy = bbox.center
    half = max(bbox.w, bbox.h) * (1.0 + pad_ratio) / 2.0
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(w, cx + half))
    y2 = int(min(h, cy + half))
    if x2 <= x1 or y2 <= y1:
        return image[max(0, bbox.y):bbox.y + bbox.h, max(0, bbox.x):bbox.x + bbox.w], bbox.x, bbox.y
    return image[y1:y2, x1:x2], x1, y1


def _landmark_plausibility(landmarks) -> float:
    """Fraction of landmarks that land within (a slightly padded) [0,1]
    normalized crop range - a cheap proxy for "did the mesh actually fit
    the face" since MediaPipe doesn't expose a face-mesh confidence score."""
    if not landmarks:
        return 0.0
    ok = 0
    for lm in landmarks:
        if -0.05 <= lm.x <= 1.05 and -0.05 <= lm.y <= 1.05:
            ok += 1
    return ok / len(landmarks)


class FaceLandmarker:
    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._mp = mp
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def analyze(self, bgr_image: np.ndarray, bbox: BBox, detection_confidence: float) -> Optional[dict]:
        import cv2

        crop, _ox, _oy = padded_square_crop(bgr_image, bbox)
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for cat in result.face_blendshapes[0]:
                blendshapes[cat.category_name] = float(cat.score)

        transform = None
        if result.facial_transformation_matrixes:
            transform = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)

        plausibility = _landmark_plausibility(landmarks)
        confidence = float(np.clip(detection_confidence * (0.5 + 0.5 * plausibility), 0.0, 1.0))

        # A cheap 5-point summary (eyes/nose/mouth corners) normalized to the
        # ORIGINAL bbox space, reused later as a compact "landmark
        # configuration" fingerprint for near-duplicate detection.
        key_idx = [33, 263, 1, 61, 291]  # left eye outer, right eye outer, nose tip, mouth L, mouth R (MediaPipe FaceMesh indices)
        five_pt = np.array([[landmarks[i].x, landmarks[i].y] for i in key_idx], dtype=np.float32)

        return {
            "landmarks": landmarks,
            "blendshapes": blendshapes,
            "transform": transform,
            "landmark_confidence": confidence,
            "five_pt": five_pt,
        }

    def close(self) -> None:
        self._landmarker.close()
