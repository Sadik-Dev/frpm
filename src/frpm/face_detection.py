"""Face detection via OpenCV YuNet (Apache-2.0, opencv_zoo)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import MIN_FACE_SIZE_PX
from .models import BBox

logger = logging.getLogger("frpm.face_detection")


class YuNetDetector:
    def __init__(self, model_path: Path, score_threshold: float = 0.8, nms_threshold: float = 0.3, top_k: int = 5000):
        import cv2

        self._detector = cv2.FaceDetectorYN_create(
            str(model_path), "", (320, 320), score_threshold, nms_threshold, top_k
        )

    def detect(self, image: np.ndarray) -> list[tuple[BBox, float, list[float]]]:
        """Returns a list of (bbox, confidence, raw_yunet_row).

        ``raw_yunet_row`` is the full 15-value YuNet output row (bbox + 5
        landmark points + score) which SFace's ``alignCrop`` expects
        directly, so we keep it around unmodified.
        """
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return []
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(image)
        results: list[tuple[BBox, float, list[float]]] = []
        if faces is None:
            return results
        for row in faces:
            x, y, fw, fh = row[0], row[1], row[2], row[3]
            score = float(row[14])
            bbox = BBox(
                x=max(0, int(round(x))),
                y=max(0, int(round(y))),
                w=max(1, int(round(fw))),
                h=max(1, int(round(fh))),
            )
            results.append((bbox, score, [float(v) for v in row]))
        return results


def filter_detections(
    detections: list[tuple[BBox, float, list[float]]], min_size: int = MIN_FACE_SIZE_PX
) -> list[tuple[BBox, float, list[float]]]:
    """Drop tiny faces, but never return an empty list if the input wasn't
    empty - keep the largest one as a fallback per "ignore extremely small
    faces unless no better alternative exists"."""
    good = [d for d in detections if min(d[0].w, d[0].h) >= min_size]
    if good:
        return good
    if detections:
        return [max(detections, key=lambda d: d[0].area)]
    return []
