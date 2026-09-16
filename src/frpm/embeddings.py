"""Face recognition embeddings via OpenCV SFace (Apache-2.0, opencv_zoo).

Used purely for local identity clustering/deduplication - never leaves the
machine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class SFaceEmbedder:
    def __init__(self, model_path: Path):
        import cv2

        self._recognizer = cv2.FaceRecognizerSF_create(str(model_path), "")

    def embed(self, image: np.ndarray, yunet_row: list[float]) -> np.ndarray:
        row = np.array(yunet_row, dtype=np.float32).reshape(1, -1)
        aligned = self._recognizer.alignCrop(image, row)
        feature = self._recognizer.feature(aligned)
        return np.asarray(feature).flatten().astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
