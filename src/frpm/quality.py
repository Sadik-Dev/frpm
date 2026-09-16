"""Per-face quality scoring: sharpness, exposure, resolution, landmark
confidence, compression-artifact heuristic, and obstruction heuristic,
combined into a single normalized 0..1 ``quality_score`` (sub-scores are
kept too). All of this is plain image-processing math - no ML models - so
it is fully unit-testable with synthetic arrays.
"""
from __future__ import annotations

import numpy as np

from .config import QualityWeights
from .models import BBox, QualityScoreRecord


def sharpness_score(gray_face: np.ndarray, midpoint: float = 350.0) -> float:
    """Variance-of-Laplacian, mapped through a saturating curve so results
    stay in [0, 1) regardless of resolution/content (raw variance is
    unbounded)."""
    import cv2

    if gray_face.size == 0:
        return 0.0
    variance = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    return float(variance / (variance + midpoint))


def exposure_score(gray_face: np.ndarray) -> float:
    """Penalizes clipped (over/under-exposed) pixels and mean luminance far
    from mid-gray."""
    if gray_face.size == 0:
        return 0.0
    hist = np.bincount(gray_face.flatten(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    clipped_low = hist[:5].sum() / total
    clipped_high = hist[251:].sum() / total
    clip_penalty = min(1.0, (clipped_low + clipped_high) * 3.0)
    mean_lum = float(np.average(np.arange(256), weights=hist))
    deviation_penalty = min(1.0, abs(mean_lum - 128.0) / 128.0)
    return float(np.clip(1.0 - 0.7 * clip_penalty - 0.3 * deviation_penalty, 0.0, 1.0))


def face_resolution_score(bbox: BBox, target_px: int = 220) -> float:
    """1.0 once the (square-ish) face reaches ``target_px`` on a side."""
    target_area = float(target_px * target_px)
    if target_area <= 0:
        return 0.0
    return float(np.clip(bbox.area / target_area, 0.0, 1.0))


def artifact_score(gray_face: np.ndarray) -> float:
    """Cheap 8x8-grid blockiness heuristic (JPEG-style compression damage):
    compares the pixel-difference energy exactly at block boundaries against
    the average difference energy everywhere. 1.0 = clean, lower = blockier.
    """
    if gray_face.size == 0 or min(gray_face.shape[:2]) < 16:
        return 1.0
    g = gray_face.astype(np.float64)
    h, w = g.shape[:2]
    col_diff = np.abs(np.diff(g, axis=1))
    row_diff = np.abs(np.diff(g, axis=0))
    block_cols = list(range(7, w - 1, 8))
    block_rows = list(range(7, h - 1, 8))
    if not block_cols or not block_rows:
        return 1.0
    block_col_energy = col_diff[:, block_cols].mean()
    block_row_energy = row_diff[block_rows, :].mean()
    overall_col_energy = col_diff.mean() + 1e-6
    overall_row_energy = row_diff.mean() + 1e-6
    blockiness = 0.5 * (block_col_energy / overall_col_energy) + 0.5 * (block_row_energy / overall_row_energy)
    excess = max(0.0, blockiness - 1.0)
    return float(np.clip(1.0 - excess, 0.0, 1.0))


def obstruction_score(
    bbox: BBox, frame_w: int, frame_h: int, landmark_plausibility: float, border_margin: int = 2
) -> float:
    """Penalizes faces cropped by the frame edge and blends in landmark
    plausibility (a face mesh that doesn't fit well is a good proxy for
    "something is covering/cropping the face")."""
    touches_border = (
        bbox.x <= border_margin
        or bbox.y <= border_margin
        or bbox.x + bbox.w >= frame_w - border_margin
        or bbox.y + bbox.h >= frame_h - border_margin
    )
    score = 1.0 - (0.4 if touches_border else 0.0)
    score = 0.5 * score + 0.5 * landmark_plausibility
    return float(np.clip(score, 0.0, 1.0))


def compute_quality(
    frame_bgr: np.ndarray,
    bbox: BBox,
    landmark_confidence: float,
    weights: QualityWeights | None = None,
) -> QualityScoreRecord:
    import cv2

    weights = weights or QualityWeights()
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, bbox.x), max(0, bbox.y)
    x2, y2 = min(w, bbox.x + bbox.w), min(h, bbox.y + bbox.h)
    face = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.size else np.zeros((1, 1), dtype=np.uint8)

    sharpness = sharpness_score(gray)
    exposure = exposure_score(gray)
    face_resolution = face_resolution_score(bbox)
    artifact = artifact_score(gray)
    obstruction = obstruction_score(bbox, w, h, landmark_confidence)

    total_weight = (
        weights.sharpness
        + weights.exposure
        + weights.face_resolution
        + weights.landmark_confidence
        + weights.artifact
        + weights.obstruction
    )
    quality = (
        weights.sharpness * sharpness
        + weights.exposure * exposure
        + weights.face_resolution * face_resolution
        + weights.landmark_confidence * landmark_confidence
        + weights.artifact * artifact
        + weights.obstruction * obstruction
    )
    if total_weight > 0:
        quality /= total_weight

    return QualityScoreRecord(
        quality=float(np.clip(quality, 0.0, 1.0)),
        sharpness=sharpness,
        exposure=exposure,
        face_resolution=face_resolution,
        landmark_confidence=float(np.clip(landmark_confidence, 0.0, 1.0)),
        artifact=artifact,
        obstruction=obstruction,
    )
