"""Quality scoring - pure image-processing math, no ML models needed."""
import numpy as np

from frpm.config import QualityWeights
from frpm.models import BBox
from frpm.quality import (
    artifact_score,
    compute_quality,
    exposure_score,
    face_resolution_score,
    obstruction_score,
    sharpness_score,
)


def _solid(value: int, size=(200, 200)) -> np.ndarray:
    return np.full(size, value, dtype=np.uint8)


def _checkerboard(size=(200, 200), block=4) -> np.ndarray:
    y, x = np.indices(size)
    return (((x // block) + (y // block)) % 2 * 255).astype(np.uint8)


def _noisy(size=(200, 200), mean=128, sigma=40, seed=0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.clip(mean + rng.randn(*size) * sigma, 0, 255).astype(np.uint8)


# --- sharpness -------------------------------------------------------------

def test_sharpness_flat_image_is_low():
    assert sharpness_score(_solid(128)) < 0.05


def test_sharpness_high_frequency_image_is_high():
    assert sharpness_score(_checkerboard()) > 0.5


def test_sharpness_is_monotonic_with_variance():
    assert sharpness_score(_noisy(sigma=40)) > sharpness_score(_solid(128))


def test_sharpness_empty_image_returns_zero():
    assert sharpness_score(np.zeros((0, 0), dtype=np.uint8)) == 0.0


# --- exposure ---------------------------------------------------------------

def test_exposure_mid_gray_is_high():
    assert exposure_score(_solid(128)) > 0.9


def test_exposure_overexposed_is_penalized():
    assert exposure_score(_solid(255)) < 0.3


def test_exposure_underexposed_is_penalized():
    assert exposure_score(_solid(0)) < 0.3


# --- resolution ---------------------------------------------------------------

def test_face_resolution_scales_with_area():
    small = face_resolution_score(BBox(x=0, y=0, w=50, h=50))
    large = face_resolution_score(BBox(x=0, y=0, w=300, h=300))
    assert 0 <= small < large <= 1.0
    assert large == 1.0


# --- artifact / obstruction ---------------------------------------------------

def test_artifact_score_bounds():
    assert 0.0 <= artifact_score(_checkerboard()) <= 1.0
    assert 0.0 <= artifact_score(_solid(128)) <= 1.0


def test_obstruction_penalizes_border_touch():
    center = obstruction_score(BBox(x=50, y=50, w=50, h=50), frame_w=200, frame_h=200, landmark_plausibility=1.0)
    edge = obstruction_score(BBox(x=0, y=50, w=50, h=50), frame_w=200, frame_h=200, landmark_plausibility=1.0)
    assert edge < center


def test_obstruction_scales_with_landmark_plausibility():
    good = obstruction_score(BBox(x=50, y=50, w=50, h=50), 200, 200, landmark_plausibility=1.0)
    bad = obstruction_score(BBox(x=50, y=50, w=50, h=50), 200, 200, landmark_plausibility=0.0)
    assert bad < good


# --- end-to-end ----------------------------------------------------------------

def test_compute_quality_end_to_end_is_bounded():
    gray = _checkerboard(size=(300, 300))
    frame_bgr = np.stack([gray, gray, gray], axis=-1)
    bbox = BBox(x=50, y=50, w=150, h=150)
    result = compute_quality(frame_bgr, bbox, landmark_confidence=0.9, weights=QualityWeights())
    assert 0.0 <= result.quality <= 1.0
    assert 0.0 <= result.sharpness <= 1.0
    assert 0.0 <= result.exposure <= 1.0


def test_compute_quality_prefers_sharper_face():
    bbox = BBox(x=50, y=50, w=150, h=150)
    flat = np.full((300, 300, 3), 128, dtype=np.uint8)
    textured_gray = _noisy(size=(300, 300), sigma=40)
    textured = np.stack([textured_gray, textured_gray, textured_gray], axis=-1)

    q_flat = compute_quality(flat, bbox, landmark_confidence=0.9, weights=QualityWeights())
    q_textured = compute_quality(textured, bbox, landmark_confidence=0.9, weights=QualityWeights())
    assert q_textured.quality > q_flat.quality


def test_compute_quality_penalizes_low_landmark_confidence():
    gray = _noisy(size=(300, 300), sigma=40)
    frame_bgr = np.stack([gray, gray, gray], axis=-1)
    bbox = BBox(x=50, y=50, w=150, h=150)
    high_conf = compute_quality(frame_bgr, bbox, landmark_confidence=1.0, weights=QualityWeights())
    low_conf = compute_quality(frame_bgr, bbox, landmark_confidence=0.0, weights=QualityWeights())
    assert high_conf.quality > low_conf.quality
