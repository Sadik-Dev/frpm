"""Real-model integration test: exercises the actual OpenCV YuNet/SFace and
MediaPipe Face/Pose Landmarker wrappers against a real image.

Marked ``slow`` (excluded from the default ``pytest`` run - see
``pyproject.toml``'s ``addopts``) since it downloads ~10MB of model weights
on first run and needs network access. Run explicitly with:

    pytest tests/integration -m slow -v
"""
from pathlib import Path

import pytest

from frpm.body import BodyPoseDetector, classify_body_visibility
from frpm.config import AppConfig
from frpm.embeddings import SFaceEmbedder, cosine_similarity
from frpm.expressions import classify_expressions
from frpm.face_detection import YuNetDetector, filter_detections
from frpm.landmarks import FaceLandmarker
from frpm.pose import compute_head_pose
from frpm.quality import compute_quality
from frpm.utils import ensure_model, imread_unicode

pytestmark = pytest.mark.slow

FIXTURE = Path(__file__).parent.parent / "fixtures" / "test_face.jpg"


@pytest.fixture(scope="module")
def cfg():
    return AppConfig(url="https://youtu.be/xxxxxxxxxxx", output_dir="output")


@pytest.fixture(scope="module")
def test_image():
    if not FIXTURE.exists():
        pytest.skip(f"Test fixture not found at {FIXTURE}")
    img = imread_unicode(FIXTURE)
    if img is None:
        pytest.skip("Could not decode test fixture image")
    return img


def test_yunet_detects_a_face(cfg, test_image):
    detector = YuNetDetector(ensure_model("yunet", cfg.models_cache_dir))
    dets = filter_detections(detector.detect(test_image))
    assert len(dets) >= 1


def test_sface_embedding_is_self_similar(cfg, test_image):
    detector = YuNetDetector(ensure_model("yunet", cfg.models_cache_dir))
    bbox, conf, row = max(filter_detections(detector.detect(test_image)), key=lambda d: d[0].area)
    embedder = SFaceEmbedder(ensure_model("sface", cfg.models_cache_dir))
    emb = embedder.embed(test_image, row)
    assert emb.shape == (128,)
    assert cosine_similarity(emb, emb) == pytest.approx(1.0, abs=1e-4)


def test_face_landmarker_produces_blendshapes_and_transform(cfg, test_image):
    detector = YuNetDetector(ensure_model("yunet", cfg.models_cache_dir))
    bbox, conf, _row = max(filter_detections(detector.detect(test_image)), key=lambda d: d[0].area)
    landmarker = FaceLandmarker(ensure_model("face_landmarker", cfg.models_cache_dir))
    try:
        result = landmarker.analyze(test_image, bbox, conf)
    finally:
        landmarker.close()
    assert result is not None
    assert len(result["blendshapes"]) == 52
    assert result["transform"].shape == (4, 4)
    assert 0.0 <= result["landmark_confidence"] <= 1.0

    pose = compute_head_pose(result["transform"], cfg.pose_thresholds)
    assert isinstance(pose.yaw, float) and isinstance(pose.pitch, float)

    quality = compute_quality(test_image, bbox, result["landmark_confidence"], cfg.quality_weights)
    assert 0.0 <= quality.quality <= 1.0

    expressions = classify_expressions(result["blendshapes"], cfg.expression_thresholds)
    assert len(expressions.tags) >= 1


def test_body_pose_detector_runs(cfg, test_image):
    bbox = filter_detections(YuNetDetector(ensure_model("yunet", cfg.models_cache_dir)).detect(test_image))[0][0]
    body_detector = BodyPoseDetector(ensure_model("pose_landmarker", cfg.models_cache_dir))
    try:
        poses = body_detector.detect(test_image)
        visibility = classify_body_visibility(poses, bbox, test_image.shape[1], test_image.shape[0])
    finally:
        body_detector.close()
    assert visibility is not None
