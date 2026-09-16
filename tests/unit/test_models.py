"""Core data model behavior: BBox geometry, Candidate cache round-trip."""
import numpy as np
import pytest

from frpm.models import BBox, Candidate, ExpressionRecord, HeadPoseRecord, PoseBucket, QualityScoreRecord


def test_bbox_area_and_center():
    b = BBox(x=10, y=20, w=100, h=50)
    assert b.area == 5000
    assert b.center == (60.0, 45.0)


def test_bbox_to_xyxy():
    b = BBox(x=10, y=20, w=100, h=50)
    assert b.to_xyxy() == (10, 20, 110, 70)


def test_bbox_iou_identical():
    b = BBox(x=0, y=0, w=10, h=10)
    assert b.iou(b) == 1.0


def test_bbox_iou_disjoint():
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=100, y=100, w=10, h=10)
    assert a.iou(b) == 0.0


def test_bbox_iou_partial_overlap():
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=5, y=0, w=10, h=10)
    assert a.iou(b) == pytest.approx(50 / 150)


def test_candidate_cache_round_trip():
    c = Candidate(
        candidate_id="x1", frame_path="f.jpg", timestamp=1.5, frame_number=45,
        bbox=BBox(x=1, y=2, w=3, h=4), detection_confidence=0.88, yunet_row=[1.0, 2.0],
    )
    c.embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    c.quality = QualityScoreRecord(
        quality=0.9, sharpness=0.8, exposure=0.7, face_resolution=0.6,
        landmark_confidence=0.5, artifact=0.4, obstruction=0.3,
    )
    c.pose = HeadPoseRecord(yaw=1.0, pitch=2.0, roll=3.0, pose=PoseBucket.FRONT)
    c.expressions = ExpressionRecord(tags=["neutral"], blendshapes={"jawOpen": 0.1})

    restored = Candidate.from_cache_dict(c.to_cache_dict())

    assert restored.candidate_id == c.candidate_id
    assert restored.timestamp == c.timestamp
    assert restored.frame_number == c.frame_number
    assert restored.bbox == c.bbox
    assert np.allclose(restored.embedding, c.embedding)
    assert restored.quality.quality == c.quality.quality
    assert restored.pose.pose == PoseBucket.FRONT
    assert restored.expressions.tags == ["neutral"]
    assert restored.expressions.blendshapes == {"jawOpen": 0.1}


def test_candidate_cache_round_trip_handles_missing_optional_fields():
    c = Candidate(
        candidate_id="x2", frame_path="f.jpg", timestamp=0.0, frame_number=0,
        bbox=BBox(x=0, y=0, w=1, h=1), detection_confidence=0.5,
    )
    restored = Candidate.from_cache_dict(c.to_cache_dict())
    assert restored.embedding is None
    assert restored.quality is None
    assert restored.pose is None
