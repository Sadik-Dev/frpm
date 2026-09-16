"""Shared test fixtures - fast, synthetic, no real models or network calls."""
from __future__ import annotations

import numpy as np
import pytest

from frpm.models import BBox, Candidate, ExpressionRecord, HeadPoseRecord, PoseBucket, QualityScoreRecord


def make_bbox(x: int = 100, y: int = 100, w: int = 120, h: int = 140) -> BBox:
    return BBox(x=x, y=y, w=w, h=h)


def make_candidate(
    candidate_id: str,
    timestamp: float = 0.0,
    frame_number: int = 0,
    embedding: "np.ndarray | None" = None,
    quality: float = 0.8,
    pose: str = "front",
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    expressions: "list[str] | None" = None,
    phash=None,
    identity_cluster: "int | None" = 0,
    frame_path: str = "dummy.jpg",
    bbox: "BBox | None" = None,
) -> Candidate:
    """Builds a fully-populated synthetic Candidate for unit tests, so tests
    never need to touch real images or ML models."""
    c = Candidate(
        candidate_id=candidate_id,
        frame_path=frame_path,
        timestamp=timestamp,
        frame_number=frame_number,
        bbox=bbox or make_bbox(),
        detection_confidence=0.95,
    )
    if embedding is not None:
        c.embedding = embedding
    else:
        seed = abs(hash(candidate_id)) % (2**31)
        c.embedding = np.random.RandomState(seed).randn(128).astype(np.float32)
    c.identity_cluster = identity_cluster
    c.quality = QualityScoreRecord(
        quality=quality, sharpness=quality, exposure=quality, face_resolution=quality,
        landmark_confidence=quality, artifact=quality, obstruction=quality,
    )
    c.pose = HeadPoseRecord(yaw=yaw, pitch=pitch, roll=roll, pose=PoseBucket(pose))
    c.expressions = ExpressionRecord(tags=expressions or ["neutral"], blendshapes={})
    c.phash = phash
    return c


@pytest.fixture
def bbox_factory():
    return make_bbox


@pytest.fixture
def candidate_factory():
    return make_candidate
