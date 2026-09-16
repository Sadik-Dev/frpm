"""Typed data contracts shared across pipeline stages.

Pydantic models are used for anything that gets serialized to JSON
(caches, ``manifest.json``). A plain ``Candidate`` dataclass carries the
richer in-memory, per-run state (numpy arrays, etc.) as it flows through the
pipeline; it exposes ``to_cache_dict`` / ``from_cache_dict`` for the subset
of fields that need to survive a resumed run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


class PoseBucket(str, Enum):
    FRONT = "front"
    THREE_QUARTER_LEFT = "three_quarter_left"
    THREE_QUARTER_RIGHT = "three_quarter_right"
    PROFILE_LEFT = "profile_left"
    PROFILE_RIGHT = "profile_right"
    LOOKING_UP = "looking_up"
    LOOKING_DOWN = "looking_down"


class BodyVisibility(str, Enum):
    NONE = "none"
    UPPER_BODY = "upper_body"
    HALF_BODY = "half_body"
    FULL_BODY = "full_body"


# Practical expression vocabulary the classifier is allowed to emit.
EXPRESSION_VOCAB = [
    "neutral",
    "slight_smile",
    "confident_smile",
    "big_smile",
    "laughing",
    "serious",
    "focused",
    "surprised",
    "mouth_open",
    "talking",
    "looking_down",
    "looking_up",
    "eyes_left",
    "eyes_right",
    "eyes_closed",
    "raised_eyebrows",
    "frowning",
]


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def iou(self, other: "BBox") -> float:
        ax1, ay1, ax2, ay2 = self.to_xyxy()
        bx1, by1, bx2, by2 = other.to_xyxy()
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class HeadPoseRecord(BaseModel):
    yaw: float
    pitch: float
    roll: float
    pose: PoseBucket


class QualityScoreRecord(BaseModel):
    quality: float
    sharpness: float
    exposure: float
    face_resolution: float
    landmark_confidence: float
    artifact: float
    obstruction: float


class ExpressionRecord(BaseModel):
    tags: list[str] = Field(default_factory=list)
    blendshapes: dict[str, float] = Field(default_factory=dict)


class VideoMetadata(BaseModel):
    video_id: str
    title: str = ""
    url: str = ""
    duration: float = 0.0
    uploader: str = ""
    upload_date: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0


class IdentityClusterInfo(BaseModel):
    cluster_id: int
    count: int
    representative_file: Optional[str] = None


class SelectedImageRecord(BaseModel):
    file: str
    source_timestamp: float
    frame_number: int
    identity_score: float
    quality_score: float
    face: HeadPoseRecord
    expressions: list[str]
    body_visibility: str
    categories: list[str] = Field(default_factory=list)
    face_crop: Optional[str] = None
    portrait_crop: Optional[str] = None
    original: Optional[str] = None


class Manifest(BaseModel):
    video: VideoMetadata
    identity_cluster_id: int
    frames_examined: int
    faces_found: int
    identities_found: int
    candidates_after_dedup: int
    pack_size_requested: int
    images: list[SelectedImageRecord] = Field(default_factory=list)
    reference_pack: list[str] = Field(default_factory=list)
    lora_dataset: list[str] = Field(default_factory=list)
    quality_distribution: dict[str, int] = Field(default_factory=dict)
    expression_distribution: dict[str, int] = Field(default_factory=dict)
    pose_distribution: dict[str, int] = Field(default_factory=dict)


@dataclass
class Candidate:
    """Rich, in-memory, per-face-per-frame record used while processing.

    One instance is created per detected face per sampled frame. Fields are
    filled in progressively by later pipeline stages; anything left as
    ``None`` simply hasn't been computed (or wasn't applicable) yet.
    """

    candidate_id: str
    frame_path: str
    timestamp: float
    frame_number: int
    bbox: BBox
    detection_confidence: float
    yunet_row: list[float] = field(default_factory=list)  # raw 15-value YuNet row (bbox+5pt+score)

    embedding: Optional[np.ndarray] = None
    identity_cluster: Optional[int] = None

    landmarks_5pt: Optional[np.ndarray] = None
    blendshapes: dict[str, float] = field(default_factory=dict)
    landmark_confidence: float = 0.0
    face_transform_matrix: Optional[np.ndarray] = None

    quality: Optional[QualityScoreRecord] = None
    pose: Optional[HeadPoseRecord] = None
    expressions: Optional[ExpressionRecord] = None
    body_visibility: BodyVisibility = BodyVisibility.NONE

    phash: Optional[Any] = None
    duplicate_of: Optional[str] = None  # candidate_id of the kept representative, if this one was dropped

    def to_cache_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "frame_path": self.frame_path,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
            "bbox": self.bbox.model_dump(),
            "detection_confidence": self.detection_confidence,
            "yunet_row": list(self.yunet_row),
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "identity_cluster": self.identity_cluster,
            "landmarks_5pt": self.landmarks_5pt.tolist() if self.landmarks_5pt is not None else None,
            "blendshapes": self.blendshapes,
            "landmark_confidence": self.landmark_confidence,
            "quality": self.quality.model_dump() if self.quality else None,
            "pose": self.pose.model_dump(mode="json") if self.pose else None,
            "expressions": self.expressions.model_dump() if self.expressions else None,
            "body_visibility": self.body_visibility.value,
            "phash": str(self.phash) if self.phash is not None else None,
            "duplicate_of": self.duplicate_of,
        }

    @classmethod
    def from_cache_dict(cls, d: dict) -> "Candidate":
        import imagehash

        c = cls(
            candidate_id=d["candidate_id"],
            frame_path=d["frame_path"],
            timestamp=d["timestamp"],
            frame_number=d["frame_number"],
            bbox=BBox(**d["bbox"]),
            detection_confidence=d["detection_confidence"],
            yunet_row=d.get("yunet_row", []),
        )
        if d.get("embedding") is not None:
            c.embedding = np.array(d["embedding"], dtype=np.float32)
        c.identity_cluster = d.get("identity_cluster")
        if d.get("landmarks_5pt") is not None:
            c.landmarks_5pt = np.array(d["landmarks_5pt"], dtype=np.float32)
        c.blendshapes = d.get("blendshapes") or {}
        c.landmark_confidence = d.get("landmark_confidence", 0.0)
        if d.get("quality"):
            c.quality = QualityScoreRecord(**d["quality"])
        if d.get("pose"):
            c.pose = HeadPoseRecord(**d["pose"])
        if d.get("expressions"):
            c.expressions = ExpressionRecord(**d["expressions"])
        c.body_visibility = BodyVisibility(d.get("body_visibility", "none"))
        if d.get("phash"):
            try:
                c.phash = imagehash.hex_to_hash(d["phash"])
            except Exception:
                c.phash = None
        c.duplicate_of = d.get("duplicate_of")
        return c
