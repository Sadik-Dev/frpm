"""Body-visibility detection (upper/half/full body) via MediaPipe Pose
Landmarker (Apache-2.0).

Run only on the (much smaller) post-dedup candidate set belonging to the
already-selected identity - not on every sampled frame - since we only care
about the target person's body framing, not any bystander's.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .models import BBox, BodyVisibility

# MediaPipe Pose landmark indices (33-point BlazePose topology).
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

VISIBILITY_THRESHOLD = 0.5


class BodyPoseDetector:
    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=3,
        )
        self._mp = mp
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(self, bgr_image: np.ndarray) -> list:
        import cv2

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        return result.pose_landmarks or []

    def close(self) -> None:
        self._landmarker.close()


def _closest_pose_to_face(poses: list, bbox: BBox, image_w: int, image_h: int) -> Optional[list]:
    if not poses:
        return None
    face_cx, face_cy = bbox.center
    face_cx, face_cy = face_cx / image_w, face_cy / image_h
    best, best_dist = None, float("inf")
    for pose in poses:
        nose = pose[NOSE]
        visibility = getattr(nose, "visibility", None)
        if visibility is not None and visibility < VISIBILITY_THRESHOLD:
            continue
        dist = (nose.x - face_cx) ** 2 + (nose.y - face_cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best = pose
    return best


def classify_body_visibility(poses: list, bbox: BBox, image_w: int, image_h: int) -> BodyVisibility:
    pose = _closest_pose_to_face(poses, bbox, image_w, image_h)
    if pose is None:
        return BodyVisibility.NONE

    def vis(idx: int) -> float:
        v = getattr(pose[idx], "visibility", None)
        return v if v is not None else 0.0

    shoulders = max(vis(LEFT_SHOULDER), vis(RIGHT_SHOULDER))
    hips = max(vis(LEFT_HIP), vis(RIGHT_HIP))
    knees_ankles = max(vis(LEFT_KNEE), vis(RIGHT_KNEE), vis(LEFT_ANKLE), vis(RIGHT_ANKLE))

    if knees_ankles >= VISIBILITY_THRESHOLD:
        return BodyVisibility.FULL_BODY
    if hips >= VISIBILITY_THRESHOLD:
        return BodyVisibility.HALF_BODY
    if shoulders >= VISIBILITY_THRESHOLD:
        return BodyVisibility.UPPER_BODY
    return BodyVisibility.NONE
