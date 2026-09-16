"""Head pose (yaw/pitch/roll) extraction and bucket classification.

Angles are extracted from MediaPipe's 4x4 facial transformation matrix using
the standard Rx(pitch) * Ry(yaw) * Rz(roll) Euler decomposition. The exact
sign of "positive yaw"/"positive pitch" in MediaPipe's coordinate space is
a matter of convention rather than something a synthetic unit test can
prove; ``PoseThresholds.invert_yaw`` / ``invert_pitch`` exist as a one-line
fix if real footage ever shows a mirrored classification.
"""
from __future__ import annotations

import numpy as np

from .config import PoseThresholds
from .models import HeadPoseRecord, PoseBucket


def rotation_matrix_to_euler_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """Returns (pitch, yaw, roll) in degrees from a 3x3 (or 4x4, top-left
    3x3 used) rotation matrix, assuming Rx * Ry * Rz composition order."""
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    pitch = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(-r[2, 0], np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    roll = np.arctan2(r[1, 0], r[0, 0])
    return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))


def euler_angles_to_rotation_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Inverse of the above (R = Rz(roll) @ Ry(yaw) @ Rx(pitch), matching
    the extraction formula above). Used mainly for round-trip unit testing."""
    p, y, r = np.radians([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def classify_pose(yaw: float, pitch: float, roll: float, thresholds: PoseThresholds | None = None) -> PoseBucket:
    t = thresholds or PoseThresholds()
    if t.invert_yaw:
        yaw = -yaw
    if t.invert_pitch:
        pitch = -pitch

    if abs(yaw) < t.three_quarter_yaw_max:
        if pitch >= t.pitch_down_deg:
            return PoseBucket.LOOKING_DOWN
        if pitch <= -t.pitch_up_deg:
            return PoseBucket.LOOKING_UP
    if abs(yaw) <= t.front_yaw_max:
        return PoseBucket.FRONT
    if abs(yaw) <= t.three_quarter_yaw_max:
        return PoseBucket.THREE_QUARTER_LEFT if yaw < 0 else PoseBucket.THREE_QUARTER_RIGHT
    return PoseBucket.PROFILE_LEFT if yaw < 0 else PoseBucket.PROFILE_RIGHT


def compute_head_pose(
    transform_matrix: np.ndarray | None, thresholds: PoseThresholds | None = None
) -> HeadPoseRecord:
    if transform_matrix is None:
        return HeadPoseRecord(yaw=0.0, pitch=0.0, roll=0.0, pose=PoseBucket.FRONT)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(transform_matrix)
    pose = classify_pose(yaw, pitch, roll, thresholds)
    return HeadPoseRecord(yaw=yaw, pitch=pitch, roll=roll, pose=pose)
