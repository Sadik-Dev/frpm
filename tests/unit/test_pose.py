"""Head-pose extraction/classification - pure math, no ML models needed."""
import pytest

from frpm.config import PoseThresholds
from frpm.models import PoseBucket
from frpm.pose import classify_pose, euler_angles_to_rotation_matrix, rotation_matrix_to_euler_angles


@pytest.mark.parametrize("pitch,yaw,roll", [
    (0, 0, 0),
    (10, -30, 5),
    (-15, 40, -8),
    (5, 80, 0),
    (-5, -80, 2),
    (25, 0, -15),
])
def test_euler_angle_round_trip(pitch, yaw, roll):
    matrix = euler_angles_to_rotation_matrix(pitch, yaw, roll)
    p2, y2, r2 = rotation_matrix_to_euler_angles(matrix)
    assert p2 == pytest.approx(pitch, abs=1e-3)
    assert y2 == pytest.approx(yaw, abs=1e-3)
    assert r2 == pytest.approx(roll, abs=1e-3)


def test_classify_pose_front():
    assert classify_pose(yaw=0, pitch=0, roll=0) == PoseBucket.FRONT
    assert classify_pose(yaw=10, pitch=0, roll=0) == PoseBucket.FRONT


def test_classify_pose_three_quarter():
    assert classify_pose(yaw=-30, pitch=0, roll=0) == PoseBucket.THREE_QUARTER_LEFT
    assert classify_pose(yaw=30, pitch=0, roll=0) == PoseBucket.THREE_QUARTER_RIGHT


def test_classify_pose_profile():
    assert classify_pose(yaw=-60, pitch=0, roll=0) == PoseBucket.PROFILE_LEFT
    assert classify_pose(yaw=60, pitch=0, roll=0) == PoseBucket.PROFILE_RIGHT


def test_classify_pose_looking_up_down():
    assert classify_pose(yaw=0, pitch=20, roll=0) == PoseBucket.LOOKING_DOWN
    assert classify_pose(yaw=0, pitch=-20, roll=0) == PoseBucket.LOOKING_UP


def test_classify_pose_extreme_yaw_dominates_over_pitch():
    # A strong profile shot should stay classified as profile even if there's
    # also some vertical component - profile is the more useful/distinctive label.
    assert classify_pose(yaw=70, pitch=20, roll=0) == PoseBucket.PROFILE_RIGHT


def test_classify_pose_thresholds_are_configurable():
    t = PoseThresholds(front_yaw_max=5.0)
    assert classify_pose(yaw=10, pitch=0, roll=0, thresholds=t) != PoseBucket.FRONT


def test_classify_pose_invert_yaw_flag():
    t = PoseThresholds(invert_yaw=True)
    assert classify_pose(yaw=-30, pitch=0, roll=0, thresholds=t) == PoseBucket.THREE_QUARTER_RIGHT


def test_classify_pose_invert_pitch_flag():
    t = PoseThresholds(invert_pitch=True)
    assert classify_pose(yaw=0, pitch=20, roll=0, thresholds=t) == PoseBucket.LOOKING_UP
