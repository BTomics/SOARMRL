"""Executable spec for conversion.py, built on real measurements (2026-07-19).

Run from SOARMRL/:  python -m pytest tests/ -v
"""

import math

import pytest

from soarmrl.conversion import (
    ARM_JOINTS,
    DEFAULT_POSE_RAD,
    GRIPPER_HOLD_N,
    JOINTS,
    LIMITS,
    n_to_rad,
    obs_to_joint_pos,
    rad_to_n,
    targets_to_action,
)

# Held snapshot at the sim default pose (flex/roll corrected values).
DEFAULT_POSE_ROW = {
    "shoulder_pan": 10.99,
    "shoulder_lift": 5.98,
    "elbow_flex": 1.58,
    "wrist_flex": 89.6,
    "wrist_roll": -47.76,
    "gripper": 0.32,
}

# Measured 2026-07-19 with the elbow folded to its physical end stop:
# independent of the data the constants were fitted to.
FOLD_STOP_ELBOW_N = 96.44


def test_default_pose_row_converts_to_default_pose():
    obs = {f"{j}.pos": v for j, v in DEFAULT_POSE_ROW.items()}
    rads = obs_to_joint_pos(obs)
    assert rads == pytest.approx(DEFAULT_POSE_RAD, abs=0.05)


def test_fold_stop_measurement():
    # Calibrated stop slightly overshoots the URDF limit -> ~+0.05 expected.
    assert n_to_rad("elbow_flex", FOLD_STOP_ELBOW_N) == pytest.approx(0.0, abs=0.1)


def test_round_trip():
    for joint in JOINTS:
        for n in (-80.0, -5.0, 0.0, 42.5, 95.0):
            assert rad_to_n(joint, n_to_rad(joint, n)) == pytest.approx(n, abs=1e-9)


def test_pan_direction_matches_geometry():
    # FK ground truth: +pan -> +x (viewer-right from behind); left is -x = -rad.
    # Raw read "swinging left dropped n" => lower n goes with -rad, so n and rad
    # move together => direction +1. Confirmed by the LEFT/RIGHT reach test
    # 2026-07-20; the earlier -1 came from mislabeling "left" as +rad.
    assert n_to_rad("shoulder_pan", 20.0) > n_to_rad("shoulder_pan", 0.0)


def test_obs_requires_all_joints():
    obs = {f"{j}.pos": v for j, v in DEFAULT_POSE_ROW.items()}
    del obs["wrist_flex.pos"]
    with pytest.raises(KeyError):
        obs_to_joint_pos(obs)


def test_action_at_default_pose_holds_position():
    action = targets_to_action(DEFAULT_POSE_RAD[:5])
    for joint in ARM_JOINTS:
        assert action[f"{joint}.pos"] == pytest.approx(DEFAULT_POSE_ROW[joint], abs=0.5)
    assert action["gripper.pos"] == pytest.approx(GRIPPER_HOLD_N)


def test_action_clamps_to_urdf_limits():
    # Elbow commanded far past its upper limit must clamp to the limit,
    # not pass through.
    wild = list(DEFAULT_POSE_RAD[:5])
    wild[2] = 2.0  # elbow limit is [-3.14158, 0]
    action = targets_to_action(wild)
    clamped_n = rad_to_n("elbow_flex", LIMITS["elbow_flex"][1])
    assert action["elbow_flex.pos"] == pytest.approx(clamped_n, abs=1e-6)
