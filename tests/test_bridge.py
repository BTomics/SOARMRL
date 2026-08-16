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
from soarmrl.bridge import builds_obs, clamp_delta
# Held snapshot at the sim default pose (flex/roll corrected values).
DEFAULT_POSE_ROW = {
    "shoulder_pan": 10.99,
    "shoulder_lift": 5.98,
    "elbow_flex": 1.58,
    # 55.3, not the old 89.6: wrist_flex's calibration reference was corrected
    # by physical measurement 2026-08-16 (see conversion.CALIB). The 89.6 here was
    # recorded at a pose ASSUMED to be 1.0 rad that was really ~1.5 rad, so this
    # fixture was encoding the same 0.644 rad error the constants were.
    "wrist_flex": 55.3,
    "wrist_roll": -47.76,
    "gripper": 0.32,
}

def test_builds_obs_shape():
    obs_dict = {f"{j}.pos": v for j, v in DEFAULT_POSE_ROW.items()}
    prev_joint_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    last_action = [0.0, 0.0, 0.0, 0.0, 0.0]
    pose_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dt = 1/30
    obs = builds_obs(obs_dict, pose_command, last_action, prev_joint_pos, dt)
    assert len(obs) == 24
    
def test_builds_obs_content():
    obs_dict = {f"{j}.pos": v for j, v in DEFAULT_POSE_ROW.items()}
    prev_joint_pos = DEFAULT_POSE_RAD  # held still: no motion since last tick
    last_action = [0.0, 0.0, 0.0, 0.0, 0.0]
    pose_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dt = 1/30
    obs = builds_obs(obs_dict, pose_command, last_action, prev_joint_pos, dt)
    assert obs[:6] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=0.05)
    assert obs[6:12] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=0.05)
    assert obs[12:19] == pytest.approx(pose_command, abs=0.05)
    assert obs[19:24] == pytest.approx(last_action, abs=0.05)

def test_clamp_delta_nochange():
    prev_target = [0.0, 0.0, 0.0, 0.0, 0.0]
    new_target = [0.05, -0.03, 0.07, -0.09, 0.02]
    max_delta = 0.1
    clamped_targets = clamp_delta(prev_target, new_target, max_delta)
    assert clamped_targets == pytest.approx([0.05, -0.03, 0.07, -0.09, 0.02], abs=1e-9)

def test_clamp_delta_large_jump_pos():
    prev_target = [0.0, 0.0, 0.0, 0.0, 0.0]
    new_target = [0.5, 0.0, 0.0, 0.0, 0.0]
    max_delta = 0.1
    clamped_targets = clamp_delta(prev_target, new_target, max_delta)
    assert clamped_targets == pytest.approx([0.1, 0.0, 0.0, 0.0, 0.0], abs=0.05)

def test_clamp_delta_neg_jump():
    prev_target = [0.0, 0.0, 0.0, 0.0, 0.0]
    new_target = [-0.5, 0.0, 0.0, 0.0, 0.0]
    max_delta = 0.1
    clamped_targets = clamp_delta(prev_target, new_target, max_delta)
    assert clamped_targets == pytest.approx([-0.1, 0.0, 0.0, 0.0, 0.0], abs=0.05)