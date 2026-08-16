"""Normalized LeRobot units <-> sim radians for the SO-101 follower.

Constants derived 2026-07-19 from the calibration tick spans, a held
default-pose snapshot, and FK direction analysis (full derivation in the
bridge.py contract and the PRO-101 session log).

    rad = default_rad + direction * scale * (n - n_default)

TODO(balazs): implement the four functions; make tests/test_conversion.py
pass. No I/O in this module — pure math, dicts in, numbers out.
"""

# Sim joint order — index i here is index i in every obs/action vector.
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:5]  # policy actions cover these; gripper is held

# (direction, scale rad/unit, n_default, default_rad)
CALIB = {
    # direction confirmed +1 by LEFT/RIGHT physical test 2026-07-20 (was -1: x came out mirrored)
    "shoulder_pan": (+1, 0.01953, 10.99, 0.0),
    "shoulder_lift": (+1, 0.01870, 5.98, 1.57),
    "elbow_flex": (+1, 0.01707, 1.58, -1.57),
    # CORRECTED 2026-08-16 by physical measurement (scripts/bringup/wrist_check.py).
    # Was (89.6, 1.0), which was never measured — it was derived by walking back
    # from the +100 end stop. It was wrong by 0.644 rad, and that single error is
    # the root cause of three separate symptoms: the PickPlace home (1.57) looked
    # unreachable, ~48% of the policy's wrist commands appeared to saturate, and
    # the lift bridge needed a 2x ACTION_SCALE to stop the arm undershooting.
    # Measurement: commanded the home pose with shoulder_lift/elbow_flex confirmed
    # at ~0 (so the gripper angle isolates the wrist); encoder read n=85.7 while a
    # phone level showed the gripper 6 deg from vertical, which FK puts at 1.57 rad.
    # Reachable range goes [-2.555,+1.195] -> [-1.912,+1.838], so the whole URDF
    # range now fits with room at both ends.
    "wrist_flex": (+1, 0.01875, 85.7, 1.57),
    # PROVISIONAL: direction unconfirmed (jar-lid test pending); n_default
    # assumes direction +1 (quarter turn below the 2.24 reading at roll=0).
    "wrist_roll": (+1, 0.03141, -47.76, -1.57),
    "gripper": (+1, 0.02395, 0.32, 0.0),
}

# URDF joint limits (rad) — clamp every outgoing target to these.
LIMITS = {
    "shoulder_pan": (-2.0, 2.0),
    "shoulder_lift": (0.0, 3.5),
    "elbow_flex": (-3.14158, 0.0),
    "wrist_flex": (-1.658, 1.658),  # URDF so_arm101 wrist_flex limit; was (-2.5, 1.2) which clamped the lift home (1.57)
    "wrist_roll": (-3.14158, 3.14158),
    "gripper": (-0.2, 2.0),
}

GRIPPER_HOLD_N = 0.32  # normalized command that keeps the gripper at ~0 rad

DEFAULT_POSE_RAD = [0.0, 1.57, -1.57, 1.0, -1.57, 0.0]  # sim order


def n_to_rad(joint: str, n: float) -> float:
    """One normalized reading -> sim radians."""
    direction, scale, n_at_default, default_rad = CALIB[joint]
    return default_rad + direction * scale * (n - n_at_default)


def rad_to_n(joint: str, rad: float) -> float:
    """Sim radians -> normalized units (inverse of n_to_rad)."""
    direction, scale, n_at_default, default_rad = CALIB[joint]
    return n_at_default + (rad - default_rad) / (direction * scale)


def obs_to_joint_pos(obs: dict[str, float]) -> list[float]:
    """LeRobot get_observation() dict ('<joint>.pos' keys) -> 6 radians in
    sim order. Look up by name, never by dict order; raise KeyError if a
    joint is missing."""
    return [n_to_rad(j, obs[f"{j}.pos"]) for j in JOINTS]


def targets_to_action(targets_rad: list[float], gripper_rad: float | None = None) -> dict[str, float]:
    """5 arm joint targets (rad, sim order) -> LeRobot send_action() dict:
    5 converted+clamped arm entries plus the gripper. Clamp to LIMITS in
    radians BEFORE converting. gripper_rad=None holds the gripper at
    GRIPPER_HOLD_N (the reach policy does not actuate it); pass a radian target
    to drive it (the grasp/lift policy)."""
    action = {}
    for i, j in enumerate(ARM_JOINTS):
        lower, upper = LIMITS[j]
        action[f"{j}.pos"] = rad_to_n(j, min(max(targets_rad[i], lower), upper))
    if gripper_rad is None:
        action["gripper.pos"] = GRIPPER_HOLD_N
    else:
        lo, hi = LIMITS["gripper"]
        action["gripper.pos"] = rad_to_n("gripper", min(max(gripper_rad, lo), hi))
    return action
