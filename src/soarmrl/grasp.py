"""Scripted pick-up (teach-and-replay) — the demo finale (soarmrl.grasp).

The lift RL policy limit-cycles on hardware (it saturates from the upright home
and thrashes across the whole throttle range — 2026-07-26 session log), so the
pick is scripted instead: replay the hand-taught waypoints (scripts/grasp/teach.py
-> grasp_waypoints.json) with go_to-style gripper-aware ramps, close the gripper,
confirm the hold with grasp_bridge.is_grasping, then lift. Reuses the same
calibration/limits (conversion) and the same physical detector (grasp_bridge) as
the policy path — only the trajectory source differs (taught, not learned).

Sequence: approach (jaws open, above the cube) -> grasp (descend, jaws open) ->
close + confirm -> lift (jaws closed). The confirm GATES the lift: no confirmed
grasp -> open and stop, so the arm never lifts an empty gripper.
"""

import json
import time
from pathlib import Path

from soarmrl import conversion
from soarmrl import grasp_bridge as gb

GRIPPER_IDX = conversion.JOINTS.index("gripper")


def load_waypoints(path) -> dict[str, list[float]]:
    """teach.py's JSON -> {name: 6 joint angles (rad, sim order)}."""
    return json.loads(Path(path).read_text())["waypoints"]


def move_to(robot, arm_target_rad, gripper_rad, seconds, hz: float = 30.0) -> None:
    """Ramp the 5 arm joints from their current position to arm_target_rad while
    holding the gripper at gripper_rad. Blocks. The gripper-actuating peer of
    bridge.go_to (which holds the gripper fixed)."""
    start = conversion.obs_to_joint_pos(robot.get_observation())[:5]
    steps = max(1, round(seconds * hz))
    for k in range(1, steps + 1):
        alpha = k / steps
        step = [s + alpha * (t - s) for s, t in zip(start, arm_target_rad[:5])]
        robot.send_action(conversion.targets_to_action(step, gripper_rad))
        time.sleep(1.0 / hz)


def close_and_confirm(robot, grasp_arm, seconds: float = 1.5, hz: float = 30.0,
                      verbose: bool = False) -> bool:
    """Hold the grasp pose and command the gripper closed for `seconds`, then
    report whether grasp_bridge.is_grasping stayed true for GRASP_TRIGGER_TICKS
    in a row — a real, sustained hold, not a transient closing spike."""
    dt = 1.0 / hz
    n = max(1, round(seconds * hz))
    streak = 0
    for tick in range(n):
        robot.send_action(conversion.targets_to_action(grasp_arm[:5], gb.GRIPPER_CLOSE_RAD))
        grip = conversion.obs_to_joint_pos(robot.get_observation())[GRIPPER_IDX]
        held = gb.is_grasping(robot, grip, True)
        streak = streak + 1 if held else 0
        if verbose:
            load = abs(robot.bus.read("Present_Load", "gripper"))
            tag = "HOLD" if held else "    "
            print(f"  close {tick:3d} {tag} grip={grip:+.3f} load={load:4.0f} streak={streak}")
        time.sleep(dt)
    return streak >= gb.GRASP_TRIGGER_TICKS


def pick(robot, waypoints, approach_s: float = 4.0, descend_s: float = 2.0,
         close_s: float = 1.5, lift_s: float = 2.5, hz: float = 30.0,
         verbose: bool = True) -> bool:
    """Full scripted pick: approach -> descend -> close+confirm -> lift. Returns
    True iff the grasp was confirmed and the arm lifted. On a failed confirm it
    opens the gripper at the grasp pose and returns False (no lift)."""
    approach = waypoints["approach"]
    grasp_pose = waypoints["grasp"]
    lift = waypoints["lift"]

    print("approach: moving above the cube (jaws open)...")
    move_to(robot, approach, gb.GRIPPER_OPEN_RAD, approach_s, hz)

    print("descend: lowering to the cube (jaws open)...")
    move_to(robot, grasp_pose, gb.GRIPPER_OPEN_RAD, descend_s, hz)

    print("close: gripping and confirming the hold...")
    if not close_and_confirm(robot, grasp_pose, close_s, hz, verbose):
        print("NO grasp confirmed -> opening, not lifting.")
        move_to(robot, grasp_pose, gb.GRIPPER_OPEN_RAD, 0.5, hz)
        return False

    print("grasp confirmed -> lifting.")
    move_to(robot, lift, gb.GRIPPER_CLOSE_RAD, lift_s, hz)
    return True
