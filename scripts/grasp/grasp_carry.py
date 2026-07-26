"""Grasp with the lift policy, then CARRY with the reach policy.

Two policies, one run. The lift/grasp policy is good at the contact-rich grab but
its lift endpoint is a weak knob (the goal-tracking reward is loose). The reach
policy is the opposite: it servos the end-effector to a COMMANDED pose in a proven,
controllable box -- and its home (wrist_flex 1.0) is one the real arm can actually
reach, so it runs in-distribution (no 2x scale hack). So: grasp with lift, then hand
the held cube to reach to place it where you ask.

Sequence:
  1. ramp to the LIFT home
  2. grasp_hold (lift policy) -> grab + small lift; proceed ONLY if the grasp latched
  3. gentle-grip ramp from the grasp pose to the REACH home (brings the arm into
     reach's in-distribution start while still holding the cube)
  4. reach_hold (reach policy) carries the cube to --place, gripper held gently
  5. release at the destination, return to rest, torque off

The gripper is held at grasp_bridge.GRIPPER_HOLD_RAD the whole carry (not the reach
default's ~0 rad firm close) so it keeps the cube without stall-browning the 5 V rail
-- same anti-brownout fix as grasp_live, plus the gripper Torque_Limit cap at connect.

    python scripts/grasp/grasp_carry.py --port COM8 --id follower1
    python scripts/grasp/grasp_carry.py --port COM8 --id follower1 --place 0.05,-0.22,0.25

Arm clamped, cube on its known spot, hand on the switch. Power-cycle the arm first
if a previous run left a latched servo fault.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from soarmrl import bridge, conversion, grasp
from soarmrl import grasp_bridge as gb

REPO = Path(__file__).resolve().parent.parent.parent
LIFT_POLICY_PATH = REPO / "policies" / "lift_policy.pt"
REACH_POLICY_PATH = REPO / "policies" / "policy.pt"

# Grasp phase (lift policy) -- same known cube spot + working throttle as grasp_live.
KNOWN_OBJECT_POS = [0.45, 0.01, 0.2]
LIFT_TARGET_POSE = [0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0]  # in-box lift goal (grab + rise)
GRASP_SLOW = 0.225
GRASP_MAX_DELTA = 0.03
# Stop the lift policy this many ticks after the grasp latches — just enough to
# clear the cube off the table, then hand off to reach. Skips the policy's own
# (janky, weak-target) drive to LIFT_TARGET_POSE, which reach supersedes anyway.
CLEARANCE_LIFT_TICKS = 20  # ~0.7 s at 30 Hz

# Carry phase (reach policy) -- pose_command is the REACH base frame: -y is FORWARD,
# z up. Trained box: x[-0.1,0.1], y[-0.25,-0.1], z[0.1,0.3]. Stay inside it.
PLACE_POSE = [-0.1, -0.25, 0.1, 1.0, 0.0, 0.0, 0.0]  # centered, ~18cm fwd, ~20cm up
CARRY_SLOW = 0.20       # gentler than reach's 0.25 -- carrying a payload
CARRY_MAX_DELTA = 0.04

HZ = 30.0
GRIPPER_TORQUE_LIMIT = 380  # anti-brownout cap on the gripper servo (see grasp_live)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--ramp-seconds", type=float, default=6.0)
    parser.add_argument("--grasp-duration", type=float, default=6.0, help="lift-policy grasp phase, s")
    parser.add_argument("--carry-duration", type=float, default=8.0, help="reach-policy carry phase, s")
    parser.add_argument("--object", help="static cube pos 'x,y,z' (default: built-in KNOWN_OBJECT_POS)")
    parser.add_argument("--place", help="reach destination 'x,y,z' in the reach box (default: PLACE_POSE)")
    parser.add_argument("--gripper-torque", type=int, default=GRIPPER_TORQUE_LIMIT,
                        help="gripper Torque_Limit cap 0-1000 (0 = factory default)")
    args = parser.parse_args()

    object_pos = KNOWN_OBJECT_POS if args.object is None else [float(v) for v in args.object.split(",")]
    place_pose = PLACE_POSE if args.place is None else \
        [float(v) for v in args.place.split(",")] + [1.0, 0.0, 0.0, 0.0]

    lift_policy = torch.jit.load(str(LIFT_POLICY_PATH)); lift_policy.eval()
    reach_policy = torch.jit.load(str(REACH_POLICY_PATH)); reach_policy.eval()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    if args.gripper_torque > 0:
        robot.bus.write("Torque_Limit", "gripper", args.gripper_torque)
        print(f"gripper Torque_Limit capped at {args.gripper_torque}/1000 (anti-brownout)")

    start_pose = None
    try:
        start_pose = conversion.obs_to_joint_pos(robot.get_observation())[:5]

        # 1) Grasp with the lift policy.
        print(f"ramping to LIFT home {gb.LIFT_DEFAULT_POSE_RAD[:5]} over {args.ramp_seconds:.0f} s...")
        bridge.go_to(robot, gb.LIFT_DEFAULT_POSE_RAD[:5], args.ramp_seconds)

        grasp_ticks = max(1, round(args.grasp_duration * HZ))
        print(f"GRASP phase (lift policy): object={object_pos}  {grasp_ticks} ticks")
        _, grasped = gb.grasp_hold(robot, lift_policy, LIFT_TARGET_POSE, object_pos,
                                   grasp_ticks, GRASP_SLOW, GRASP_MAX_DELTA, hz=HZ, verbose=True,
                                   hold_ticks_after_grasp=CLEARANCE_LIFT_TICKS)

        if not grasped:
            print("\nno grasp confirmed -> not carrying. Returning to rest.")
            return

        # 2) Hand off to reach: ramp (holding the cube gently) into reach's
        #    in-distribution home, then let reach servo the cube to --place.
        print(f"\ngrasp confirmed -> handing off to REACH. Ramping to reach home "
              f"{conversion.DEFAULT_POSE_RAD[:5]} (holding the cube)...")
        grasp.move_to(robot, conversion.DEFAULT_POSE_RAD[:5], gb.GRIPPER_HOLD_RAD, args.ramp_seconds)

        carry_ticks = max(1, round(args.carry_duration * HZ))
        print(f"CARRY phase (reach policy): place={place_pose[:3]}  {carry_ticks} ticks")
        bridge.reach_hold(robot, reach_policy, place_pose, carry_ticks,
                          CARRY_SLOW, CARRY_MAX_DELTA, hz=HZ, verbose=True,
                          gripper_rad=gb.GRIPPER_HOLD_RAD)

        # 3) Release at the destination.
        print("\nreleasing the cube...")
        placed = conversion.obs_to_joint_pos(robot.get_observation())[:5]
        grasp.move_to(robot, placed, gb.GRIPPER_OPEN_RAD, 0.8)

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if start_pose is not None:
            print("returning to rest (jaws open)...")
            try:
                grasp.move_to(robot, start_pose, gb.GRIPPER_OPEN_RAD, args.ramp_seconds)
            except RuntimeError as e:
                print(f"could not return to rest (servo faulted?): {e}")
        try:
            robot.disconnect()
            print("torque off.")
        except RuntimeError as e:
            print(f"disconnect failed -- power-cycle the arm to clear the fault: {e}")


if __name__ == "__main__":
    main()
