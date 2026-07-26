"""Live loop: run the lift/grasp policy CLOSED-LOOP against the real arm.

Ramps to LIFT_DEFAULT_POSE_RAD, then drives the policy via grasp_bridge.grasp_hold
(scale -> slow-blend -> clamp -> send arm + binary gripper, velocity zeroed).
Ctrl+C or timeout ramps back to rest and torques off.

object_position is STATIC (pre-grasp only, v1 bridge validation) — put the cube
at KNOWN_OBJECT_POS. The policy saturates from the upright home (it wants to dive
at the cube), so defaults here are deliberately gentle. First run: arm clamped,
gripper clear, hand on the switch. Confirm the wrist_roll calibration sign first.

    python scripts/grasp/grasp_live.py --port COM8 --id follower1
    python scripts/grasp/grasp_live.py --port COM8 --id follower1 --slow 0.05 --duration 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from soarmrl import conversion
from soarmrl import grasp_bridge as gb
from soarmrl.bridge import go_to

POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "policies" / "lift_policy.pt"

# Static object pose in the LIFT policy's root frame: x = FORWARD (ahead),
# y = lateral, z = up. (Differs from the reach pose_command frame where ahead
# was -y — the first-run sideways swing was that mismatch.) Cube 15 cm ahead,
# centered; z ~ sim cube-on-table center (1 cm cube sits lower — see notes).
KNOWN_OBJECT_POS = [0.45, 0.01, 0.2]

# Lift goal pose (base frame xyz + identity quat), inside the trained in-air box.
# z near the top of the trained goal range (pos_z in [0.2, 0.35]) for a high lift.
TARGET_POSE = [0, 0, 0.0, 1.0, 0.0, 0.0, 0.0]

HZ = 30.0
SLOW = 0.225      # gentler than reach's 0.25 — the policy saturates from home
MAX_DELTA = 0.03  # tighter per-tick clamp than reach's 0.05, rad

# Cap the gripper servo's holding torque (Feetech Torque_Limit reg, 0-1000, same
# scale as Present_Load). The gripper stall-holds the cube the whole lift and was
# browning out the 5 V rail (Input voltage error, id 6). Capping the torque limits
# its current without moving the position command (which would corrupt the obs).

# Must stay above GRIP_LOAD_MIN (276) so the grasp still holds + detects; below the
# ~500 stall it was drawing. 0 disables the cap (restore factory default).
GRIPPER_TORQUE_LIMIT = 380


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--ramp-seconds", type=float, default=6.0)
    parser.add_argument("--duration", type=float, default=6.0, help="live-loop run length, seconds")
    parser.add_argument("--slow", type=float, default=SLOW, help="slow-mode blend factor (lower = gentler)")
    parser.add_argument("--max-delta", type=float, default=MAX_DELTA, help="max per-tick joint change, rad")
    parser.add_argument("--object", help="static object pos 'x,y,z' (default: built-in KNOWN_OBJECT_POS)")
    parser.add_argument("--target", help="lift goal 'x,y,z' (default: built-in TARGET_POSE)")
    parser.add_argument("--gripper-torque", type=int, default=GRIPPER_TORQUE_LIMIT,
                        help="gripper Torque_Limit cap 0-1000 (0 = factory default); "
                             "lower if it still browns out, raise toward 500 if the grip slips")
    args = parser.parse_args()

    object_pos = KNOWN_OBJECT_POS if args.object is None else [float(v) for v in args.object.split(",")]
    target_pose = TARGET_POSE if args.target is None else \
        [float(v) for v in args.target.split(",")] + [1.0, 0.0, 0.0, 0.0]

    policy = torch.jit.load(str(POLICY_PATH))
    policy.eval()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    if args.gripper_torque > 0:
        robot.bus.write("Torque_Limit", "gripper", args.gripper_torque)
        print(f"gripper Torque_Limit capped at {args.gripper_torque}/1000 (anti-brownout)")

    start_pose = None
    try:
        start_pose = conversion.obs_to_joint_pos(robot.get_observation())[:5]

        print(f"ramping to LIFT home {gb.LIFT_DEFAULT_POSE_RAD[:5]} over {args.ramp_seconds:.0f} s...")
        go_to(robot, gb.LIFT_DEFAULT_POSE_RAD[:5], args.ramp_seconds)

        n_ticks = max(1, round(args.duration * HZ))
        print(f"grasp loop: object={object_pos}  target={target_pose[:3]}  "
              f"slow={args.slow}  max_delta={args.max_delta}  {n_ticks} ticks at {HZ:.0f} Hz")
        print("Ctrl+C to abort -> ramps back to rest.")
        gb.grasp_hold(robot, policy, target_pose, object_pos, n_ticks,
                      args.slow, args.max_delta, hz=HZ, verbose=True)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if start_pose is not None:
            print("ramping back to rest...")
            try:
                go_to(robot, start_pose, args.ramp_seconds)
            except RuntimeError as e:
                print(f"could not ramp back (servo faulted?): {e}")
        try:
            robot.disconnect()
            print("torque off.")
        except RuntimeError as e:
            print(f"disconnect failed -- power-cycle the arm to clear the fault: {e}")


if __name__ == "__main__":
    main()
