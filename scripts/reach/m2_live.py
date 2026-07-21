"""M2/M3 live loop: run the reach policy CLOSED-LOOP against the real arm.

Ramps to a start pose, then drives the policy toward POSE_COMMAND via
bridge.reach_hold (scale -> slow-blend -> clamp -> send, velocity zeroed).
Ctrl+C or timeout ramps back to rest and torques off.

    python scripts/reach/m2_live.py --port COM8 --id follower1                  # M2 hold (near fixed point)
    python scripts/reach/m2_live.py --port COM8 --id follower1 --start default  # M3 reach (from default pose)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from soarmrl import conversion
from soarmrl.bridge import go_to, reach_hold

POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "policies" / "policy.pt"

# Hand-posed joint target (rad, sim order) near the policy's fixed point for
# POSE_COMMAND -- measured 2026-07-20. Used as the M2 hold-test start.
HAND_POSED_TARGET_RAD = [0.3279, 1.4139, -1.8310, 0.7829, -0.2285, 0.0292]

# In-box EE target (base frame xyz + identity quat).
POSE_COMMAND = [0.0, -0.221, 0.212, 1.0, 0.0, 0.0, 0.0]

HZ = 30.0
SLOW = 0.25       # slow-mode blend (override with --slow)
MAX_DELTA = 0.05  # max per-tick joint change, rad (override with --max-delta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--ramp-seconds", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=10.0, help="live-loop run length, seconds")
    parser.add_argument("--slow", type=float, default=SLOW, help="slow-mode blend factor (lower = gentler)")
    parser.add_argument("--max-delta", type=float, default=MAX_DELTA, help="max per-tick joint change, rad")
    parser.add_argument("--start", choices=["default", "hand-posed"], default="hand-posed",
                        help="ramp start: 'default' = full reach from default pose (M3), "
                             "'hand-posed' = near the fixed point (M2 hold test)")
    parser.add_argument("--target", help="in-box EE target 'x,y,z' (default: built-in POSE_COMMAND)")
    args = parser.parse_args()

    pose_command = POSE_COMMAND if args.target is None else \
        [float(v) for v in args.target.split(",")] + [1.0, 0.0, 0.0, 0.0]

    policy = torch.jit.load(str(POLICY_PATH))
    policy.eval()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    start_pose = None
    try:
        start_pose = conversion.obs_to_joint_pos(robot.get_observation())[:5]

        start_target = (
            conversion.DEFAULT_POSE_RAD[:5] if args.start == "default"
            else HAND_POSED_TARGET_RAD[:5]
        )
        print(f"ramping to {args.start} start over {args.ramp_seconds:.0f} s...")
        go_to(robot, start_target, args.ramp_seconds)

        n_ticks = max(1, round(args.duration * HZ))
        print(f"reaching to {pose_command[:3]} for {n_ticks} ticks at {HZ:.0f} Hz...")
        reach_hold(robot, policy, pose_command, n_ticks, args.slow, args.max_delta,
                   hz=HZ, verbose=True)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if start_pose is not None:
            print("ramping back to rest...")
            go_to(robot, start_pose, args.ramp_seconds)
        robot.disconnect()
        print("torque off.")


if __name__ == "__main__":
    main()
