"""Waypoint test: reach 5 named, predetermined in-box targets in sequence.

Ramps to default, then reaches each named waypoint via bridge.reach_hold for a
fixed dwell, printing the target and the settled joint pos. Fixed points (not
random) at round, spatially distinct spots so the motion is easy to recognize
against docs/reach_targets viz. Ctrl+C or completion ramps back to rest.

    python scripts/reach/waypoints.py --port COM8 --id follower1
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

# Named in-box EE waypoints (base frame xyz), visited in this order. Round,
# spatially distinct spots -- keep in sync with the viz's WP list.
WAYPOINTS = [
    ("CENTER",   [0.00, -0.175, 0.20]),
    ("LOW-NEAR", [0.00, -0.12, 0.12]),
    ("HIGH-FAR", [0.00, -0.24, 0.28]),
    ("LEFT",     [-0.15, -0.175, 0.20]),
    ("RIGHT",    [0.09, -0.175, 0.20]),
]

HZ = 30.0
SLOW = 0.25
MAX_DELTA = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--ramp-seconds", type=float, default=5.0)
    parser.add_argument("--dwell", type=float, default=4.0, help="seconds to reach/settle per waypoint")
    parser.add_argument("--slow", type=float, default=SLOW)
    parser.add_argument("--max-delta", type=float, default=MAX_DELTA)
    args = parser.parse_args()

    policy = torch.jit.load(str(POLICY_PATH))
    policy.eval()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    start_pose = None
    try:
        start_pose = conversion.obs_to_joint_pos(robot.get_observation())[:5]

        print(f"ramping to default over {args.ramp_seconds:.0f} s...")
        go_to(robot, conversion.DEFAULT_POSE_RAD[:5], args.ramp_seconds)

        dwell_ticks = max(1, round(args.dwell * HZ))
        for i, (name, xyz) in enumerate(WAYPOINTS, 1):
            pose_command = xyz + [1.0, 0.0, 0.0, 0.0]
            print(f"\n[{i}/{len(WAYPOINTS)}] -> {name}  {xyz}")
            settled = reach_hold(robot, policy, pose_command, dwell_ticks,
                                 args.slow, args.max_delta, hz=HZ)
            labeled = "  ".join(f"{j}={p:+.3f}" for j, p in zip(conversion.JOINTS, settled))
            print(f"      settled: {labeled}")
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
