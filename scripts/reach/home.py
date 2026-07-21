"""Homing ramp test: rest pose -> default (crane) pose -> back to rest.

First hardware run of the conversion module. Arm clamped to the table,
workspace clear, hand near the power switch. Default ramp is slow (8 s);
Ctrl+C ramps back down before releasing torque.

    python scripts/reach/home.py --port COM4 --id follower1
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from soarmrl import conversion
from soarmrl.bridge import go_to


def print_pose(label: str, robot: SO101Follower) -> list[float]:
    pose = conversion.obs_to_joint_pos(robot.get_observation())
    print(f"{label}: " + "  ".join(f"{j}={r:+.2f}" for j, r in zip(conversion.JOINTS, pose)))
    return pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    start_pose = None
    try:
        start_pose = print_pose("rest pose (rad)", robot)[:5]

        print(f"ramping to default pose over {args.seconds:.0f} s...")
        go_to(robot, conversion.DEFAULT_POSE_RAD[:5], args.seconds)

        arrived = print_pose("arrived (rad)", robot)
        errors = [a - t for a, t in zip(arrived[:5], conversion.DEFAULT_POSE_RAD[:5])]
        print("error vs default: " + "  ".join(f"{e:+.3f}" for e in errors))

        print("holding 3 s — check the pose against the crane shape...")
        time.sleep(3.0)
    except KeyboardInterrupt:
        print("\ninterrupted — returning to rest before torque-off")
    finally:
        if start_pose is not None:
            print("ramping back to rest...")
            go_to(robot, start_pose, args.seconds)
        robot.disconnect()
        print("torque off.")


if __name__ == "__main__":
    main()
