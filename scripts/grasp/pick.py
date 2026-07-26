"""Run the scripted pick-up: replay taught waypoints, close, confirm, lift.

Reads scripts/grasp/grasp_waypoints.json (from teach.py) and drives the
soarmrl.grasp sequence. On success the arm lifts the cube, waits, then places it
back and returns to rest; on a failed grasp or Ctrl+C it opens the gripper and
returns to where it started. Torque off on exit.

Arm clamped, cube on its taught spot, hand on the switch.

    python scripts/grasp/pick.py --port COM8 --id follower1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from soarmrl import conversion, grasp
from soarmrl import grasp_bridge as gb

DEFAULT_WP = Path(__file__).resolve().parent / "grasp_waypoints.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--waypoints", default=str(DEFAULT_WP))
    parser.add_argument("--ramp-seconds", type=float, default=3.0, help="return-to-rest ramp")
    args = parser.parse_args()

    waypoints = grasp.load_waypoints(args.waypoints)
    for name in ("approach", "grasp", "lift"):
        if name not in waypoints:
            raise SystemExit(f"waypoint '{name}' missing from {args.waypoints} -- run teach.py")

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    start_pose = None
    try:
        start_pose = conversion.obs_to_joint_pos(robot.get_observation())[:5]
        ok = grasp.pick(robot, waypoints, verbose=True)
        print("\nPICK OK -- cube lifted." if ok else "\nPICK FAILED -- no grasp confirmed.")

        if ok:
            input("press enter to place it back down and return to rest...")
            grasp.move_to(robot, waypoints["grasp"], gb.GRIPPER_CLOSE_RAD, 2.5)  # lower, still holding
            grasp.move_to(robot, waypoints["grasp"], gb.GRIPPER_OPEN_RAD, 0.5)   # release on the table
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if start_pose is not None:
            print("returning to rest (jaws open)...")
            grasp.move_to(robot, start_pose, gb.GRIPPER_OPEN_RAD, args.ramp_seconds)
        robot.disconnect()
        print("torque off.")


if __name__ == "__main__":
    main()
