"""Teach-and-replay, teach half: record the arm poses for a scripted pick-up.

The lift RL policy limit-cycles on hardware (it saturates from the upright home
and thrashes across the whole throttle range — see the 2026-07-26 session log),
so the pick-up is scripted instead of policy-driven. Hand-pose the arm through a
few waypoints; this records the joint angles. The replay half (soarmrl.grasp)
drives go_to through them, closes the gripper, confirms the hold with
grasp_bridge.is_grasping, and lifts.

Kinesthetic teaching: torque is DISABLED so you can move the arm by hand.
  - SUPPORT the arm before releasing torque and the whole time — it goes limp
    and will drop under gravity otherwise.
  - Keep the gripper OPEN while posing; the close is scripted at replay time.
  - For each waypoint, hold the pose steady and press enter to capture (with
    torque off the arm sags if you let go, so capture while holding it).

Waypoints (default):
  approach : above the cube, jaws open, with a clear straight-down path to it
  grasp    : jaws straddling the cube at table height, ready to close
  lift     : raised clear of the table with the (imagined) cube in the jaws

    python scripts/grasp/teach.py --port COM8 --id follower1
    python scripts/grasp/teach.py --port COM8 --id follower1 --out scripts/grasp/grasp_waypoints.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from soarmrl import conversion

WAYPOINTS = ["approach", "grasp", "lift"]
DEFAULT_OUT = Path(__file__).resolve().parent / "grasp_waypoints.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the waypoint JSON")
    parser.add_argument("--names", help="comma-separated waypoint names (default: approach,grasp,lift)")
    args = parser.parse_args()

    names = WAYPOINTS if args.names is None else [n.strip() for n in args.names.split(",")]

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    recorded: dict[str, list[float]] = {}
    try:
        input("\nSUPPORT THE ARM, then press enter to release torque (arm goes limp)...")
        robot.bus.disable_torque()
        print("torque OFF — move the arm by hand. Keep the gripper open.\n")

        for name in names:
            input(f"[{name}] hold the pose steady and press enter to capture...")
            joint_pos = conversion.obs_to_joint_pos(robot.get_observation())
            recorded[name] = joint_pos
            labeled = "  ".join(f"{j}={v:+.3f}" for j, v in zip(conversion.JOINTS, joint_pos))
            print(f"  captured: {labeled}\n")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"joints": conversion.JOINTS, "waypoints": recorded}, indent=2))
        print(f"saved {len(recorded)} waypoints -> {out}")
    except KeyboardInterrupt:
        print("\ninterrupted — nothing saved" if not recorded else
              f"\ninterrupted — {len(recorded)} pose(s) captured but NOT saved")
    finally:
        # Leave torque off: the arm is limp where you left it. Support it.
        robot.disconnect()
        print("disconnected (torque off — support the arm before letting go).")


if __name__ == "__main__":
    main()
