"""Bridge recon: live joint readings while moving the arm by hand.

Torque is disabled after connect — SUPPORT THE ARM when it starts, the
shoulder will slump under gravity. Move one joint at a time and note the
units, sign direction, and the readings at known physical poses (especially
the sim default pose). Ctrl+C to exit.

    python scripts/bringup/recon.py --port COM4 --id follower1
"""

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="COM port, e.g. COM4")
    parser.add_argument("--id", required=True, help="robot id used during lerobot-calibrate")
    args = parser.parse_args()

    config = SO101FollowerConfig(port=args.port, id=args.id)
    robot = SO101Follower(config)
    robot.connect()
    robot.bus.disable_torque()

    print("Torque OFF — hold the arm. Move joints by hand; Ctrl+C to stop.\n")
    header = "".join(f"{j:>14}" for j in JOINTS)
    print(header)

    try:
        while True:
            obs = robot.get_observation()
            line = "".join(f"{obs[f'{j}.pos']:>14.2f}" for j in JOINTS)
            print(f"\r{line}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        robot.disconnect()


if __name__ == "__main__":
    main()
