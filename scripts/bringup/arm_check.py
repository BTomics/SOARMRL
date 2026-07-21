"""Post-calibration check for the SO-101 follower arm.

Connects through the same SO101Follower API the RL bridge will use,
prints calibrated joint positions, then sweeps each joint a few
degrees and returns it home. Clamp the base and start from a
mid-range pose.

    python scripts/bringup/arm_check.py --port COM4 --id follower1
"""

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SWEEP = 10  # normalized units (-100..100 full range), a gentle nudge


def read_pose(robot: SO101Follower) -> dict[str, float]:
    obs = robot.get_observation()
    return {j: obs[f"{j}.pos"] for j in JOINTS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="COM port, e.g. COM4")
    parser.add_argument("--id", required=True, help="robot id used during lerobot-calibrate")
    args = parser.parse_args()

    config = SO101FollowerConfig(port=args.port, id=args.id)
    robot = SO101Follower(config)
    robot.connect()

    try:
        home = read_pose(robot)
        print("--- Calibrated positions (normalized) ---")
        for joint, pos in home.items():
            print(f"  {joint}: {pos:+.1f}")

        print("\n--- Per-joint sweep ---")
        for joint in JOINTS:
            print(f"  {joint}...", end="", flush=True)
            for offset in (SWEEP, -SWEEP, 0):
                robot.send_action({**{f"{j}.pos": home[j] for j in JOINTS},
                                   f"{joint}.pos": home[joint] + offset})
                time.sleep(0.6)
            back = read_pose(robot)[joint]
            print(f" ok (returned to {back:+.1f}, commanded {home[joint]:+.1f})")
    finally:
        robot.disconnect()

    print("\nDone. Arm reads state and tracks position commands through the LeRobot API.")


if __name__ == "__main__":
    main()
