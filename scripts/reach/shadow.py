"""Shadow mode: run the reach policy against live encoder reads WITHOUT
sending its actions to the arm. Sanity-checks the obs layout and the
wrist_roll sign (bridge.py note b) before anything is allowed to move.

Sequence: rest -> ramp to default pose (go_to) -> loop at 30 Hz, printing
what the policy WOULD command, arm untouched -> Ctrl+C or timeout ramps
back to rest.

Watch the wrist_roll column: a wrong sign shows up as a large, roughly
constant correction on that one joint (bridge.py note b) -- flip
CALIB["wrist_roll"] direction in conversion.py if seen.

    python scripts/reach/shadow.py --port COM4 --id follower1
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from soarmrl import conversion
from soarmrl.bridge import builds_obs, go_to

POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "policies" / "policy.pt"

# Hand-posed joint target (rad, sim order) that physically puts the gripper
# at POSE_COMMAND below -- measured 2026-07-20 via encoder readback.
HAND_POSED_TARGET_RAD = [0.3279, 1.4139, -1.8310, 0.7829, -0.2285, 0.0292]

# M2 hold-still EE target (base frame xyz + identity quaternion) -- the
# point HAND_POSED_TARGET_RAD physically puts the gripper at. See bridge.py.
POSE_COMMAND = [0.0, -0.221, 0.212, 1.0, 0.0, 0.0, 0.0]

HZ = 30.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--ramp-seconds", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=10.0, help="shadow-mode run length, seconds")
    args = parser.parse_args()

    policy = torch.jit.load(str(POLICY_PATH))
    policy.eval()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    start_pose = None
    try:
        start_pose = conversion.obs_to_joint_pos(robot.get_observation())[:5]

        print(f"ramping to hand-posed target over {args.ramp_seconds:.0f} s...")
        go_to(robot, HAND_POSED_TARGET_RAD[:5], args.ramp_seconds)

        n_ticks = max(1, round(args.duration * HZ))
        dt = 1.0 / HZ
        print(f"shadow mode: {n_ticks} ticks at {HZ:.0f} Hz -- NOT sending actions, arm stays put")

        prev_joint_pos = conversion.obs_to_joint_pos(robot.get_observation())
        last_action = [0.0] * 5

        for tick in range(n_ticks):
            t_start = time.perf_counter()

            observation = robot.get_observation()
            obs = builds_obs(observation, POSE_COMMAND, last_action, prev_joint_pos, dt)
            assert len(obs) == 24, f"obs shape wrong: {len(obs)}"

            with torch.no_grad():
                action = policy(torch.tensor([obs], dtype=torch.float32)).squeeze(0).tolist()

            labeled = "  ".join(f"{j}={a:+.3f}" for j, a in zip(conversion.ARM_JOINTS, action))
            print(f"tick {tick:4d}: {labeled}")

            prev_joint_pos = conversion.obs_to_joint_pos(observation)
            last_action = action

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, dt - elapsed))
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
