"""Shadow mode for the lift/grasp policy: run it against live encoder reads
WITHOUT sending its actions to the arm. First hardware contact for the 28-dim
policy — sanity-checks the obs layout on real data and shows what it WOULD
command (arm AND the binary gripper) before anything is allowed to move.

Sequence: rest -> ramp to LIFT_DEFAULT_POSE_RAD (go_to) -> loop at 30 Hz,
printing the commanded action, arm untouched -> Ctrl+C or timeout ramps to rest.

The ramp is the first, supervised check that the lift home pose [0,0,0,1.57,0,0]
(more upright than the reach home) is safe to enter — hand-pose toward it and
clear the workspace first. Watch the wrist_roll column: a wrong calibration sign
shows as a large, roughly constant correction (bridge.py note b) -> flip
CALIB["wrist_roll"] in conversion.py if seen. Watch the grip column: it should
stay open (no cube in hand) — an early CLOSE means the obs is off.

    python scripts/grasp/shadow_grasp.py --port COM4 --id follower1
"""

import argparse
import sys
import time
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
KNOWN_OBJECT_POS = [0.22, 0.0, 0.015]

# The lift goal pose (base frame xyz + identity quaternion), inside the trained
# in-air ranges x[-0.1,0.1] y[-0.3,-0.1] z[0.2,0.35]. YOU choose it.
TARGET_POSE = [0.0, -0.15, 0.28, 1.0, 0.0, 0.0, 0.0]

HZ = 30.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--ramp-seconds", type=float, default=6.0)
    parser.add_argument("--duration", type=float, default=10.0, help="shadow-mode run length, seconds")
    args = parser.parse_args()

    policy = torch.jit.load(str(POLICY_PATH))
    policy.eval()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    start_pose = None
    try:
        start_pose = conversion.obs_to_joint_pos(robot.get_observation())[:5]

        print(f"ramping to LIFT home {gb.LIFT_DEFAULT_POSE_RAD[:5]} over {args.ramp_seconds:.0f} s...")
        go_to(robot, gb.LIFT_DEFAULT_POSE_RAD[:5], args.ramp_seconds)

        n_ticks = max(1, round(args.duration * HZ))
        dt = 1.0 / HZ
        print(f"shadow mode: {n_ticks} ticks at {HZ:.0f} Hz -- NOT sending actions, arm stays put")

        prev_joint_pos = conversion.obs_to_joint_pos(robot.get_observation())
        last_action = [0.0] * gb.ACTION_DIM

        for tick in range(n_ticks):
            t_start = time.perf_counter()

            observation = robot.get_observation()
            obs = gb.build_grasp_obs(observation, KNOWN_OBJECT_POS, TARGET_POSE,
                                     last_action, prev_joint_pos, dt)
            obs[6:12] = [0.0] * 6  # match grasp_hold's velocity zeroing
            assert len(obs) == gb.OBS_DIM, f"obs shape wrong: {len(obs)}"

            with torch.no_grad():
                action = policy(torch.tensor([obs], dtype=torch.float32)).squeeze(0).tolist()

            _, gripper_rad = gb.scale_grasp_action(action)
            grip = "OPEN" if gripper_rad == gb.GRIPPER_OPEN_RAD else "CLOSE"
            labeled = "  ".join(f"{j}={a:+.3f}" for j, a in zip(conversion.ARM_JOINTS, action[:5]))
            print(f"tick {tick:4d}: {labeled}  grip={grip}({action[5]:+.2f})")

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
