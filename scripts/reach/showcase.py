"""Showcase: sample N random in-box EE targets and reach each in turn.

Ramps to default once, then for each sampled point sets it as the pose_command
and runs bridge.reach_hold for a fixed dwell, letting the policy reach and
settle before moving to the next. Ctrl+C or completion ramps back to rest and
torques off. Reuses the same reach loop as m2_live.

    python scripts/reach/showcase.py --port COM8 --id follower1
    python scripts/reach/showcase.py --port COM8 --id follower1 --n 10 --seed 0
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from soarmrl import conversion
from soarmrl.bridge import go_to, reach_hold

POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "policies" / "policy.pt"

# Trained command box (base frame), from reach_env_cfg.py CommandsCfg ranges.
BOX_X = (-0.1, 0.1)
BOX_Y = (-0.25, -0.1)
BOX_Z = (0.1, 0.3)

HZ = 30.0
SLOW = 0.25
MAX_DELTA = 0.05


def sample_pose_command(rng: random.Random) -> list[float]:
    """One random in-box EE target: [x, y, z, 1, 0, 0, 0] (identity quat).

    TODO: draw x/y/z uniformly from BOX_X/BOX_Y/BOX_Z, append the identity quat.
    """
    x = rng.uniform(BOX_X[0], BOX_X[1])
    y = rng.uniform(BOX_Y[0], BOX_Y[1])
    z = rng.uniform(BOX_Z[0], BOX_Z[1])
    return [x, y, z, 1.0, 0.0, 0.0, 0.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--n", type=int, default=10, help="number of random targets")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for a repeatable demo")
    parser.add_argument("--ramp-seconds", type=float, default=5.0)
    parser.add_argument("--dwell", type=float, default=4.0, help="seconds to reach/settle per target")
    parser.add_argument("--slow", type=float, default=SLOW)
    parser.add_argument("--max-delta", type=float, default=MAX_DELTA)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    policy = torch.jit.load(str(POLICY_PATH))
    policy.eval()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    start_pose = None
    try:
        start_pose = conversion.obs_to_joint_pos(robot.get_observation())[:5]

        # TODO: ramp to the default pose once (go_to, conversion.DEFAULT_POSE_RAD[:5]).
        go_to(robot, conversion.DEFAULT_POSE_RAD[:5], args.ramp_seconds)

        dwell_ticks = max(1, round(args.dwell * HZ))
        for i in range(args.n):
            pose_command = sample_pose_command(rng)
            # TODO: print "target i/n -> xyz"; call
            #   reach_hold(robot, policy, pose_command, dwell_ticks,
            #              args.slow, args.max_delta, hz=HZ)
            # optionally report the settled joint pos it returns.
            settled_pos = reach_hold(robot, policy, pose_command, dwell_ticks,
                   args.slow, args.max_delta, hz=HZ)
            print("target", i + 1, "/", args.n, "→", pose_command[:3])
            print("settled joint pos:", "  ".join(f"{j}={p:+.3f}" for j, p in zip(conversion.ARM_JOINTS, settled_pos)))


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
