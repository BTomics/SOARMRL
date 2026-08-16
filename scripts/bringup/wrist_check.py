"""Is the wrist_flex calibration honest? Commands the PickPlace home and measures.

The PickPlace policy's home pose wants wrist_flex = 1.57 rad. The calibrated
servo range tops out at 1.195 (range_max 3169 ticks), so ~48% of the policy's
wrist commands are unreachable and the lift bridge needed a 2x ACTION_SCALE to
compensate. Either the servo really is short, or n_at_default is wrong.

The encoder cannot answer this on its own -- it reports 100 at the physical stop
by definition, whatever angle that stop actually is. So measure the gripper angle
physically:

    gripper 6 deg from straight down   -> the joint DID reach 1.57
                                          => calibration is wrong, fix n_at_default
    gripper 28 deg from straight down  -> the joint stopped at 1.195
                                          => servo really is short, lower the sim
                                             home and retrain

    python scripts/bringup/wrist_check.py --port COM4 --id follower1

CLEAR THE WORKSPACE and clamp the base. This ramps to an arm-straight-out pose.
Ctrl+C at any point ramps back to where it started and releases torque.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from soarmrl import conversion
from soarmrl.bridge import go_to

# PickPlace home (isaac_so_arm101 SO_ARM101_CFG init_state), sim joint order.
PICKPLACE_HOME = [0.0, 0.0, 0.0, 1.57, 0.0]
SERVO_MAX_WRIST_RAD = conversion.n_to_rad("wrist_flex", 100.0)


def read(robot):
    pos = conversion.obs_to_joint_pos(robot.get_observation())
    obs = robot.get_observation()
    return pos, [obs[f"{j}.pos"] for j in conversion.JOINTS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--seconds", type=float, default=8.0, help="ramp time")
    ap.add_argument("--wrist", type=float, default=1.57,
                    help="wrist_flex target in rad; take two points to fit the offset")
    args = ap.parse_args()
    PICKPLACE_HOME[3] = args.wrist

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    start_rad, _ = read(robot)
    print(f"start: " + "  ".join(f"{j}={r:+.2f}" for j, r in zip(conversion.JOINTS, start_rad)))

    try:
        print(f"\nramping to PickPlace home over {args.seconds:.0f}s -- SUPPORT THE ARM")
        go_to(robot, PICKPLACE_HOME, args.seconds)
        time.sleep(0.5)

        rad, norm = read(robot)
        print(f"\ncommanded wrist_flex {args.wrist:.3f} rad "
              f"(normalized {conversion.rad_to_n('wrist_flex', args.wrist):.1f})")
        print("\nHELD POSE — all joints (the other pitch joints must read ~0 for the")
        print("gripper angle to isolate wrist_flex):")
        print(f"  {'joint':15}{'rad (per CALIB)':>18}{'normalized':>13}")
        for j, r, n in zip(conversion.JOINTS, rad, norm):
            flag = "  <-- should be ~0" if j in ("shoulder_lift", "elbow_flex") else ""
            print(f"  {j:15}{r:+18.3f}{n:13.1f}{flag}")

        print("\nNOW MEASURE the gripper angle from vertical with a phone level.")
        input("press Enter to ramp back...")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("ramping back")
        go_to(robot, start_rad[:5], args.seconds)
        robot.bus.disable_torque()
        robot.disconnect()
        print("torque off, disconnected")


if __name__ == "__main__":
    main()
