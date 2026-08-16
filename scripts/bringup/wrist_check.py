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
    args = ap.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    start_rad, _ = read(robot)
    print(f"start: " + "  ".join(f"{j}={r:+.2f}" for j, r in zip(conversion.JOINTS, start_rad)))

    try:
        print(f"\nramping to PickPlace home over {args.seconds:.0f}s -- SUPPORT THE ARM")
        go_to(robot, PICKPLACE_HOME, args.seconds)
        time.sleep(0.5)

        rad, norm = read(robot)
        wf_rad, wf_n = rad[3], norm[3]
        print("\ncommanded wrist_flex 1.570 rad (normalized 120.0)")
        print(f"reached   wrist_flex {wf_rad:.3f} rad (normalized {wf_n:.1f})")
        if wf_n >= 99.0:
            print("  -> SATURATED at the servo's range_max, as predicted")
        else:
            print("  -> did NOT saturate; the range assumption is wrong, tell Claude")

        print("\nNOW MEASURE: hold a phone level flat against the gripper.")
        print("  ~6 deg from vertical  -> calibration is WRONG (joint reached 1.57)")
        print("  ~28 deg from vertical -> servo really is short (stopped at 1.195)")
        input("\npress Enter to ramp back...")
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
