"""Characterize the gripper's grasp signals so the detector has real thresholds.

There is no perception on the arm, so "did I actually grab something?" has to
come from the gripper servo itself. The Feetech STS3215 exposes three tells,
only meaningful WHILE the gripper is commanded closed:
  - Present_Position -> finger opening (rad): an object blocks the jaws, so the
    fingers stall OPEN by ~the object width instead of reaching the empty-close
    position.
  - Present_Load     -> motor effort, sign-magnitude, |mag| 0..1000 (0..100%):
    empty close spikes then relaxes to ~0; holding an object stays elevated.
  - Present_Current  -> mA draw (same story as load).

This script does NOT run a policy. It ramps the arm to the lift home so the
gripper is out front, then runs supervised EMPTY vs HELD close/hold cycles and
prints the steady-state separation + a suggested threshold for each signal.
Feed those numbers into grasp_bridge.is_grasping.

    python scripts/grasp/grip_probe.py --port COM8 --id follower1

Arm clamped, gripper clear, hand on the switch. You'll be prompted to place /
remove the cube between cycles; when the gripper opens on a HELD cycle the cube
drops — catch it.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from soarmrl import conversion
from soarmrl import grasp_bridge as gb
from soarmrl.bridge import go_to

HZ = 30.0
HOLD_SECONDS = 2.0          # how long to hold each close while logging
SETTLE_FRACTION = 0.5       # use only the last half of a hold for steady-state stats
GRIPPER_IDX = conversion.JOINTS.index("gripper")


def read_gripper(robot) -> tuple[float, int, int]:
    """(gripper position rad, Present_Load, Present_Current) for one tick.

    Position comes through the normal obs path; load/current are extra
    single-motor bus reads (sign-magnitude already decoded by the bus)."""
    pos_rad = conversion.obs_to_joint_pos(robot.get_observation())[GRIPPER_IDX]
    load = robot.bus.read("Present_Load", "gripper")
    current = robot.bus.read("Present_Current", "gripper")
    return pos_rad, load, current


def close_and_log(robot, label: str) -> dict[str, list[float]]:
    """Command the gripper closed (arm held at lift home), log pos/load/current
    for HOLD_SECONDS, then open. Returns the per-tick samples."""
    arm = gb.LIFT_DEFAULT_POSE_RAD[:5]
    dt = 1.0 / HZ
    n = max(1, round(HOLD_SECONDS * HZ))
    samples = {"pos": [], "load": [], "current": []}

    print(f"  [{label}] closing...")
    robot.send_action(conversion.targets_to_action(arm, gb.GRIPPER_CLOSE_RAD))
    for tick in range(n):
        t0 = time.perf_counter()
        # keep re-commanding close so the servo holds effort against any object
        robot.send_action(conversion.targets_to_action(arm, gb.GRIPPER_CLOSE_RAD))
        pos, load, current = read_gripper(robot)
        samples["pos"].append(pos)
        samples["load"].append(load)
        samples["current"].append(current)
        if tick % 5 == 0:
            print(f"    tick {tick:3d}  pos={pos:+.3f} rad  load={load:+5d}  current={current:+5d}")
        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))

    print(f"  [{label}] opening...")
    robot.send_action(conversion.targets_to_action(arm, gb.GRIPPER_OPEN_RAD))
    time.sleep(0.5)
    return samples


def steady(samples: list[float]) -> list[float]:
    """The settled tail of a hold (drop the closing transient)."""
    cut = int(len(samples) * SETTLE_FRACTION)
    return samples[cut:]


def summarize(label: str, samples: dict[str, list[float]]) -> dict[str, float]:
    pos = steady(samples["pos"])
    load = [abs(v) for v in steady(samples["load"])]
    current = [abs(v) for v in steady(samples["current"])]
    stats = {
        "pos_med": statistics.median(pos),
        "load_med": statistics.median(load),
        "load_max": max(load),
        "current_med": statistics.median(current),
        "current_max": max(current),
    }
    print(f"\n  {label} steady-state (last {int(SETTLE_FRACTION * 100)}% of hold):")
    print(f"    gripper pos : median {stats['pos_med']:+.3f} rad")
    print(f"    load |mag|  : median {stats['load_med']:.0f}   max {stats['load_max']:.0f}   (0..1000)")
    print(f"    current     : median {stats['current_med']:.0f}   max {stats['current_max']:.0f}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--ramp-seconds", type=float, default=6.0)
    parser.add_argument("--cycles", type=int, default=2, help="repeats per EMPTY/HELD phase")
    args = parser.parse_args()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    robot.connect()

    start_pose = None
    try:
        start_pose = conversion.obs_to_joint_pos(robot.get_observation())[:5]
        print(f"ramping to lift home {gb.LIFT_DEFAULT_POSE_RAD[:5]} over {args.ramp_seconds:.0f} s...")
        go_to(robot, gb.LIFT_DEFAULT_POSE_RAD[:5], args.ramp_seconds)

        empty, held = [], []

        input("\nEMPTY phase: make sure the gripper is EMPTY, then press enter...")
        for c in range(args.cycles):
            print(f"\n-- EMPTY cycle {c + 1}/{args.cycles} --")
            empty.append(summarize(f"EMPTY {c + 1}", close_and_log(robot, "EMPTY")))

        input("\nHELD phase: hold the cube between the OPEN jaws, then press enter "
              "(gripper will close on it)...")
        for c in range(args.cycles):
            print(f"\n-- HELD cycle {c + 1}/{args.cycles} --  (jaws open, place cube)")
            input("  cube in place? press enter to close on it...")
            held.append(summarize(f"HELD {c + 1}", close_and_log(robot, "HELD")))

        # --- suggested thresholds ---
        e_pos = statistics.median([s["pos_med"] for s in empty])
        h_pos = statistics.median([s["pos_med"] for s in held])
        e_load = max(s["load_max"] for s in empty)
        h_load = statistics.median([s["load_med"] for s in held])

        print("\n" + "=" * 60)
        print("SUGGESTED THRESHOLDS for grasp_bridge.is_grasping")
        print("=" * 60)
        print(f"  empty-close pos : {e_pos:+.3f} rad   held-close pos : {h_pos:+.3f} rad")
        print(f"    -> GRIP_POS_STALL_RAD = {(e_pos + h_pos) / 2:+.3f}  "
              f"(held pos ABOVE this = fingers stalled on an object)")
        print(f"  empty-close load max : {e_load:.0f}   held-close load median : {h_load:.0f}")
        print(f"    -> GRIP_LOAD_MIN = {max(e_load * 1.5, (e_load + h_load) / 2):.0f}  "
              f"(load ABOVE this while commanded closed = holding)")
        print("=" * 60)
        print("Sanity: held pos should be clearly > empty pos, and held load >> empty load.")
        print("If they overlap, the cube is too small/soft for that signal — lean on the other.")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if start_pose is not None:
            print("\nramping back to rest...")
            go_to(robot, start_pose, args.ramp_seconds)
        robot.disconnect()
        print("torque off.")


if __name__ == "__main__":
    main()
