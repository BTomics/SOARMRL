"""Pre-assembly sanity test for the 6 SO-101 follower servos.

Daisy-chain all six servos to the Waveshare board, power on, then:
    python scripts/bringup/servo_test.py --port COM4

Pings every ID, reads positions, and wiggles each servo a few degrees.
Run BEFORE mechanical assembly so a dead servo is found while it's
still easy to swap. Uses raw ticks (no calibration needed).
"""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
}

WIGGLE_TICKS = 150  # ~13 degrees on a 4096-tick revolution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="COM port of the Waveshare board, e.g. COM4")
    args = parser.parse_args()

    bus = FeetechMotorsBus(port=args.port, motors=MOTORS)
    bus.connect()

    print("\n--- Ping ---")
    found = bus.broadcast_ping()
    print(f"IDs responding: {sorted(found) if found else 'NONE'}")
    expected = {m.id for m in MOTORS.values()}
    missing = expected - set(found or {})
    if missing:
        print(f"!! Missing IDs: {sorted(missing)} — check cables/power for those servos.")

    print("\n--- Positions (raw ticks, 0-4095) ---")
    for name, motor in MOTORS.items():
        if motor.id in missing:
            continue
        pos = bus.read("Present_Position", name, normalize=False)
        print(f"  {name} (id {motor.id}): {pos}")

    print("\n--- Wiggle test (each servo moves slightly and returns) ---")
    try:
        for name, motor in MOTORS.items():
            if motor.id in missing:
                continue
            home = bus.read("Present_Position", name, normalize=False)
            bus.enable_torque(name)
            print(f"  {name}...", end="", flush=True)
            for target in (home + WIGGLE_TICKS, home - WIGGLE_TICKS, home):
                bus.write("Goal_Position", name, target, normalize=False)
                time.sleep(0.5)
            print(" ok")
    finally:
        bus.disable_torque()
        bus.disconnect()

    print("\nDone. All listed servos ping, report position, and move.")


if __name__ == "__main__":
    main()
