# Laptop Setup — LeRobot Hardware Control (Windows)

The laptop talks to the arm: servo ID setup (before assembly!), calibration, and later the deployment bridge. No training here.

## 1. Environment

```powershell
conda create -n soarmrl python=3.10 -y
conda activate soarmrl
pip install "lerobot[feetech]"
python -c "import lerobot; print(lerobot.__version__)"
```

If the PyPI package lags a needed fix, the fallback is the source install from the [LeRobot docs](https://huggingface.co/docs/lerobot/installation): clone `huggingface/lerobot`, then `pip install -e ".[feetech]"`.

## 2. Find the COM port (when the board arrives)

Plug the Waveshare board in via USB **and** connect its power supply. On Windows the port shows up as `COMx` (Device Manager → Ports). Or let LeRobot find it:

```powershell
lerobot-find-port
```

If the board doesn't enumerate at all, install the USB-serial driver for its chip (CP210x or CH343, check the Waveshare wiki page for the board).

⚠️ Waveshare board: both jumpers must be on the **B** channel (USB), or the motors won't respond.

## 3. Set servo IDs — one motor at a time, BEFORE assembly

Every STS3215 ships as ID 1. The script walks through the motors one by one (it starts with the gripper, ID 6). Only ever have **one** motor wired to the board during this step, not daisy-chained to anything.

```powershell
lerobot-setup-motors --robot.type=so101_follower --robot.port=COM4
```

IDs and baudrate are written to the servo's EEPROM — this is done once, ever.

## 4. Calibrate — after assembly

```powershell
lerobot-calibrate --robot.type=so101_follower --robot.port=COM4 --robot.id=soarm_follower
```

Procedure (video in the [SO-101 docs](https://huggingface.co/docs/lerobot/so101)): move every joint to mid-range, press Enter, then sweep each joint through its full range.

## 5. Verify

```python
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

follower = SO101Follower(SO101FollowerConfig(port="COM4", id="soarm_follower"))
follower.connect()
print(follower.get_observation())   # live joint positions — move the arm by hand and re-read
follower.disconnect()
```

Then sweep each joint gently through part of its range with position commands before ever running a policy.
