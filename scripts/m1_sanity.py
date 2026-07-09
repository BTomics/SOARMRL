"""Bridge milestone M1: offline sanity check of the exported reach policy.

Verified against policies/reach_v0_params/env.yaml (train-time dump) and the
checkpoint's own weight shapes on 2026-07-09. This comment block is the
reference for everything in src/soarmrl/bridge.py.

Observation layout, N = 24 (concatenation order per env.yaml):
    [0:6]    joint_pos_rel   rad, joint angle - default pose, all 6 joints
    [6:12]   joint_vel_rel   rad/s
    [12:19]  pose_command    target x, y, z + quaternion (w, x, y, z),
                             robot base frame; orientation fixed at identity
    [19:24]  last_action     previous raw network output (5 arm joints)

Joint order (defines "joint i" everywhere, incl. the default pose below):
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

Default pose (rad): [0.0, 1.57, -1.57, 1.0, -1.57, 0.0]

Action: 5 outputs (gripper NOT actuated by the reach policy).
    target_joint_pos = default_pose[:5] + 0.5 * output      (scale = 0.5)

Other train-time facts:
    - policy rate 30 Hz (sim 60 Hz, decimation 2)
    - normalizer = Identity -> feed raw observations, no mean/std scaling
    - pose_command trained ranges (base frame): x in [-0.1, 0.1],
      y in [-0.25, -0.1] ("in front" is negative y), z in [0.1, 0.3]
    - joint pos/vel obs had uniform +/-0.01 noise in training (robustness
      margin; do not replicate on hardware)
"""

from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / "policies" / "policy.pt"

OBS_DIM = 24
ACTION_DIM = 5
POSE_CMD = slice(12, 19)
CENTERED_TARGET = [0.0, -0.175, 0.2, 1.0, 0.0, 0.0, 0.0]


def main() -> None:
    policy = torch.jit.load(str(POLICY_PATH))
    policy.eval()

    weights = [p for p in policy.parameters() if p.dim() == 2]
    assert weights[0].shape[1] == OBS_DIM, (
        f"expected obs dim {OBS_DIM}, checkpoint wants {weights[0].shape[1]}"
    )
    assert weights[-1].shape[0] == ACTION_DIM, (
        f"expected action dim {ACTION_DIM}, checkpoint emits {weights[-1].shape[0]}"
    )
    print(f"checkpoint OK: obs {OBS_DIM} -> action {ACTION_DIM}")

    obs = torch.zeros(1, OBS_DIM)
    obs[0, POSE_CMD] = torch.tensor(CENTERED_TARGET)

    left = obs.clone()
    left[0, 12] = -0.1
    right = obs.clone()
    right[0, 12] = 0.1

    with torch.no_grad():
        for name, o in [("centered", obs), ("left", left), ("right", right)]:
            action = policy(o)
            assert torch.isfinite(action).all(), f"{name}: non-finite output"
            print(f"{name:>8}: {action.squeeze(0).tolist()}")

        pan_left = policy(left)[0, 0].item()
        pan_right = policy(right)[0, 0].item()
    assert pan_left * pan_right < 0, (
        f"mirror test failed: shoulder_pan {pan_left:.3f} vs {pan_right:.3f} "
        "did not flip sign"
    )
    print(f"mirror test OK: shoulder_pan {pan_left:.3f} (left) vs {pan_right:.3f} (right)")


if __name__ == "__main__":
    main()
