"""Grasp bridge milestone G1: offline sanity check of the exported lift policy.

The 28-dim analogue of scripts/reach/m1_sanity.py. No hardware. Verifies:
  - the checkpoint's input layer is 28-wide and it emits 6 actions
  - grasp_bridge.build_grasp_obs assembles a 28-vector from a synthetic
    "arm at the lift default pose" reading (this is what catches obs-layout bugs)
  - at the default pose the joint_pos_rel block is ~0
  - the policy returns 6 finite values, and a garbage obs gives a different one

Ground truth: policies/lift_params/env.yaml (verified 2026-07-21).

    python scripts/grasp/g1_sanity.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import torch

from soarmrl import conversion
from soarmrl import grasp_bridge as gb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_PATH = REPO_ROOT / "policies" / "lift_policy.pt"

# A synthetic get_observation() at the LIFT default pose -> joint_pos_rel == 0.
DEFAULT_OBS = {
    f"{j}.pos": conversion.rad_to_n(j, gb.LIFT_DEFAULT_POSE_RAD[i])
    for i, j in enumerate(conversion.JOINTS)
}
KNOWN_OBJECT_POS = [0.0, -0.20, 0.02]                  # cube on the table, base frame
TARGET_POSE = [0.0, -0.20, 0.28, 1.0, 0.0, 0.0, 0.0]   # in the air, within lift ranges


def main() -> None:
    policy = torch.jit.load(str(POLICY_PATH))
    policy.eval()

    weights = [p for p in policy.parameters() if p.dim() == 2]
    assert weights[0].shape[1] == gb.OBS_DIM, (
        f"expected obs dim {gb.OBS_DIM}, checkpoint wants {weights[0].shape[1]}"
    )
    assert weights[-1].shape[0] == gb.ACTION_DIM, (
        f"expected action dim {gb.ACTION_DIM}, checkpoint emits {weights[-1].shape[0]}"
    )
    print(f"checkpoint OK: obs {gb.OBS_DIM} -> action {gb.ACTION_DIM}")

    # Build the obs through the real code path; at the default pose pos/vel are ~0.
    obs = gb.build_grasp_obs(
        DEFAULT_OBS, KNOWN_OBJECT_POS, TARGET_POSE,
        last_action=[0.0] * gb.ACTION_DIM,
        prev_joint_pos=list(gb.LIFT_DEFAULT_POSE_RAD),
        dt=1.0 / 30.0,
    )
    assert len(obs) == gb.OBS_DIM, f"build_grasp_obs made {len(obs)} dims, need {gb.OBS_DIM}"
    assert max(abs(v) for v in obs[0:6]) < 1e-6, f"pos_rel not ~0 at default pose: {obs[0:6]}"
    print(f"build_grasp_obs OK: {len(obs)} dims, pos_rel~0 at default pose")

    obs[6:12] = [0.0] * 6  # the same velocity zeroing grasp_hold applies
    with torch.no_grad():
        action = policy(torch.tensor([obs], dtype=torch.float32)).squeeze(0).tolist()
    assert len(action) == gb.ACTION_DIM and all(map(math.isfinite, action)), action
    print(f"policy output ({gb.ACTION_DIM}): {[round(a, 3) for a in action]}")

    arm, gripper = gb.scale_grasp_action(action)
    assert len(arm) == 5 and gripper in (gb.GRIPPER_OPEN_RAD, gb.GRIPPER_CLOSE_RAD)
    print(f"scaled: arm={[round(a, 3) for a in arm]}  gripper={gripper} rad")

    # A garbage obs must produce a different action (confidence the layout matters).
    garbage = [5.0] * gb.OBS_DIM
    with torch.no_grad():
        garbage_action = policy(torch.tensor([garbage], dtype=torch.float32)).squeeze(0).tolist()
    assert garbage_action != action, "garbage obs gave identical output — obs is being ignored"
    print("garbage-obs differs OK")


if __name__ == "__main__":
    main()
