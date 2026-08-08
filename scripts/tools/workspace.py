"""Where on the table can the SO-101 actually work, and how comfortably.

Sizes the PickPlace spawn/goal boxes from the URDF instead of guessing. Two
questions, both answered by one batched FK sweep over the soft joint limits:

  density   How many arm configurations put the fingertip at a given target?
            A point reachable by exactly ONE contorted pose still counts as
            "reachable" — density is what separates comfortable from cramped.
  cos_down  Is a TOP-DOWN grasp available there? cos_down is the quantity
            grasp_top_down rewards: -(gripper approach axis)_z, +1 straight down.

    python scripts/tools/workspace.py                    # both, default 3M samples
    python scripts/tools/workspace.py --samples 500000   # quicker, noisier
    python scripts/tools/workspace.py --z 0.20           # slice at another height

WHY THIS EXISTS: the goal-box x floor was once raised on a measurement showing
near targets were config-starved. That sweep was taken with the goal box AIRBORNE
at z 0.06-0.20. At TABLE height the relationship INVERTS — density is highest
near the base — and the genuinely cramped region is the outer y corners. Re-run
this before quoting any workspace number; the answer depends on the height you
ask about.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from soarmrl.kinematics import CHAIN, _URDF_PATH, _chain_origins  # noqa: E402

# gripper_link -> fingertip, matching the task's ee_frame offset. Same vector is
# the gripper's approach axis; its SIGN is empirical and FK-verified (the raw
# offset gives cos_down +0.994 at the arm's own home pose). Do not re-derive it.
EE_OFFSET = np.array([0.01, 0.0, -0.09])
CUBE_Z = 0.015          # cube centre at rest on the table (3 cm cube)
SOFT_MARGIN = 0.05      # keep off the hard joint stops


def sample(n: int, seed: int = 0):
    """Uniform draw over soft joint limits -> (fingertip xyz, cos_down, |joint angles|)."""
    root = ET.parse(Path(_URDF_PATH)).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    rng = np.random.default_rng(seed)

    angles = {}
    for name in CHAIN:
        lim = joints[name].find("limit")
        lo, hi = float(lim.get("lower")), float(lim.get("upper"))
        m = SOFT_MARGIN * (hi - lo)
        angles[name] = rng.uniform(lo + m, hi - m, n)

    parsed = _chain_origins()
    T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for name in CHAIN:
        T_origin, axis = parsed[name]
        T_joint = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        T_joint[:, :3, :3] = Rotation.from_rotvec(axis[None, :] * angles[name][:, None]).as_matrix()
        T = T @ T_origin[None] @ T_joint

    R = T[:, :3, :3]
    tip = (R @ EE_OFFSET[None, :, None]).squeeze(-1) + T[:, :3, 3]
    approach = EE_OFFSET / np.linalg.norm(EE_OFFSET)
    cos_down = -(R @ approach[None, :, None]).squeeze(-1)[:, 2]

    above = tip[:, 2] >= 0.0  # a fingertip inside the table is not a pose
    return tip[above], cos_down[above], {k: np.abs(v)[above] for k, v in angles.items()}


def report(tip, cos_down, angles, z: float, ball: float):
    print(f"local sweep within {ball*100:.1f} cm of a target at (x, y=0, z={z})\n")
    print(f"{'x':>6s} {'n':>7s} {'density':>9s} {'cos_down':>9s} {'>0.85':>8s} "
          f"{'|elbow|':>9s} {'|sh_lift|':>10s}")
    rows = []
    for x in np.arange(0.08, 0.33, 0.02):
        m = np.linalg.norm(tip - np.array([x, 0.0, z]), axis=1) < ball
        if m.sum() < 30:
            continue
        rows.append((x, int(m.sum()), cos_down[m], angles["elbow_flex"][m],
                     angles["shoulder_lift"][m]))
    best = max(r[1] for r in rows)
    for x, n, c, ef, sl in rows:
        print(f"{x:6.2f} {n:7d} {n/best*100:8.1f}% {c.mean():9.3f} {(c>0.85).mean()*100:7.1f}% "
              f"{ef.mean():9.3f} {sl.mean():10.3f}")
    print("\ndensity is relative to the densest band; cos_down is the mean over "
          "configurations\nreaching that point. The MAX is ~1.000 everywhere, so "
          "top-down is always\navailable - a low mean means the policy has to be "
          "steered into it, not that\nit is impossible.")


def compare_boxes(tip, cos_down, z: float, ball: float, seed: int = 0):
    boxes = [
        ("pre-1f spawn  x[0.10,0.30] y+-0.25", (0.10, 0.30), (-0.25, 0.25)),
        ("1f spawn/goal x[0.15,0.30] y+-0.20", (0.15, 0.30), (-0.20, 0.20)),
    ]
    rng = np.random.default_rng(seed)
    print(f"\naveraged over uniform targets inside each box (z={z}):")
    for label, xr, yr in boxes:
        targets = np.stack([rng.uniform(*xr, 400), rng.uniform(*yr, 400),
                            np.full(400, z)], axis=1)
        dens, mean_c, share = [], [], []
        for t in targets:
            m = np.linalg.norm(tip - t, axis=1) < ball
            if m.sum() >= 30:
                dens.append(m.sum())
                mean_c.append(cos_down[m].mean())
                share.append((cos_down[m] > 0.85).mean())
        print(f"  {label}:  density {np.mean(dens):7.1f}   cos_down {np.mean(mean_c):.3f}   "
              f"share>0.85 {np.mean(share)*100:5.2f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=3_000_000)
    p.add_argument("--ball", type=float, default=0.025, help="target radius, metres")
    p.add_argument("--z", type=float, default=CUBE_Z, help="height of the slice, metres")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tip, cos_down, angles = sample(args.samples, args.seed)
    print(f"{args.samples} samples, {len(tip)} with the fingertip above the table "
          f"({len(tip)/args.samples*100:.1f}%)\n")
    report(tip, cos_down, angles, args.z, args.ball)
    compare_boxes(tip, cos_down, args.z, args.ball, args.seed)


if __name__ == "__main__":
    main()
