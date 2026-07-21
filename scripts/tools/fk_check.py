"""Forward-kinematics accuracy check for the reach bridge.

Feed a joint config (rad, sim order) and a URDF; prints the resulting
end-effector xyz + quaternion in the robot base frame. With --command it also
prints the error vs the commanded target.

Verifies the reach loop is accurate: take a SETTLED joint config from a run
(what the arm holds at) and FK it -- the EE it lands on should match the
pose_command that was sent. Use the SO-100 URDF: the deployed policy trained
against the SO-100 config (env.yaml body_names: gripper).

    python scripts/tools/fk_check.py                                   # FK self-test at the default pose
    python scripts/tools/fk_check.py --joints -0.056,1.065,-0.72,-0.69,-0.03,0.02 --command 0,-0.221,0.212
    python scripts/tools/fk_check.py --urdf so_arm101                  # switch model

Confirm the EE frame first (see CHAIN below): grep the URDF for the link named
'gripper' and check whether wrist_roll's child IS that link or a fixed joint
sits in between -- if so, add that fixed joint's name to CHAIN.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from soarmrl import conversion

ROBOTS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "isaac_so_arm101" / "src" / "isaac_so_arm101" / "robots"
)


def urdf_path_for(name: str) -> Path:
    """so_arm100 -> robots/trs_so100/urdf/so_arm100.urdf (SO-101 analogous)."""
    subdir = "trs_" + name.replace("so_arm", "so")
    return ROBOTS_DIR / subdir / "urdf" / f"{name}.urdf"

# Joints to compose, base -> EE. FK returns the pose of the LAST joint's child
# link -- confirm that link is the trained EE frame (SO-100 reward body:
# 'gripper'); append an intervening fixed joint's name if the URDF has one.
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def forward_kinematics(angles_by_name: dict, urdf_path: Path, chain: list[str]) -> np.ndarray:
    """Compose the chain's URDF transforms into a 4x4 base->EE pose."""
    joints = {j.get("name"): j for j in ET.parse(urdf_path).getroot().findall("joint")}
    T = np.eye(4)
    for name in chain:
        j = joints[name]
        origin = j.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else "0 0 0".split())])
        rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0").split() if origin is not None else "0 0 0".split())])
        axis_el = j.find("axis")
        T_origin = np.eye(4)
        T_origin[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        T_origin[:3, 3] = xyz
        T_joint = np.eye(4)
        if axis_el is not None:
            axis = np.array([float(v) for v in axis_el.get("xyz", "1 0 0").split()])
            T_joint[:3, :3] = Rotation.from_rotvec(axis * angles_by_name.get(name, 0.0)).as_matrix()
        T = T @ T_origin @ T_joint
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", help="6 comma-separated rad, sim order; default = the sim default pose")
    ap.add_argument("--command", help="x,y,z commanded target to compare the EE against")
    ap.add_argument("--urdf", default="so_arm100", help="URDF basename in the robots urdf dir (default so_arm100)")
    args = ap.parse_args()

    joints = ([float(v) for v in args.joints.split(",")] if args.joints
              else list(conversion.DEFAULT_POSE_RAD))
    if len(joints) != 6:
        ap.error(f"--joints needs 6 values (sim order {conversion.JOINTS}), got {len(joints)}")

    urdf_path = urdf_path_for(args.urdf)
    if not urdf_path.exists():
        ap.error(f"URDF not found: {urdf_path}\n(check --urdf name / ROBOTS_DIR)")

    angles = dict(zip(conversion.JOINTS, joints))
    T = forward_kinematics(angles, urdf_path, CHAIN)
    xyz = T[:3, 3]
    quat = Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True)

    print(f"URDF:  {urdf_path.name}   (EE frame = {CHAIN[-1]}'s child link)")
    print(f"joints (rad): {[round(a, 3) for a in joints]}")
    print(f"EE xyz (base): {xyz.round(4)}")
    print(f"EE quat wxyz:  {quat.round(4)}")
    if args.command:
        cmd = np.array([float(v) for v in args.command.split(",")])
        err = xyz - cmd
        print(f"commanded xyz: {cmd.round(4)}")
        print(f"error EE-cmd:  {err.round(4)}   |err| = {np.linalg.norm(err) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
