"""Forward kinematics for the SO-101 follower: sim-order joint angles -> EE xyz.

Ported from scripts/tools/fk_check.py. Loads the SO-101 URDF once (lazily) and
composes the base -> gripper_link chain. grasp_bridge uses this after a grasp to
track object_position ~ the end-effector: once the cube is grasped it rides with
the gripper, so a static object obs would lie and the policy never commits to the
lift (see grasp_bridge.object_position_for_tick / bridge.py).

No I/O beyond reading the URDF file; pure math otherwise.
"""

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from soarmrl.conversion import JOINTS

# base -> EE chain (EE = wrist_roll's child link = gripper_link on the SO-101).
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# SO-101 URDF in the sibling isaac_so_arm101 repo (same path fk_check.py uses).
_URDF_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "isaac_so_arm101" / "src" / "isaac_so_arm101" / "robots"
    / "trs_so101" / "urdf" / "so_arm101.urdf"
)


@lru_cache(maxsize=1)
def _chain_origins():
    """Parse the URDF once: joint name -> (T_origin 4x4, axis 3-vec or None)."""
    joints = {j.get("name"): j for j in ET.parse(_URDF_PATH).getroot().findall("joint")}
    parsed = {}
    for name in CHAIN:
        j = joints[name]
        origin = j.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else ["0", "0", "0"])]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split() if origin is not None else ["0", "0", "0"])]
        T_origin = np.eye(4)
        T_origin[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        T_origin[:3, 3] = xyz
        axis_el = j.find("axis")
        axis = np.array([float(v) for v in axis_el.get("xyz", "1 0 0").split()]) if axis_el is not None else None
        parsed[name] = (T_origin, axis)
    return parsed


def ee_position(joint_angles_rad) -> list[float]:
    """6 sim-order joint angles (rad, conversion.JOINTS order) -> gripper_link
    xyz in the robot base frame. Only the 5 arm joints in CHAIN affect the pose."""
    angles = dict(zip(JOINTS, joint_angles_rad))
    T = np.eye(4)
    for name, (T_origin, axis) in ((n, _chain_origins()[n]) for n in CHAIN):
        T_joint = np.eye(4)
        if axis is not None:
            T_joint[:3, :3] = Rotation.from_rotvec(axis * angles.get(name, 0.0)).as_matrix()
        T = T @ T_origin @ T_joint
    return T[:3, 3].tolist()
