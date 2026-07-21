"""Forward kinematics of the sim default pose, straight from the URDF.

WHY: general FK reference/cross-check against the URDF. NOT a source of a
usable pose_command -- computed default-pose xyz lands far outside the
trained command box (x/y/z ranges in reach_env_cfg.py CommandsCfg), because
the default "candle" posture was never meant to already be near a sampled
reach target. A "policy should barely move" hardware test instead needs the
arm physically positioned (via go_to, hand-picked joints) at a point that IS
inside the box, with pose_command set to that same point.

URDF: ../../../isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf
Joint chain (base -> EE), matches conversion.JOINTS order:
    base_link -[shoulder_pan]-> shoulder_link
    shoulder_link -[shoulder_lift]-> upper_arm_link
    upper_arm_link -[elbow_flex]-> lower_arm_link
    lower_arm_link -[wrist_flex]-> wrist_link
    wrist_link -[wrist_roll]-> gripper_link   (true EE frame -- confirmed via
        isaac_so_arm101 tasks/reach/joint_pos_env_cfg.py:92,
        SoArm101ReachEnvCfg.commands.ee_pose.body_name = ["gripper_link"])
(gripper_frame_joint/link and the "gripper" revolute jaw joint are NOT part
of this chain -- the trained policy's EE frame stops at gripper_link)

    python scripts/tools/fk_default_pose.py
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from soarmrl import conversion

URDF_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "isaac_so_arm101"
    / "src"
    / "isaac_so_arm101"
    / "robots"
    / "trs_so101"
    / "urdf"
    / "so_arm101.urdf"
)

# Order of joints to walk, base to EE (gripper_link -- see module docstring).
# Angles come from conversion.DEFAULT_POSE_RAD by name.
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def parse_joint_origins(urdf_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """joint name -> (xyz origin, rpy origin) from the URDF, both as float arrays.

    TODO: ET.parse, root.findall('joint'), for each joint of interest read
    its <origin xyz="..." rpy="..."/> child (default to zeros if missing),
    and its <axis xyz="..."/> for revolute joints (store alongside, or in a
    second dict -- your call).
    """
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    origins = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in CHAIN:
            continue
            
        origin = joint.find("origin")
        if origin is not None:
            xyz = np.array([float(x) for x in origin.get("xyz", "0 0 0").split()])
            rpy = np.array([float(x) for x in origin.get("rpy", "0 0 0").split()])
        else:
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            
        axis_elem = joint.find("axis")
        if axis_elem is not None:
            axis = np.array([float(x) for x in axis_elem.get("xyz", "1 0 0").split()])
        else:
            axis = None
            
        origins[name] = (xyz, rpy, axis)
        
    return origins

def joint_transform(xyz: np.ndarray, rpy: np.ndarray, axis: np.ndarray | None, angle: float) -> np.ndarray:
    """4x4 homogeneous transform for one joint: fixed origin offset (xyz, rpy)
    composed with a rotation of `angle` about `axis` (axis=None -> fixed joint,
    no rotation, e.g. gripper_frame_joint).

    TODO: build the fixed-origin transform from xyz/rpy (roll-pitch-yaw ->
    rotation matrix, translation xyz), build the joint rotation transform
    (Rodrigues' formula or scipy.spatial.transform.Rotation about `axis`),
    return their product in the right order (origin first, then joint
    rotation -- URDF convention).
    """
    T_origin = np.eye(4)
    # URDF rpy is extrinsic fixed-axis x-y-z
    T_origin[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T_origin[:3, 3] = xyz
    
    T_joint = np.eye(4)
    if axis is not None:
        T_joint[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        
    return T_origin @ T_joint


def forward_kinematics(joint_angles_rad: dict[str, float], urdf_path: Path = URDF_PATH) -> np.ndarray:
    """Compose CHAIN's transforms base->EE, return the final 4x4 pose.

    TODO: parse_joint_origins once, fold joint_transform over CHAIN
    (T = T @ joint_transform(...) for each joint in order), return T.
    """
    origins = parse_joint_origins(urdf_path)
    T = np.eye(4)
    
    for name in CHAIN:
        xyz, rpy, axis = origins[name]
        angle = joint_angles_rad.get(name, 0.0)
        T = T @ joint_transform(xyz, rpy, axis, angle)
        
    return T


def main() -> None:
    default_angles = dict(zip(conversion.JOINTS, conversion.DEFAULT_POSE_RAD))

    T = forward_kinematics(default_angles)
    xyz = T[:3, 3]
    # scalar_first=True gives (w, x, y, z) -- scipy's plain as_quat() default
    # is (x, y, z, w) and would silently mislabel this.
    quat_wxyz = Rotation.from_matrix(T[:3, :3]).as_quat(scalar_first=True)
    print(f"default-pose EE xyz (gripper_link, base frame): {xyz.round(5)}")
    print(f"default-pose EE quat (w,x,y,z): {quat_wxyz.round(5)}")
    print(
        "Note: this is far outside the trained command box "
        "(x in [-0.1, 0.1], y in [-0.25, -0.1], z in [0.1, 0.3]) -- expected, "
        "see module docstring. Do not use this xyz directly as pose_command."
    )


if __name__ == "__main__":
    main()
