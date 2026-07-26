"""Deployment bridge: exported Isaac Lab lift/grasp policy -> real SO-101.

SKELETON — signatures + contract only; the implementations are yours. The full
delta contract already lives in src/soarmrl/bridge.py (section "GRASP/LIFT
POLICY — BRIDGE DELTAS") and the sim-side task spec in docs/pickplace_contract.md.

This is the grasp-family peer of bridge.py (which deploys the reach policy):
the lift policy (policies/lift_policy.pt) deploys here now, and SO-ARM101-
PickPlace reuses this module unchanged once trained — its obs/action spec is
identical (maybe +1 gripper-state obs). Ground truth for every number below:
policies/lift_params/env.yaml, verified 2026-07-21.

Differences from the reach bridge (see bridge.py for the full table + reasons):
  - DIFFERENT home pose: LIFT_DEFAULT_POSE_RAD below, NOT the reach default in
    conversion.DEFAULT_POSE_RAD. use_default_offset=true, so it is both
    subtracted in joint_pos_rel and added back in the action offset.
  - obs 24 -> 28: adds object_position(3) and a 6-dim last_action.
  - action 5 -> 6: adds a binary gripper.
  - object_position has NO real-world sensor: static known spot pre-grasp,
    EE-from-FK post-grasp (the decision lives in object_position_for_tick).

Reused unchanged: go_to / clamp_delta / slow_blend from bridge.py, and all of
conversion (same physical arm, same calibration -> n_to_rad/rad_to_n hold).
"""

import time

from soarmrl import conversion, kinematics
from soarmrl.bridge import clamp_delta, go_to, slow_blend  # arm is the same; reuse

# --- policy I/O spec (env.yaml, retrain c079143: action_l2 + scale 0.25) ---
OBS_DIM = 28
ACTION_DIM = 6
# DELIBERATELY 2x the trained env.yaml scale (0.25). This is NOT a sync bug: the real
# arm can't reach the policy's wrist_flex home (sim 1.57, hardware tops out ~0.95, so
# it starts ~0.6 rad too high/back), so a trained-magnitude descent undershoots the
# cube. Hardware-confirmed 2026-07-26: at 0.25 it falls short, at 0.5 it reaches and
# grasps. The 2x compensates for the home offset. Drop to 0.25 ONLY after the home
# mismatch is fixed (recalibrate wrist_flex, or lower the sim home + retrain).
ACTION_SCALE = 0.5  # arm target_rad = LIFT_DEFAULT_POSE_RAD + ACTION_SCALE * output

# The lift policy's home pose (env.yaml init_state joint_pos), sim joint order.
# NOT conversion.DEFAULT_POSE_RAD (that is the reach home) — see bridge.py.
LIFT_DEFAULT_POSE_RAD = [0.0, 0.0, 0.0, 1.57, 0.0, 0.0]

# Binary gripper joint targets, rad. Open is wider than the sim's 0.5 rad so the
# real jaws actually clear the cube (0.5 rad ~= only 21% of the follower's gripper
# travel). Tune GRIPPER_OPEN_RAD by eye; it only weakly affects the obs (gripper
# pos_rel), which the policy barely uses.
GRIPPER_OPEN_RAD = 1.0
GRIPPER_CLOSE_RAD = 0.0  # firm close used through approach + grasp detection; matches
                         # sim's close_command, so the obs and the load-based detector
                         # stay valid. DON'T raise this — it corrupts both.
# Gentle hold applied ONLY after the grasp has latched. The cube blocks the jaws at
# ~0.29 rad, so once held the encoder reads ~0.29 no matter what we command -> the
# obs is unchanged, and the grasp is already detected. Backing the command off from
# 0.0 to here keeps the jaws gripping (command still < the ~0.29 block point) but at
# a fraction of the push force -> far less sustained current during the lift, which
# is what browned out the rail / tripped the servo overload. Raise toward 0.0 if the
# cube slips under the lift's accelerations; lower toward 0.29 if it still faults.
GRIPPER_HOLD_RAD = 0.30
# Deadband on the raw gripper output: |action[5]| must exceed this to flip the
# binary jaw state, so a near-zero policy output can't flap open/close every tick.
GRIPPER_DEADBAND = 0.3

# Latch "grasped" after the policy commands the gripper closed this many ticks in
# a row -> object_position switches from the static spot to the FK end-effector.
GRASP_TRIGGER_TICKS = 15  # ~0.5 s at 30 Hz
# Real grasp detection (grip_probe.py, 2026-07-26). Both gated on commanded-close.
GRIP_POS_STALL_RAD = 0.157   # gripper pos ABOVE this = jaws stalled open on an object
GRIP_LOAD_MIN = 276          # |Present_Load| ABOVE this = sustained holding effort

def build_grasp_obs(observation, object_position, target_pose, last_action,
                    prev_joint_pos, dt) -> list[float]:
    """Return the 28-dim obs, concatenation order per env.yaml:
        [0:6]   joint_pos_rel   = joint_pos - LIFT_DEFAULT_POSE_RAD  (mind the pose)
        [6:12]  joint_vel_rel   finite-diff; ZERO it in the loop (see grasp_hold)
        [12:15] object_position base frame (from object_position_for_tick)
        [15:22] target_pose     7 = xyz + quat wxyz, base frame; YOU choose it
        [22:28] last_action     previous raw 6-dim policy output, zeros on step 0

    `observation` is a LeRobot get_observation() dict ('<joint>.pos' keys), same
    no-I/O convention as bridge.builds_obs.
    """
    joint_pos = conversion.obs_to_joint_pos(observation)
   
    # joint_pos_rel over all 6 joints (gripper included); grasp_hold zeros vel_rel.
    pos_rel = [joint_pos[i] - LIFT_DEFAULT_POSE_RAD[i] for i in range(6)]
    vel_rel = [0.0] * 6

    return (
        pos_rel                     # [0:6]
        + vel_rel                   # [6:12]  (zeroed by grasp_hold)
        + list(object_position)     # [12:15]
        + list(target_pose)         # [15:22]
        + list(last_action)         # [22:28]  6-dim, used directly
    )


def binarize_gripper(gripper_action: float) -> float:
    """Map the policy's raw gripper scalar (action[5]) to a gripper joint target
    in rad by thresholding at 0 -> GRIPPER_OPEN_RAD / GRIPPER_CLOSE_RAD.

    VERIFY which side is open against the IsaacLab BinaryJointPositionAction
    source before hardware — a flipped threshold inverts the gripper.
    """
    return GRIPPER_OPEN_RAD if gripper_action > 0 else GRIPPER_CLOSE_RAD


def decide_gripper(gripper_action: float, currently_closed: bool, grasped: bool) -> bool:
    """Stateful (hysteresis) version of binarize_gripper's decision, returning the
    new closed/open state instead of a sign flip every tick. SAME polarity as
    binarize_gripper: positive action -> OPEN, negative -> CLOSE.
      - once `grasped` latches, stay closed (the policy can't reopen mid-lift);
      - otherwise the raw output must clear +/-GRIPPER_DEADBAND to flip the jaws;
      - inside the deadband, hold the current state (kills the near-zero flapping).
    """
    if grasped:
        return True
    if gripper_action > GRIPPER_DEADBAND:   # policy wants OPEN
        return False
    if gripper_action < -GRIPPER_DEADBAND:  # policy wants CLOSE
        return True
    return currently_closed


def scale_grasp_action(action: list[float]) -> tuple[list[float], float]:
    """Raw 6-dim policy output -> (5 arm joint targets in rad, gripper target rad):
        arm[i]  = LIFT_DEFAULT_POSE_RAD[i] + ACTION_SCALE * action[i]  for i < 5
        gripper = binarize_gripper(action[5])
    """
    scaled_arm = [LIFT_DEFAULT_POSE_RAD[i] + ACTION_SCALE * action[i] for i in range(5)]
    scaled_gripper = binarize_gripper(action[5])
    return scaled_arm, scaled_gripper


def object_position_for_tick(known_object_pos, joint_pos, grasped: bool) -> list[float]:
    """Source obs[12:15] — there is no perception on the real arm (see bridge.py):
        pre-grasp  (grasped=False): the static known cube spot (base frame)
        post-grasp (grasped=True):  the gripper position from FK — the cube rides
                                    with the gripper, so a static value would lie
                                    and the policy would never commit to the lift.

    joint_pos: the 6 sim-order joint angles (rad) from conversion.obs_to_joint_pos.
    """
    if grasped:
        return kinematics.ee_position(joint_pos)
    return list(known_object_pos)
    
def is_grasping(robot, gripper_pos_rad, commanded_close) -> bool:
    """True only when the gripper is commanded closed AND physically holding:
    jaws stalled open past GRIP_POS_STALL_RAD AND load past GRIP_LOAD_MIN.
    Empty close fails both (pos ~0, load relaxes). Reads Present_Load off the
    bus (robot.bus.read("Present_Load", "gripper")) — abs it; it's sign-magnitude."""
    if not commanded_close:
        return False
    load = abs(robot.bus.read("Present_Load", "gripper"))
    return gripper_pos_rad > GRIP_POS_STALL_RAD and load > GRIP_LOAD_MIN

def grasp_hold(robot, policy, target_pose, known_object_pos, n_ticks, slow,
               max_delta, hz: float = 30.0, verbose: bool = False,
               hold_ticks_after_grasp=None):
    """Drive the lift/grasp policy for n_ticks at hz. Each tick: read encoders ->
    build the 28-dim obs (velocity block [6:12] ZEROED, same limit cycle as
    bridge.reach_hold) -> run the policy -> scale_grasp_action -> slow-blend +
    delta-clamp the 5 arm targets -> send arm + binary gripper together -> log ->
    sleep to hold hz. Ramp to LIFT_DEFAULT_POSE_RAD with go_to first (confirm
    that more-upright pose is safe to enter from rest).

    Keep every reach safety invariant: limit-clamp, slow-mode, and Ctrl+C ->
    torque-off/hold. The gripper now actuates — nothing fragile in reach on the
    first runs.

    object_position is fed static until the policy has commanded the gripper
    closed for GRASP_TRIGGER_TICKS in a row; after that it tracks the FK EE, so
    the cube "rises" with the gripper and the policy commits to the lift.

    Returns (final 6-joint position rad, grasped bool) — the flag lets a caller
    hand off to a carry phase (e.g. the reach policy) only on a confirmed grasp.

    hold_ticks_after_grasp: None runs the full n_ticks (the policy drives its own
    lift toward target_pose). An int stops that many ticks AFTER the grasp latches
    — a short clean clearance lift, without the prolonged target-hunting — so a
    caller can take over the transport itself (see scripts/grasp/grasp_carry.py).
    """
    import torch  # local: keeps the pure obs/conversion helpers importable without torch

    dt = 1.0 / hz
    prev_joint_pos = conversion.obs_to_joint_pos(robot.get_observation())
    last_action = [0.0] * 6  # full 6-dim policy output; zeros on step 0
    prev_sent = prev_joint_pos[:5]  # start at current position, no blending
    grasped = False
    post_grasp_ticks = 0  # ticks elapsed since the grasp latched (for early handoff)
    gripper_closed = False  # hysteresis state for the binary jaw (starts open)
    closed_ticks = 0  # consecutive ticks the policy has commanded the gripper closed

    for tick in range(n_ticks):
        t_start = time.perf_counter()

        try:
            # 1) Read observation
            observation = robot.get_observation()
            joint_pos = conversion.obs_to_joint_pos(observation)

            # 2) Build obs (object_position: static pre-grasp, FK EE once grasped)
            obj_pos = object_position_for_tick(known_object_pos, joint_pos, grasped)
            obs = build_grasp_obs(observation, obj_pos, target_pose, last_action, prev_joint_pos, dt)

            # 3) ZERO velocity (critical for hardware limit cycle)
            obs[6:12] = [0.0] * 6

            # 4) Run policy
            with torch.no_grad():
                action = policy(torch.tensor([obs], dtype=torch.float32)).squeeze(0).tolist()

            # 5) Scale the arm. Decide the gripper with hysteresis + a grasp latch
            #    (decide_gripper) instead of a raw sign flip, so a near-zero output
            #    can't flap the jaws. Then latch "grasped" on a REAL grasp (physical
            #    detector, not just a sustained close command) for GRASP_TRIGGER_TICKS.
            scaled_arm_targets, _ = scale_grasp_action(action)
            gripper_closed = decide_gripper(action[5], gripper_closed, grasped)
            gripper_pos = joint_pos[conversion.JOINTS.index("gripper")]
            if is_grasping(robot, gripper_pos, gripper_closed):
                closed_ticks += 1
                if closed_ticks >= GRASP_TRIGGER_TICKS and not grasped:
                    grasped = True
                    print(f"\n>>> GRASP CONFIRMED (tick {tick}) — object now tracked from FK; "
                          f"driving to target {list(target_pose[:3])}\n")
            else:
                closed_ticks = 0

            # Firm close through approach + detection (obs/detector stay valid); once
            # grasped, back off to the gentle hold to cut sustained stall current.
            if not gripper_closed:
                scaled_gripper = GRIPPER_OPEN_RAD
            else:
                scaled_gripper = GRIPPER_HOLD_RAD if grasped else GRIPPER_CLOSE_RAD

            # 6) Slow-blend
            blended = slow_blend(prev_sent, scaled_arm_targets, slow)

            # 7) Delta-clamp
            clamped = clamp_delta(prev_sent, blended, max_delta)

            # 8) Send the slow-blended/clamped arm targets plus the binary gripper.
            # The gripper is binary, so it goes straight through (no blend/clamp).
            robot.send_action(conversion.targets_to_action(clamped, scaled_gripper))

            # 9) Log (read load only when verbose so we can watch the latch fire)
            if verbose:
                labeled = "  ".join(f"{j}={v:+.3f}" for j, v in zip(conversion.JOINTS, obs[0:6]))
                tag = "GRASP" if grasped else "     "
                load = abs(robot.bus.read("Present_Load", "gripper"))
                print(f"  tick {tick:4d} {tag} grip={gripper_pos:+.3f} load={load:4.0f} "
                      f"obj_z={obj_pos[2]:+.3f} pos_rel: {labeled}")

            # 10) Sleep
            time.sleep(max(0.0, dt - (time.perf_counter() - t_start)))

            # Update state for next loop
            prev_sent = clamped
            prev_joint_pos = joint_pos
            last_action = action

            # Early handoff: once grasped, stop after a short clearance lift instead
            # of running the policy's full (janky) drive to target_pose.
            if grasped and hold_ticks_after_grasp is not None:
                post_grasp_ticks += 1
                if post_grasp_ticks >= hold_ticks_after_grasp:
                    break

        except RuntimeError as e:
            # Feetech bus faults (input-voltage / overload / comms) surface here as
            # RuntimeError. Stop cleanly and hold, rather than crash the run with a
            # confusing double-traceback. The caller's finally still ramps back.
            print(f"\n!!! SERVO FAULT at tick {tick} — stopping, holding position.\n"
                  f"    {e}\n"
                  f"    Likely a brownout under load (Feetech 'Input voltage error') or a\n"
                  f"    stalled servo ('Overload error'). Power-cycle the arm to clear the\n"
                  f"    latched fault before the next run.")
            break

    return prev_joint_pos, grasped
