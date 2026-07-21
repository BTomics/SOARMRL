"""Deployment bridge: exported Isaac Lab reach policy -> real SO-101 follower.

This file is a CONTRACT, not an implementation. Structure it however you like;
what matters is that the invariants below hold. Delete these notes as the real
code replaces them.

================================================================================
THE CORE PROBLEM
================================================================================
The policy is a function: observation vector (24) -> 5 action values.
It was trained against ONE exact observation layout. Reproduce it bit-for-bit
from real encoder data, or the policy outputs garbage that LOOKS plausible.

Ground truth for everything below: policies/reach_v0_params/env.yaml
(and agent.yaml). Values below verified against that yaml 2026-07-19.

OBSERVATION VECTOR, 24-dim (concatenated in this order, env.yaml
`observations:`):
  1. joint_pos_rel   (6)  joint positions MINUS the default pose, radians,
                          sim joint order. Default pose (init_state):
                            shoulder_pan 0.0, shoulder_lift 1.57,
                            elbow_flex -1.57, wrist_flex 1.0,
                            wrist_roll -1.57, gripper 0.0
  2. joint_vel_rel   (6)  joint velocities, rad/s (default vel is 0)
  3. pose_command    (7)  target: xyz + quaternion wxyz of desired EE pose,
                          robot base frame — YOU choose this. Trained ranges:
                            x [-0.1, 0.1], y [-0.25, -0.1], z [0.1, 0.3],
                            orientation always identity (1, 0, 0, 0).
                          Outside that box = out-of-distribution; test
                          targets go inside it.
  4. last_action     (5)  previous raw policy output (before scaling!),
                          zeros on the first step. 5, NOT 6 — see ACTION.

  Gotchas that will bite:
  - sim joint ORDER and NAMES: check the isaac_so_arm101 source; LeRobot's
    motor order is not guaranteed to match — build an explicit index map,
    never assume.
  - UNITS: sim is radians; LeRobot get_observation() returns degrees or
    normalized values depending on config — check, convert, write a test.
  - training added uniform obs noise (±0.01) on joint pos/vel — that's a
    train-time robustness trick, do NOT add noise at deployment.

ACTION (policy output, 5 values — the GRIPPER IS NOT CONTROLLED):
  env.yaml `actions:` has arm_action over [shoulder_pan, shoulder_lift,
  elbow_flex, wrist_flex, wrist_roll] and gripper_action: null.
    target_joint_pos = default_pose + 0.5 * policy_output   (radians)
  The bridge holds the gripper at a fixed position itself, every tick.

================================================================================
CONVERSION CONSTANTS (derived 2026-07-19 from calibration file + held-pose
recon + FK direction analysis; raw data in the session log on PRO-101)
================================================================================
  rad = default_rad + direction * scale * (n - n_at_default)

  joint          dir   scale(rad/unit)  n_at_default  default_rad
  shoulder_pan   -1    0.01953          10.99          0.0
  shoulder_lift  +1    0.01870           5.98          1.57
  elbow_flex     +1    0.01707           1.58         -1.57
  wrist_flex     +1    0.01875          ~89.6 (a)      1.0
  wrist_roll     +1?   0.03141          2.24 -/+ 50    -1.57  (b)
  gripper        held fixed at n ~= 0.3

  (a) derived by walking back from the +100 end stop; verify with one held
      reading (~90 expected, gripper supported ~10 deg off the stop).
  (b) direction PROVISIONAL and n_at_default pending the quarter-turn
      reading. A wrong roll sign shows up in shadow mode as the policy
      demanding a large constant roll correction — flip it there if seen.

  Scales come from the LeRobot calibration tick spans (2*pi/4096 rad/tick);
  n is normalized units as returned by get_observation(). wrist_roll's
  calibration spans exactly one full revolution (no stops).

  Verification poses (independent checks of the map):
  - held default pose -> (0.0, 1.57, -1.57, 1.0, -1.57, ~0)
  - elbow folded to its stop (forearm hugging upper arm) -> elbow ~= 0.0 rad
  - M2 hold-still EE target, base frame, from FK: (0.0, -0.221, 0.212)

  STARTUP/SHUTDOWN (policy never starts from the rest pose — training reset
  band was 0.5-1.5x default, folded is out of distribution):
    rest (folded, torque off) -> enable torque -> scripted ramp to default
    pose over 3-5 s -> policy loop -> scripted ramp back to rest -> torque
    off.

================================================================================
SAFETY INVARIANTS (non-negotiable, in every code path that moves the arm)
================================================================================
  - clamp every target to joint limits (env.yaml has them; tighten further
    for the first runs)
  - clamp the per-step CHANGE in target (max delta per tick) so a policy
    glitch cannot command a violent jump
  - slow mode first: scale all motion down (e.g. interpolate targets at 25%)
    until the arm has proven itself
  - e-stop: Ctrl+C must ALWAYS work -> wrap the loop, and on exit command
    torque-off or hold, never leave the last target racing
  - never run the loop the first time with the gripper near anything fragile

================================================================================
CONTROL LOOP SHAPE (30 Hz — exactly: sim dt 1/60 x decimation 2)
================================================================================
  read encoders -> build obs (exact layout above) -> policy(obs) ->
  scale + clamp -> send targets (5 arm + held gripper) -> log tick (see
  trajectory_log.py) -> sleep to hold the rate (measure real latency; it is
  an obs-freshness issue AND a data point for the writeup)

================================================================================
MILESTONES (build/test in this order — matches PRO-101)
================================================================================
  M1  offline sanity, NO hardware:
      torch.jit.load("policies/policy.pt"), feed a hand-built 24-dim obs for
      "arm at default pose, target inside the trained command box" -> expect
      5 finite, small-ish values. First assert the network's input layer is
      actually 24-wide — that check alone validates the obs-layout math.
      Feed garbage -> confirm you can tell the difference.
  M2  the loop against hardware, slow mode, arm clamped to the table,
      target = current pose (policy should barely move)
  M3  real reach: fixed target poses, compare against sim rollouts
  M4  scripted pick-up on top (see grasp.py)

  LeRobot side (see docs/laptop_setup.md): SO101Follower.get_observation()
  to read, .send_action() to command. Verify dict keys/units empirically
  with the arm powered but held, before trusting them in the loop.

================================================================================
GRASP/LIFT POLICY — BRIDGE DELTAS (policies/lift_policy.pt; verified against
policies/lift_params/env.yaml 2026-07-21)
================================================================================
Same physical arm, so CALIB and n_to_rad/rad_to_n/obs_to_joint_pos are REUSED
unchanged. But it is a different policy: obs 24->28, action 5->6, and — the trap
— a DIFFERENT home pose. Full sim-side task contract: docs/pickplace_contract.md.

  DEFAULT POSE DIFFERS — do NOT reuse the reach DEFAULT_POSE_RAD:
    reach: [0.0, 1.57, -1.57, 1.0, -1.57, 0.0]
    lift : [0.0, 0.0,  0.0,   1.57, 0.0,  0.0]   (env.yaml init_state joint_pos)
  use_default_offset=true, so this pose is BOTH subtracted in joint_pos_rel AND
  added back in the action offset. Feed the reach home and every obs and every
  target is wrong. The lift bridge needs its own LIFT_DEFAULT_POSE_RAD, used in
  the obs builder, the action scaler, AND the startup ramp. Verify the more-
  upright lift home is safe to ramp into from rest before the first hardware run.

  OBS, 28-dim (env.yaml observations, concatenation order):
    [0:6]    joint_pos_rel          minus LIFT default pose
    [6:12]   joint_vel_rel          still zero it (same finite-diff limit cycle)
    [12:15]  object_position        cube xyz in robot BASE frame. No perception
                                    on real: pre-grasp = hardcoded known spot;
                                    post-grasp the cube moves with the gripper so
                                    a constant lies -> switch to EE-from-FK once
                                    grasped. DECIDE this before writing the loop.
    [15:22]  target_object_position goal pose (7); YOU choose it. Lift ranges
                                    x[-0.1,0.1] y[-0.3,-0.1] z[0.2,0.35] (in air).
    [22:28]  last_action            previous raw 6-dim output, zeros on step 0.

  ACTION, 6 (arm 5 + gripper): target[:5] = LIFT_DEFAULT_POSE_RAD[:5] + 0.5*a[:5].
    a[5] = gripper, BinaryJointPositionAction: threshold the scalar at 0 -> 0.5
    rad (open) / 0.0 rad (close). VERIFY which side is open against the IsaacLab
    BinaryJointPositionAction source (backwards = inverted gripper). Route the
    gripper target through conversion like any joint; targets_to_action must take
    it now instead of holding the gripper fixed at GRIPPER_HOLD_N.

  Config lineage: reach was SO-100, this lift policy is SO-101 (so_arm101.urdf,
  gripper_link) — matches the hardware. CALIB is hardware-derived, still valid.

  Build order (mirror reach M1->M3): offline sanity (assert input layer 28-wide)
  -> slow hardware loop with STATIC object_position -> add EE-from-FK post-grasp.
"""

import time

from soarmrl import conversion

ACTION_SCALE = 0.5  # target_rad = default_pose + ACTION_SCALE * policy_output (env.yaml)


def go_to(robot, target_rad: list[float], seconds: float, hz: float = 30.0) -> None:
    """Ramp the 5 arm joints from wherever they are to target_rad (sim
    order, radians) with linearly interpolated position commands. Blocks
    until done. Targets are limit-clamped by targets_to_action."""
    start = conversion.obs_to_joint_pos(robot.get_observation())[:5]
    steps = max(1, round(seconds * hz))
    for k in range(1, steps + 1):
        alpha = k / steps
        step_target = [s + alpha * (t - s) for s, t in zip(start, target_rad)]
        robot.send_action(conversion.targets_to_action(step_target))
        time.sleep(1.0 / hz)

def builds_obs(observation, pose_command, last_action, prev_joint_pos, dt) -> list[float]:
    """Return a 24 dim vector flattened in this order:
    [rel_pos, rel_vel, pose_command, last_action]

    observation is a LeRobot get_observation() dict ('<joint>.pos' keys),
    not a robot object -- no I/O in this module, same convention as
    conversion.obs_to_joint_pos.
    """
    obs = conversion.obs_to_joint_pos(observation)
    joint_pos_rel = [obs[i] - conversion.DEFAULT_POSE_RAD[i] for i in range(6)]
    joint_vel_rel = [(obs[i] - prev_joint_pos[i]) / dt for i in range(6)]
    
    return joint_pos_rel + joint_vel_rel + pose_command + last_action

def clamp_delta(prev_target: list[float], new_target: list[float], max_delta: float) -> list[float]:
    """Clamp the change in target to max_delta."""
    clamped_targets = []
    for prev, new in zip(prev_target, new_target):
        diff = new - prev
        if abs(diff) <= max_delta:
            clamped_targets.append(new)
        else:
            clamped_targets.append(prev + max_delta if diff > 0 else prev - max_delta)
    return clamped_targets


def scale_action(action: list[float]) -> list[float]:
    """Raw 5-dim policy output -> 5 arm joint targets (rad, sim order).
    target_rad = DEFAULT_POSE_RAD[:5] + ACTION_SCALE * action."""
    return [conversion.DEFAULT_POSE_RAD[i] + ACTION_SCALE * action[i] for i in range(len(action))]


def slow_blend(prev_sent: list[float], target: list[float], slow: float) -> list[float]:
    """Interpolate prev_sent toward target by `slow` (0-1); lower = gentler.
    sent = prev_sent + slow * (target - prev_sent)."""
    return [prev_sent[i] + slow * (target[i] - prev_sent[i]) for i in range(len(target))]


def reach_hold(robot, policy, pose_command, n_ticks, slow, max_delta,
               hz: float = 30.0, verbose: bool = False) -> list[float]:
    """Drive the reach policy toward pose_command for n_ticks at hz: each tick
    reads encoders, builds the obs, runs the policy, then scales -> slow-blends
    -> delta-clamps the target and sends it. Assumes the arm is already at a
    valid in-distribution start (ramp there with go_to first). The gripper is
    held fixed by targets_to_action. Returns the final 6-joint position (rad).

    The finite-difference velocity block of the obs is zeroed: on real hardware
    it is noisy/lagged and, with actuation latency, drives a limit cycle
    (confirmed 2026-07-20 -- sim uses true velocity and converges; hardware only
    settled with velocity zeroed).
    """
    import torch  # local: keeps the pure obs/conversion helpers importable without torch

    dt = 1.0 / hz
    prev_joint_pos = conversion.obs_to_joint_pos(robot.get_observation())
    last_action = [0.0] * 5
    prev_sent = prev_joint_pos[:5]

    for tick in range(n_ticks):
        t_start = time.perf_counter()

        observation = robot.get_observation()
        obs = builds_obs(observation, pose_command, last_action, prev_joint_pos, dt)
        obs[6:12] = [0.0] * 6  # zero finite-diff velocity (see docstring)

        with torch.no_grad():
            action = policy(torch.tensor([obs], dtype=torch.float32)).squeeze(0).tolist()

        clamped = clamp_delta(prev_sent, slow_blend(prev_sent, scale_action(action), slow), max_delta)
        robot.send_action(conversion.targets_to_action(clamped))

        prev_sent = clamped
        prev_joint_pos = conversion.obs_to_joint_pos(observation)
        last_action = action

        if verbose:
            labeled = "  ".join(f"{j}={v:+.3f}" for j, v in zip(conversion.JOINTS, obs[0:6]))
            print(f"  tick {tick:4d} pos_rel: {labeled}")

        time.sleep(max(0.0, dt - (time.perf_counter() - t_start)))

    return prev_joint_pos
    