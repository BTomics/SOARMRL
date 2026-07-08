"""Deployment bridge: exported Isaac Lab reach policy -> real SO-101 follower.

This file is a CONTRACT, not an implementation. Structure it however you like;
what matters is that the invariants below hold. Delete these notes as the real
code replaces them.

================================================================================
THE CORE PROBLEM
================================================================================
The policy is a function: observation vector -> 6 action values.
It was trained against ONE exact observation layout. Reproduce it bit-for-bit
from real encoder data, or the policy outputs garbage that LOOKS plausible.

Ground truth for everything below: policies/reach_v0_params/env.yaml
(and agent.yaml). Do not trust these comments over the yaml — verify.

OBSERVATION VECTOR (concatenated in this order, see env.yaml `observations:`):
  1. joint_pos_rel   (6)  joint positions MINUS the default/home pose,
                          in radians, in sim's joint order
  2. joint_vel_rel   (6)  joint velocities relative to default (default vel
                          is 0, so effectively just velocities), rad/s
  3. pose_command    (7)  the target: position xyz + quaternion wxyz of the
                          desired end-effector pose, in the robot base frame
                          — YOU choose this; it is the command input
  4. last_action     (6)  the previous raw policy output (before scaling!),
                          zeros on the first step

  Gotchas that will bite:
  - sim joint ORDER and NAMES: check env.yaml / the isaac_so_arm101 source;
    LeRobot's motor order is not guaranteed to match — build an explicit
    index map, never assume.
  - UNITS: sim is radians; LeRobot get_observation() returns degrees or
    normalized values depending on config — check, convert, write a test.
  - the DEFAULT POSE values are in env.yaml (scene -> articulation ->
    init_state); joint_pos_rel is relative to THAT, not to your calibrated
    zero.

ACTION (policy output, 6 values):
  JointPositionAction with a scale factor (env.yaml `actions:`):
    target_joint_pos = default_pose + scale * policy_output
  The result is in radians, sim joint order. Convert to LeRobot's expected
  units/order before sending.

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
CONTROL LOOP SHAPE (~30 Hz)
================================================================================
  read encoders -> build obs (exact layout above) -> policy(obs) ->
  scale + clamp -> send targets -> log tick (see trajectory_log.py) ->
  sleep to hold the rate (measure real latency; it is an obs-freshness issue
  AND a data point for the writeup)

================================================================================
MILESTONES (build/test in this order — matches PRO-101)
================================================================================
  M1  offline sanity, NO hardware:
      torch.jit.load("policies/policy.pt"), feed a hand-built obs for
      "arm at default pose, target 20 cm in front" -> expect 6 finite,
      small-ish values. Feed garbage -> confirm you can tell the difference.
  M2  the loop against hardware, slow mode, arm clamped to the table,
      target = current pose (policy should barely move)
  M3  real reach: fixed target poses, compare against sim rollouts
  M4  scripted pick-up on top (see grasp.py)

  LeRobot side (see docs/laptop_setup.md): SO101Follower.get_observation()
  to read, .send_action() to command. Verify dict keys/units empirically
  with the arm powered but held, before trusting them in the loop.
"""
