# SO-ARM101-PickPlace-v0 — Task Design Contract

The RL I author myself. Re-planned 2026-07-15: upstream `isaac_so_arm101`
already ships a lift task (`tasks/lift`) — and it's really
*pick-and-move-to-pose*: a commanded object target pose with goal-tracking
rewards, but z ∈ [0.2, 0.35] (the target is always in the air) and the
gripper never releases. Copying that isn't my RL work, so my task moves one
step further: **place** — set the cube down at a commanded spot, open the
gripper, leave it at rest.

Prior-art check (2026-07-15): SO-ARM pick-and-place exists only as
imitation/VLA (NVIDIA's [Sim-to-Real SO-101 Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)
uses teleop demos + GR00T). Pure-RL pick-and-**place** for this arm in Isaac
Lab appears undone — that's the authorship claim. This doc is direction, not
a recipe — the design decisions are mine to make and defend.

## Part 1 — upstream lift as baseline (quick, not authorship)

Train upstream's lift task as-is on the A10 (fork:
[BTomics/isaac_so_arm101](https://github.com/BTomics/isaac_so_arm101)). Goals:

- a baseline checkpoint + proof the SO-101 learns grasping in this repo
- learn the plumbing my task reuses: gripper `BinaryJointPositionActionCfg`
  (why binary and not continuous?), the cube rigid object + the
  `reset_object_position` event, `UniformPoseCommandCfg` for the object target
- watch the play render — does it actually grasp, or is it one of the traps
  below?
- check the cube spawn range (x ±0.1, y ±0.2) covers the planned ultrasonic
  arc (README rungs ⑥/⑦); widen it **before** training PickPlace if not

## Part 2 — what PickPlace changes vs. upstream lift

| Piece | Upstream lift | PickPlace (mine) |
|---|---|---|
| Command | object target pose, z ∈ [0.2, 0.35] (in the air) | place target **on the table** (z = table height) |
| Reward | reach → lift → goal-tracking | + **release + at-rest** stage on top of the staged structure |
| Termination | timeout, cube off table | + success: cube at target, at rest, **gripper open and clear** |
| Obs | joint pos/vel, object pos, target cmd, last action | likely unchanged (maybe gripper open/close state) |

## Reward design — where the real work is

Staged shaping. Known traps from lift, so I recognize them in the curves:

- **The flick**: max lift reward for batting the cube upward without
  grasping. Symptom: reward up, success rate ~0. Fix: gate the lift term on
  contact/grasp, or reward height only while fingers are closed on the cube.
- **The hover**: arm parks above the cube farming the reach term forever.
  Fix: make later stages worth much more, or decay early terms.
- **Gripper camping**: closing the gripper immediately and dragging it
  closed along the table. Watch the play render, not just the curves.

New traps that *place* adds — no in-repo answer key for these:

- **The early drop**: releasing the cube from height above the target to
  farm the place reward. Fix: gate release reward on low object velocity /
  penalize impact speed.
- **The slam**: technically placed, violently. Reward gentle set-down, not
  just final position.
- **Post-release camping**: gripper parked on top of the placed cube.
  Success requires the EE to withdraw a minimum distance.
- **Re-grasp loops**: knock the cube, re-grasp, repeat — farms reach/grasp
  terms forever. Watch renders for it.

Iterate: train short runs (don't wait for full convergence to judge a
reward), watch `play` renders, read which reward term dominates in the logs.
Upstream's lift rewards are the comparison baseline once mine trains (or
fails honestly) — the term-by-term diff is writeup material.

## Sim-to-real notes (decided context)

- The real-world object pose is fed to the policy as a **constant obs per
  episode** — first a predetermined measured spot, later an ultrasonic
  reading (HC-SR04 + Arduino, sense-then-act; see README roadmap). Either
  way it's one fixed (x, y, z), never a live perception stream.
- **Decide the cube spawn randomization range before training**: it must
  cover the whole region the swept sensor can report (an arc on the table),
  not just one spot — otherwise the ultrasonic stages need a retrain.
- Obs consistency for the object is easy, but contact-rich transfer is
  still the hard part (grasp physics, gripper friction) — and place adds
  release dynamics on top. Expected outcome honestly ranges from
  "transfers" to "sim-only + gap analysis" — both are planned-for results
  (fallback ladder in the README).
- DR to consider beyond reach's: cube mass/size/friction, gripper friction.

## Definition of done (sim side)

- upstream lift baseline trained and its play render assessed (Part 1)
- `uv run train --task SO-ARM101-PickPlace-v0 --headless` works in my fork
- success rate in play: cube set down within tolerance of the commanded
  spot, at rest, gripper clear — majority of resets
- training curves + a short note on every reward iteration that failed and
  why, plus the diff against upstream's lift rewards — that narrative is
  writeup gold
