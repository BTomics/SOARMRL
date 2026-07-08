# SO-ARM101-Lift-v0 — Task Design Contract

The RL I author myself: the arm *learns* to pick up a cube in sim. This doc is
direction, not a recipe — the design decisions are mine to make and defend.

## Starting materials (read in this order)

1. **Upstream reach task** (`isaac_so_arm101` fork) — how the SO-101 is wired
   into Isaac Lab: robot cfg, scene, obs/action/reward managers, task
   registration. My task reuses all of this plumbing.
2. **Isaac Lab's Franka lift reference** — the manager-based `Lift-Cube` task
   (`isaaclab_tasks` … `manipulation/lift`). This is the porting source: it
   already solved obs/reward design for lifting; my job is adapting it to a
   6-DOF arm with a two-finger gripper.

## What changes vs. reach

| Piece | Reach (upstream) | Lift (mine) |
|---|---|---|
| Scene | arm + target marker | arm + **rigid cube on a table** |
| Command | random EE pose | cube spawn pose (+ target lift height) |
| Obs | joint pos/vel, pose cmd, last action | + **object pose (relative to EE is usually better than absolute)** |
| Reward | EE-to-target distance | **staged**: reach cube → grasp → lift |
| Termination | timeout | + cube lifted (success), cube off table (fail) |

## Reward design — where the real work is

Staged shaping, roughly: `w1 * reach_term + w2 * grasp/contact_term +
w3 * lift_term`. Known traps, so I recognize them in the curves:

- **The flick**: max lift reward for batting the cube upward without
  grasping. Symptom: reward up, success rate ~0. Fix: gate the lift term on
  contact/grasp, or reward height only while fingers are closed on the cube.
- **The hover**: arm parks above the cube farming the reach term forever.
  Fix: make later stages worth much more, or decay early terms.
- **Gripper camping**: closing the gripper immediately and dragging it
  closed along the table. Watch the play render, not just the curves.

Iterate: train short runs (don't wait for full convergence to judge a
reward), watch `play` renders, read which reward term dominates in the logs.

## Sim-to-real notes (decided context)

- The real-world object pose is **predetermined** (measured spot on the
  table) — fed to the policy as a constant obs. No perception pipeline.
- This makes obs consistency easy for the object, but contact-rich transfer
  is still the hard part (grasp physics, gripper friction). Expected outcome
  honestly ranges from "transfers" to "sim-only + gap analysis" — both are
  planned-for results (fallback ladder in the README).
- DR to consider beyond reach's: cube mass/size/friction, gripper friction.

## Definition of done (sim side)

- `uv run train --task SO-ARM101-Lift-v0 --headless` works in my fork
- success rate in play: cube held above threshold height, majority of resets
- training curves + a short note on every reward iteration that failed and
  why — that narrative is writeup gold
