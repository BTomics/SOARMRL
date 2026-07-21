# SO-ARM101-PickPlace-v0 — Build Contract (first increments)

Concrete diff against the upstream lift env, written after reading the actual
reward code. This is the *what and why*; the reward implementations and tuning
are yours to write. Companion to [lift_task.md](lift_task.md) (the direction);
this doc is the buildable spec.

Base files you are diffing against (in the fork):
- `tasks/lift/lift_env_cfg.py` — commands, rewards, terminations, events
- `tasks/lift/mdp/rewards.py` — the reward functions
- `tasks/lift/joint_pos_env_cfg.py` — the SO-101 robot/cube/gripper wiring

**Do not edit lift in place.** Copy `tasks/lift/` to `tasks/pickplace/`, register
`SO-ARM101-PickPlace-v0` in its `__init__.py` (mirror `train.py:39-57`), and diff
there. Lift stays as your trained baseline + comparison.

---

## The blocker you must design around (found in the code)

`object_goal_distance` (`rewards.py:53-72`) gates the goal-tracking reward on
**the cube being in the air**:

```python
return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - tanh(distance/std))
```

Lift's whole reward structure assumes the goal is *above* `minimal_height`
(0.025). Your place target sits **on the table** (z ≈ cube resting height). With
the upstream gate, the moment the cube descends to the target the reward goes to
**zero** — the policy is actively punished for finishing the place. This is the
core authorship problem, and there's no upstream answer key for it.

**The fix is a reward you design**, not a config tweak. Options to weigh (pick and
defend one — this is the writeup):
- Split the task into phases: keep the `>minimal_height` gate for the *transport*
  term (carry the cube over the target while lifted), then a *separate* descent/
  place term that activates *below* a height and rewards closing the XY gap to the
  target as z decreases.
- Replace the binary height gate with a smooth function of grasp state (fingers
  closed on cube) so tracking stays alive through the descent.
- Gate on "cube is grasped" (contact) instead of "cube is high", so goal-tracking
  survives all the way to the table.

Whatever you choose, the invariant is: **the reward must not vanish when the cube
is correctly placed.** Watch the play render for the symptom — arm carries the
cube over the spot then yanks it back up to farm the airborne term.

---

## Increment 0 — place target on the table, retrain, observe (cheap)

Smallest change that turns lift into "move to a spot on the table." Purpose: see
exactly how the height-gate blocker manifests before you build the fix.

- **Command** (`lift_env_cfg.py:97-104`): lower `pos_z` from `(0.2, 0.35)` to the
  table-surface height. Verify the number in sim — it's the cube's resting z
  (cube init is `0.015`, `joint_pos_env_cfg.py:132`), not 0. Keep `pos_x/pos_y`
  for now; align them to your locked spawn range in Increment 2.
- **Nothing else yet.** Train a short run, `play` it. Expected: the arm grasps and
  hovers the cube over the spot but won't set it down (the vanishing-reward
  blocker). That failing render is the baseline you're fixing — capture it.

## Increment 1 — release + at-rest (the authorship core)

Add to `tasks/pickplace/mdp/rewards.py` (new functions; keep lift's originals for
the transport phase). Signatures to fill in — implementations are yours:

- `object_at_target_on_table(env, xy_std, z_tol, command_name, ...)` — reward the
  cube being at the commanded XY **and** at table height. This is the place term
  that replaces the airborne goal-tracking. Solves the blocker above.
- `object_released(env, command_name, vel_thresh, ...)` — reward the gripper being
  **open** while the cube is at target **and** nearly still (low object linear
  velocity). Gate release on low velocity so the policy can't farm it by dropping
  from height.
- `object_at_rest(env, vel_thresh, ...)` — reward low cube linear+angular velocity
  at the target. Distinguishes "placed" from "still being held/moved".

Reward weights: the place/release terms must dominate the surviving reach term, or
you get the hover trap (`lift_task.md:47`). Start place ≫ reach, iterate short runs.

New traps this introduces (no in-repo answer — watch renders, `lift_task.md:52-62`):
- **early drop**: releasing from height → gated by `vel_thresh` on release.
- **slam**: placing violently → add an impact-speed penalty term if it appears.

## Increment 2 — success termination + withdraw + spawn range

- **Termination** (`lift_env_cfg.py:187-194`): add a `success` `DoneTerm`. Success =
  cube within XY tol of target **and** at table z **and** at rest **and** gripper
  open **and** EE withdrawn a minimum distance from the cube. Write
  `place_success(env, ...)` in `tasks/pickplace/mdp/terminations.py`. Keep the
  existing `object_dropping` term.
- **Withdraw**: add a small reward for the EE retreating post-release, or fold
  "EE clear" into the success condition — otherwise the gripper camps on the placed
  cube (`lift_task.md:59`).
- **Lock the spawn + target ranges NOW** (the decision from planning). `reset_object_position`
  (`lift_env_cfg.py:145-153`) is the cube spawn; `object_pose` ranges are the target.
  Both must cover the whole region your ultrasonic arc can report (README rungs
  ⑥/⑦), or the sensor stages force a retrain. Set the final range before the long run.

---

## Obs / bridge coupling (decide before the long run)

Lift's obs is 28-dim: `joint_pos(6)+joint_vel(6)+object_pos(3)+target_pose(7)+
last_action(6)` (`lift_env_cfg.py:125-129`). The deployment bridge is being built
to **this** spec.

If your release logic needs the policy to know whether it has let go, add a
**gripper open/close state** obs term (28→29). This is the one obs change flagged
in `lift_task.md:38` — and it's a **one-dim bridge change**, so decide it here and
tell the bridge, don't discover it after deploy. Recommendation: try Increment 1
*without* the extra obs first (last_action already carries the gripper command);
only add the state term if the policy can't tell placed-and-released from held.

## Definition of done (sim side)

- `uv run train --task SO-ARM101-PickPlace-v0 --headless` runs in the fork.
- `play`: cube set down within tol of the commanded spot, at rest, gripper open and
  clear — majority of resets.
- A note per failed reward iteration + the term-by-term diff vs upstream lift — the
  narrative is the authorship writeup.

## Carryovers from reach/lift deployment (don't rediscover the hard way)

- **zero-vel**: `joint_vel_rel` (obs[6:12]) is the same finite-difference
  destabilizer that caused the reach limit cycle. Carry `--zero-vel` into the
  bridge for this policy too.
- **gripper is binary in sim** (`BinaryJointPositionActionCfg`,
  `joint_pos_env_cfg.py:120-125`): open=0.5, close=0.0. The bridge thresholds the
  policy's gripper scalar; it is not a continuous target.
- **domain randomization** beyond reach's: cube mass/size/friction, gripper
  friction — contact transfer is the hard part (`lift_task.md:83`). Add before the
  transfer attempt, not after.
