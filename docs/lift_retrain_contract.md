# Lift Policy — Retrain for Deployment Smoothness (Contract)

Concrete diff against the upstream lift env, written after reading the actual
reward/action code, to fix the one thing that blocks real-arm deployment: the
policy **saturates from the home pose**, so a rate-limited real arm can't track
it and the closed loop limit-cycles. This is the *what and why*; the reward/
action tuning and the training runs are yours to write. Companion to
[pickplace_contract.md](pickplace_contract.md) (the authorship task) — this doc
is only about making the *existing* lift policy deployable.

Base files you are diffing against (in the fork):
- `tasks/lift/lift_env_cfg.py` — rewards, curriculum, actions cfg (abstract)
- `tasks/lift/joint_pos_env_cfg.py` — the SO-101 action scale / robot / cube wiring
- `isaac_so_arm101/robots/…` — `SO_ARM101_CFG.init_state` (the home pose)

---

## The blocker (confirmed on hardware + localized in code)

Deploying `lift_policy.pt` on the real SO-101 fails identically at every throttle
setting (2026-07-26): tight (`slow=0.10, max_delta=0.02`) freezes the arm in a
small limit cycle; loose makes it thrash wildly and spam the gripper. It never
descends to the cube. This is **not** a bridge bug — the bridge, the object
frame, the calibration, and the grasp detector are all verified. It is a
*policy* property.

**Root cause, from the config:** the SO-101 action is
`JointPositionActionCfg(scale=0.5, use_default_offset=True)`
(`joint_pos_env_cfg.py:114`) driven by an **unbounded** Gaussian policy, and the
reward (`lift_env_cfg.py:157-207`) penalizes only:

- `action_rate_l2` — weight `-1e-4`, curriculum → `-1e-1` at 10 000 steps
- `joint_vel_l2` — weight `-1e-4`, curriculum → `-1e-1` at 10 000 steps

There is **no penalty on action magnitude.** So once the policy parks at a
saturated rail (observed raw outputs ~4–8), `action_rate_l2` is ≈0 (it isn't
*changing*), so the saturation is essentially free — and it pays off by reaching
the far cube fast. In sim that's fine: the position target is applied faithfully
each step, so `joint_pos` catches up and `last_action` stays consistent with it.
On the real arm, a safety-throttled position servo moves only a sliver per tick
while we feed the **full raw** `last_action` back into the obs → `joint_pos` and
`last_action` diverge → out-of-distribution → limit cycle.

`action_rate` alone cannot fix this: the problem is action **magnitude**, not
its rate of change.

Offline proof (last session): fed a correct home obs, the policy itself cleanly
commands a descent and responds to `object_position` — the policy "wants" the
right thing; the saturation + throttle interaction is what breaks it. A 2-loop
offline sim showed FULL-track descends, THROTTLED stays put (= the hardware
trace).

---

## The fix (ranked; yours to tune + train)

### 1. Bound the action magnitude — the missing lever
Either of:
- add `clip=(-1.0, 1.0)` on the `JointPositionActionCfg`
  (`joint_pos_env_cfg.py:114`) — **check your IsaacLab version's `JointActionCfg`
  exposes a `clip` field** first; or
- add an `action_l2` magnitude penalty to `RewardsCfg`, weight ~`-1e-2`, with a
  curriculum bump like the existing two. **Confirm `mdp.action_l2` is exported in
  your lift mdp**; if not, it is a ~3-line custom term (mean of squared raw
  actions).

Bounding to ±1 caps per-step joint motion at ±`scale` rad — precisely the
quantity the real arm must track without a throttle. This is the single most
important change.

### 2. Then reduce the action scale
`scale: 0.5 → ~0.25` (`joint_pos_env_cfg.py:116`) so a full ±1 output is a gentle
±0.25 rad/step. **Only meaningful together with step 1** — an unbounded policy
just outputs 2× larger raw values to compensate, so scale-alone does nothing.

> ⚠️ **Sync:** if you change `scale`, re-pull `env.yaml` and set
> `grasp_bridge.ACTION_SCALE` to the new value. The deployment constant must
> equal the trained scale or the arm moves the wrong amount.

### 3. Optional, last: lower / closer reset pose
Home `[0,0,0,1.57,0,0]` puts the EE ~17 cm above and behind a cube at z=0.015, so
the policy dives hard from step 0. A reset posture already partway down cuts the
initial saturated burst. Lives in `SO_ARM101_CFG.init_state` (robots/), not the
task cfg. Do this **only if 1+2 aren't enough**, because:

> ⚠️ **Sync:** changing the home pose means updating `LIFT_DEFAULT_POSE_RAD` in
> `grasp_bridge.py` to match (it's subtracted in `joint_pos_rel` and added back in
> the action offset — `use_default_offset=true`).

### 4. Keep / strengthen `action_rate`
Already curriculum'd to `-1e-1`. You can push it further, but it's a secondary
lever — magnitude (step 1) is the fix.

---

## Diagnostic to pull first (cheap, informs the weights)

From your last training run's logs: did it actually pass the 10 000-step
curriculum bump to `action_rate = -1e-1`, and if so, was `-1e-1` simply swamped by
the task rewards (`lifting_object` 15, `object_goal_tracking` 16, fine-grained 5)?
If the action penalties are ~2 orders of magnitude below the task reward at
convergence, that quantifies how hard step 1 needs to push.

---

## Verification (the test rig already exists)

1. **Offline gate (pre-hardware):** feed the retrained policy a home obs; it
   should command a *modest* descent (small, bounded raw action), not a saturated
   4–8. Reuse last session's offline probe.
2. **Hardware, throttle OFF:** `grasp_live.py --slow 1.0 --max-delta 0.5`.
   - **Success:** smooth descent where `joint_pos` tracks the commanded target and
     `last_action` stays small/bounded — no limit cycle.
   - Then the `is_grasping` detector + FK object-tracking (built 2026-07-26) take
     over for the grasp and lift automatically.
3. Place the 3 cm cube at the in-distribution spot (x≈0.2, y=0, z=0.015, matching
   sim reset).

**Do not add a throttle back to paper over residual roughness** — the whole point
is a policy the arm can track *without* one. If it still needs heavy throttling,
step 1/2 didn't go far enough.

---

## Sync discipline (non-negotiable)

`policies/lift_params/env.yaml` stays ground truth. After **every** retrain:
re-pull `env.yaml` and re-verify the bridge constants against it —
`ACTION_SCALE`, `LIFT_DEFAULT_POSE_RAD`, the 28-dim obs layout, and the binary
gripper open/close rads. A silent mismatch here reproduces the exact
symptoms of a control bug and wastes a hardware session.
