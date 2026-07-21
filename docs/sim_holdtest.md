# Sim hold-test recipe (M2 limit-cycle diagnosis)

Goal: run the reach policy through the **exact trained obs pipeline** in Isaac,
started at the hand-posed config with the target pinned, and watch whether the
joints settle. Splits "throttle is the cause" from "target is bad" for the
hardware limit cycle seen in `scripts/reach/m2_live.py`.

The isaac fork lives on the VM. Three values below are repo-specific — pull them
with the step-1 greps; everything else is verbatim.

## 0. Get on the VM and into the env
1. x2go into the VM.
2. `cd` to the fork (next to SOARMRL, e.g. `cd ~/.../isaac_so_arm101`).
3. Activate the Isaac Lab conda env used to produce `policy.pt`
   (e.g. `conda activate isaaclab`).

## 1. Get the three repo-specific names
```bash
# a) play script + the SO-100 Reach task id you normally play with
ls scripts/rsl_rl/ 2>/dev/null || find . -name "play.py"
grep -rn "gym.register" source/ | grep -i reach

# b) command term name + its ranges block (CommandsCfg file you've edited before)
grep -rn "Ranges\|UniformPoseCommand\|CommandsCfg" source/ | grep -i reach
```

## 2. Copy the play script (don't clobber your eval one)
```bash
cp scripts/rsl_rl/play.py scripts/rsl_rl/play_holdtest.py
```

## 3. Three edits to play_holdtest.py

### (a) Pin the command — in the reach env CommandsCfg (from 1b)
Collapse ranges to the single target and stop resampling:
```python
pos_x=(0.0, 0.0), pos_y=(-0.221, -0.221), pos_z=(0.212, 0.212),
roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
# also set the command term's resampling_time_range longer than the run
```
That point is inside the trained box (x in [-0.1,0.1], y in [-0.25,-0.1],
z in [0.1,0.3]) — stays in-distribution.

### (b) Start the arm at the hand-posed config — right after env.reset()
Set by joint NAME so DOF ordering can't bite:
```python
robot = env.unwrapped.scene["robot"]
print("JOINT NAMES:", robot.joint_names)   # run once, confirm the 6 names
names = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]  # replace with real names
idx = [robot.joint_names.index(n) for n in names]
HAND_POSED = [0.3279, 1.4139, -1.8310, 0.7829, -0.2285, 0.0292]
q = robot.data.default_joint_pos.clone()
q[:, idx] = torch.tensor(HAND_POSED, device=env.unwrapped.device)
robot.write_joint_state_to_sim(q, torch.zeros_like(q))
```

### (c) Log joint positions — inside the rollout loop
```python
jp = robot.data.joint_pos[0, idx].tolist()
print("  ".join(f"{n}={v:+.3f}" for n, v in zip(names, jp)))
```
Do NOT add scaling/clamping — env.step(action) already applies
`default + 0.5*action`, exactly the pipeline the hardware throttle fights.

## 4. Run
```bash
python scripts/rsl_rl/play_holdtest.py --task <YOUR_REACH_TASK_ID> --num_envs 1 \
  --checkpoint /path/to/policies/policy.pt --headless   # drop --headless to watch
```
Let it run ~300 steps (~10 s). Guard against a mid-run auto-reset (longer episode
length, or stop after 300).

## 5. Read the result
- **shoulder_lift / elbow_flex / wrist_flex settle** to steady values ->
  policy has a stable fixed point at this target applied directly -> hardware
  limit cycle is the THROTTLE (SLOW/clamp lag). Fix = the safety filter.
- **They oscillate in sim too** -> TARGET is bad (SO-100/SO-101 FK mismatch).
  Fix POSE_COMMAND / HAND_POSED_TARGET_RAD.

Fill from the VM: task id (1a), CommandsCfg ranges + term name (1b), exact joint
names (3b print). Everything else is verbatim.
