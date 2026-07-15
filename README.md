# SOARMRL — SO-ARM101 + Isaac Lab RL

Build a [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) follower arm, train a reach policy in Isaac Lab on a remote GPU, and deploy it to the real arm through a self-written sim-to-real bridge (upstream sim-to-real is still WIP — the bridge is the point of this project).

## Two machines

| Machine | Role | Setup guide |
|---|---|---|
| **Laptop (Windows)** | Hardware control: servo ID setup, calibration, deployment bridge | [docs/laptop_setup.md](docs/laptop_setup.md) |
| **A10 VM (SURF Research Cloud, Ubuntu)** | Training: Isaac Sim 5.1 + Isaac Lab 2.3 + [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) | [docs/a10_setup.md](docs/a10_setup.md) |

Training never runs on the laptop; hardware never connects to the VM. The handoff between them is an exported policy file (checked into `policies/`).

## Repo layout

```
docs/           setup guides for both machines
policies/       exported policy checkpoints (.pt/.onnx) from the A10
src/soarmrl/    deployment bridge + sim-vs-real trajectory logging (Weeks 5–6)
```

## Roadmap (summer 2026)

1. **Week 1** — order parts, print, set up both software stacks *(done — including the trained reach policy, ahead of schedule)*
2. **Weeks 2–3** — RL: train upstream's lift task as a **baseline** (it's pick-and-hold — the commanded target never touches the table and the gripper never releases), then **my own RL task**: `SO-ARM101-PickPlace-v0` — pick the cube up *and set it down* at a commanded spot (release + at-rest reward design is mine — see [docs/lift_task.md](docs/lift_task.md)); bridge milestone M1 done (offline policy inference verified)
3. **Week 4** — set servo IDs → assemble → calibrate
4. **Weeks 5–6** — deployment bridge on the real arm: reach policy → scripted pick-up of a predetermined object (`src/soarmrl/grasp.py`) → attempt the learned pick-and-place policy; log real-vs-sim trajectories throughout
5. **Weeks 6–7 (add-on — first thing cut if the lift task slips)** — ultrasonic object localization: HC-SR04 + Arduino measures the cube position and feeds it into the policy obs (sense **before** the arm enters the workspace, then act — the moving arm would corrupt the echo). Two stages:
   - **Fixed sensor**, object on its axis: distance alone determines position (1D). Near-guaranteed to work.
   - **Servo-swept sensor (stretch)**: polar localization over an arc. The ~15° beam cone smears angle by ~5 cm at 20 cm range — comparable to gripper tolerance, so this rung may fail; measuring that resolution limit is a result.
6. **Week 7** — demo video + writeup

**Fallback ladder** (every rung is a valid demo): ① reach deploys → ② scripted pick-up → ③ upstream lift baseline trained in sim → ④ own pick-and-place policy in sim → ⑤ pick-and-place transfers to hardware → ⑥ pick-up at ultrasonic-measured distance → ⑦ servo-swept detection arc. Contact-rich sim-to-real may kill rung ⑤ — quantifying *why* is a result, not a failure. Same for ⑦ and beam width.

## Upstream

- Hardware BOM + STLs: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- Assembly + calibration: [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
- Isaac Lab envs: [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) (BSD-3-Clause), forked to [BTomics/isaac_so_arm101](https://github.com/BTomics/isaac_so_arm101) — where `SO-ARM101-PickPlace-v0` lives
- Driver / API: [huggingface/lerobot](https://github.com/huggingface/lerobot)
