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
2. **Weeks 2–3** — understand the env, tune domain randomization; bridge milestone M1 (offline policy inference)
3. **Week 4** — set servo IDs → assemble → calibrate
4. **Weeks 5–6** — deployment bridge on the real arm: policy inference → joint-position targets via the LeRobot API (~30 Hz) → encoder states back as observations; log real-vs-sim trajectories; finale: **scripted pick-up of a predetermined object** (`src/soarmrl/grasp.py`)
5. **Week 7** — demo video + writeup

## Upstream

- Hardware BOM + STLs: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- Assembly + calibration: [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
- Isaac Lab envs: [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) (BSD-3-Clause)
- Driver / API: [huggingface/lerobot](https://github.com/huggingface/lerobot)
