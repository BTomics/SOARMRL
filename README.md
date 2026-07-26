# SOARMRL — SO-ARM101 + Isaac Lab RL

Build a [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) follower arm, train policies in Isaac Lab on a remote GPU, and deploy them to the real arm through a self-written sim-to-real bridge (upstream sim-to-real is still WIP — the bridge is the point of this project).

## Two machines

| Machine | Role | Setup guide |
|---|---|---|
| **Laptop (Windows)** | Hardware control: servo ID setup, calibration, deployment bridge | [docs/laptop_setup.md](docs/laptop_setup.md) |
| **A10 VM (SURF Research Cloud, Ubuntu)** | Training: Isaac Sim 5.1 + Isaac Lab 2.3 + [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) | [docs/a10_setup.md](docs/a10_setup.md) |

Training never runs on the laptop; hardware never connects to the VM. The handoff between them is an exported policy file (checked into `policies/`).

### Deployed policies & their training source

Each checkpoint in `policies/` was trained in the [BTomics/isaac_so_arm101](https://github.com/BTomics/isaac_so_arm101) fork and exported here alongside its `env.yaml` (the ground-truth obs/action spec the bridge reproduces):

- `policies/policy.pt` — reach policy.
- `policies/lift_policy.pt` (+ `lift_params/env.yaml`) — lift/grasp policy, retrained for deployment smoothness with an `action_l2` action-magnitude penalty and a reduced action scale so a rate-limited real arm can track it. Training change: [`BTomics/isaac_so_arm101` @ `c5432ea`](https://github.com/BTomics/isaac_so_arm101/commit/c5432ea).

On hardware the lift policy grasps and lifts, and `scripts/grasp/grasp_carry.py` composes it with the reach policy — the lift policy grabs the cube, then reach carries it to a commanded pose.

## Repo layout

```
docs/           setup guides for both machines
policies/       exported policy checkpoints (.pt/.onnx) + their env.yaml
src/soarmrl/    deployment bridge, conversion/kinematics, grasp + carry logic
scripts/        runnable entry points (reach, grasp, carry, teach)
```

## Status

**Working on hardware:**
- **Reach policy** — deployed on the real arm; servos the end-effector to a commanded pose (~18 mm accuracy at target, decomposed into policy vs. hardware error).
- **Lift/grasp policy** — grasps and lifts a 3 cm cube, with grasp detection read from the gripper servo's load feedback (`scripts/grasp/grasp_live.py`).
- **Grasp → carry composition** (`scripts/grasp/grasp_carry.py`) — chains both policies in one motion: the lift policy grabs the cube, then the reach policy carries it to a commanded pose and releases.

**Also built:** a scripted teach-and-replay pickup (`src/soarmrl/grasp.py`) as a fallback, and sim-vs-real trajectory logging.

**Next:** a dedicated pick-and-place policy (`SO-ARM101-PickPlace-v0` — pick *and set down* at a commanded spot; reward design in the [fork](https://github.com/BTomics/isaac_so_arm101)), and a camera for object localization to replace the currently hardcoded object position.

## Upstream

- Hardware BOM + STLs: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- Assembly + calibration: [LeRobot SO-101 docs](https://huggingface.co/docs/lerobot/so101)
- Isaac Lab envs: [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) (BSD-3-Clause), forked to [BTomics/isaac_so_arm101](https://github.com/BTomics/isaac_so_arm101) — where `SO-ARM101-PickPlace-v0` lives
- Driver / API: [huggingface/lerobot](https://github.com/huggingface/lerobot)
