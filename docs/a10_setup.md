# A10 VM Setup — Isaac Lab Training

Target: SURF Research Cloud Ubuntu VM with the NVIDIA A10. Everything here runs in a terminal on the VM (the desktop view is only needed later for watching `play` render).

Versions pinned by upstream: **Isaac Sim 5.1.0, Isaac Lab 2.3.0, Python 3.11** — all installed automatically by `uv sync`, no manual Isaac install.

**Fast path:** [`scripts/a10_setup.sh`](../scripts/a10_setup.sh) does steps 1–3 in one shot, smoke test included:

```bash
curl -fsSL https://raw.githubusercontent.com/BTomics/SOARMRL/main/scripts/a10_setup.sh | bash
```

The rest of this doc is the same thing manually, for when the script hits a wall.

## 1. Preflight

```bash
nvidia-smi          # A10 visible, driver present
df -h ~             # need ~40 GB free — Isaac Sim's pip packages are huge
```

## 2. Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/MuammerBay/isaac_so_arm101.git
cd isaac_so_arm101
uv sync             # first run downloads Isaac Sim — expect a long wait
```

## 3. Verify before training

```bash
uv run list_envs                                      # tasks are registered
uv run zero_agent --task SO-ARM100-Reach-Play-v0 --headless   # sim boots, arm loads
```

First Isaac Sim launch compiles shaders and can take several minutes — that's normal, not a hang.

## 4. Train

```bash
tmux new -s train   # survive SSH/desktop disconnects
uv run train --task SO-ARM100-Reach-v0 --headless
```

Monitor with TensorBoard against the run's log dir (`logs/rsl_rl/...` inside the repo).

## 5. Evaluate + export

```bash
uv run play --task SO-ARM100-Reach-Play-v0
```

Run this from the desktop view to watch it render. The RSL-RL play script also exports the policy (`policy.pt` / `policy.onnx` under the run's `exported/` folder) — copy that into this repo's `policies/` and commit it; the laptop deployment bridge consumes it.

## Notes

- Domain randomization (mass, friction, PD gains, latency) before the real deployment run — check what the task config already randomizes first.
- Research Cloud VMs get wiped/expire — anything worth keeping goes in this repo or gets copied off the VM same-day.
