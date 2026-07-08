#!/usr/bin/env bash
# One-shot setup for a fresh SURF Research Cloud A10 VM (Ubuntu).
# Installs uv + isaac_so_arm101 (which pulls its own pinned Isaac Sim 5.1 / Isaac Lab 2.3)
# and runs a headless smoke test. Safe to re-run; skips anything already done.
#
# Usage:  bash a10_setup.sh
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME}"
REPO_URL="https://github.com/MuammerBay/isaac_so_arm101.git"
REPO_DIR="$WORKDIR/isaac_so_arm101"

echo "==> Preflight checks"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi missing or failing — no usable GPU/driver on this VM." >&2
    echo "Pick a GPU flavor in Research Cloud or install the NVIDIA driver first." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

avail_gb=$(df --output=avail -BG "$WORKDIR" | tail -1 | tr -dc '0-9')
if [ "$avail_gb" -lt 40 ]; then
    echo "ERROR: only ${avail_gb}G free at $WORKDIR — Isaac Sim needs ~40G." >&2
    echo "Attach/resize storage, or rerun with SKIP_DISK_CHECK=1 if you know better." >&2
    [ "${SKIP_DISK_CHECK:-0}" = "1" ] || exit 1
fi
echo "Disk OK: ${avail_gb}G free"

for pkg in git curl tmux; do
    if ! command -v "$pkg" >/dev/null 2>&1; then
        echo "==> Installing $pkg"
        sudo apt-get update -qq && sudo apt-get install -y -qq "$pkg"
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

if [ ! -d "$REPO_DIR" ]; then
    echo "==> Cloning isaac_so_arm101"
    git clone "$REPO_URL" "$REPO_DIR"
fi

# Several old deps (flatdict et al., see upstream PR #87) import pkg_resources
# at build time, which new setuptools removed — pin old setuptools for all
# source builds until upstream fixes its tree
cd "$REPO_DIR"
if ! grep -q "build-constraint-dependencies" pyproject.toml; then
    echo "==> Patching pyproject.toml: setuptools<72 for source builds"
    python3 - <<'EOF'
import re, pathlib
p = pathlib.Path("pyproject.toml")
t = p.read_text()
if re.search(r"^\[tool\.uv\]$", t, re.M):
    t = re.sub(r"^\[tool\.uv\]$",
               '[tool.uv]\nbuild-constraint-dependencies = ["setuptools<72"]',
               t, count=1, flags=re.M)
else:
    t += '\n[tool.uv]\nbuild-constraint-dependencies = ["setuptools<72"]\n'
p.write_text(t)
EOF
fi

echo "==> uv sync (downloads Isaac Sim — this is the long part, tens of GB)"
uv sync

# Auto-accept the Isaac Sim EULA so the first launch doesn't stall on a prompt
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

echo "==> Smoke test: zero_agent, headless (first launch compiles shaders — several minutes)"
uv run zero_agent --task SO-ARM100-Reach-Play-v0 --headless

cat <<'EOF'

==> SETUP COMPLETE. To start training:

    tmux new -s train
    cd ~/isaac_so_arm101
    uv run train --task SO-ARM100-Reach-v0 --headless

Detach with Ctrl+B then D; reattach with `tmux attach -t train`.
Curves: `uv run tensorboard --logdir logs/rsl_rl` (or pip-installed tensorboard).
After training: `uv run play --task SO-ARM100-Reach-Play-v0` — watch it in the
desktop view; the run's exported/ folder gets policy.pt/policy.onnx for the bridge.
EOF
