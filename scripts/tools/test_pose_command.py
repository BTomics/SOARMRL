"""Exercise the REAL ObjectAwarePoseCommand._resample_command without Isaac Lab.

The place goal sits on the table in the same plane as the cube's spawn box, so a
plain uniform command can put the goal on top of the cube. The was-lifted latch
stops that being farmable — the cube still has to be picked — but it makes a large
share of episodes trivial: pick straight up, set straight down. ObjectAwarePose-
Command rejection-samples the goal to keep it min_separation away, with a
deterministic y-edge fallback for the rare env that never clears.

That fallback is the part that would misbehave silently on the A10, so this stubs
just enough of isaaclab to import the real module and run the real method.

    python scripts/tools/test_pose_command.py         # exits 1 if any goal is too close

Re-run after ANY change to the spawn or goal box: the fallback's correctness
depends on the y range spanning more than min_separation.
"""

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch

# Boxes must match pickplace_env_cfg.py. They are DECOUPLED again: the goal box
# was pushed out to x[0.20,0.35] while the spawn box stayed at x[0.15,0.30], so
# the two only overlap on x[0.20,0.30] and the rejection sampler has less room.
SPAWN_X, SPAWN_Y = (0.15, 0.30), (-0.20, 0.20)
GOAL_X, GOAL_Y, GOAL_Z = (0.20, 0.35), (-0.20, 0.20), (0.015, 0.020)
MIN_SEP = 0.12
N = 20_000

MDP = (Path(__file__).resolve().parent.parent.parent.parent
       / "isaac_so_arm101" / "src" / "isaac_so_arm101" / "tasks" / "pickplace" / "mdp")


@dataclass
class _Ranges:
    pos_x: tuple = (0.0, 1.0)
    pos_y: tuple = (0.0, 1.0)
    pos_z: tuple = (0.0, 1.0)
    roll: tuple = (0.0, 0.0)
    pitch: tuple = (0.0, 0.0)
    yaw: tuple = (0.0, 0.0)


class _StubCfg:
    Ranges = _Ranges

    def __init__(self, **kw):
        self.ranges = _Ranges()
        for k, v in kw.items():
            setattr(self, k, v)


class _StubCommand:
    """Stands in for UniformPoseCommand: draws position uniformly from cfg.ranges."""

    def __init__(self, cfg, env):
        self.cfg, self.num_envs, self.device = cfg, env.num_envs, "cpu"
        self.pose_command_b = torch.zeros(env.num_envs, 7)
        self.draws = 0

    def _resample_command(self, env_ids):
        n = len(env_ids)
        self.draws += n
        buf = torch.empty(n)
        self.pose_command_b[env_ids, 0] = buf.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = buf.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = buf.uniform_(*self.cfg.ranges.pos_z)


class _Stub:
    pass


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def install_stubs():
    assets = _module("isaaclab.assets", RigidObject=_Stub)
    envs = _module("isaaclab.envs", ManagerBasedRLEnv=object)
    utils = _module("isaaclab.utils", configclass=lambda c: c)
    _module("isaaclab.envs.mdp", UniformPoseCommand=_StubCommand, UniformPoseCommandCfg=_StubCfg)
    # subtract_frame_transforms with the base at the origin is a plain translation
    _module("isaaclab.utils.math", subtract_frame_transforms=lambda p, q, t: (t - p, None))
    root = _module("isaaclab")
    root.assets, root.envs, root.utils = assets, envs, utils


def main():
    install_stubs()
    sys.path.insert(0, str(MDP))
    import commands  # noqa: E402  (needs the stubs in place first)

    torch.manual_seed(0)
    obj, robot = _Stub(), _Stub()
    obj.data, robot.data = _Stub(), _Stub()
    obj.data.root_pos_w = torch.stack([
        torch.empty(N).uniform_(*SPAWN_X),
        torch.empty(N).uniform_(*SPAWN_Y),
        torch.full((N,), 0.015),
    ], dim=1)
    robot.data.root_state_w = torch.zeros(N, 7)  # base at the origin => base frame == world

    cfg = _StubCfg(asset_name="robot", object_name="object",
                   min_separation=MIN_SEP, max_resample_tries=8)
    cfg.ranges = _Ranges(pos_x=GOAL_X, pos_y=GOAL_Y, pos_z=GOAL_Z)
    env = types.SimpleNamespace(num_envs=N, scene={"object": obj, "robot": robot})

    cmd = commands.ObjectAwarePoseCommand(cfg, env)
    cmd._resample_command(torch.arange(N))

    goal = cmd.pose_command_b[:, :2]
    d = torch.norm(goal - obj.data.root_pos_w[:, :2], dim=1)
    violations = int((d < MIN_SEP - 1e-6).sum())
    on_edge = int(((goal[:, 1] == GOAL_Y[0]) | (goal[:, 1] == GOAL_Y[1])).sum())

    print(f"envs                {N}")
    print(f"violations (<{MIN_SEP})  {violations}")
    print(f"separation min/mean {d.min():.3f} / {d.mean():.3f}")
    print(f"goal x range        {goal[:,0].min():.3f} .. {goal[:,0].max():.3f}  (box {GOAL_X})")
    print(f"goal y range        {goal[:,1].min():.3f} .. {goal[:,1].max():.3f}  (box {GOAL_Y})")
    print(f"draws per env       {cmd.draws/N:.2f}")
    print(f"fallback fired      {on_edge} ({on_edge/N*100:.2f}%)")
    hist = torch.histc(goal[:, 1], bins=8, min=GOAL_Y[0], max=GOAL_Y[1])
    print(f"goal y histogram    {[int(v) for v in hist]}  (should not pile up at the edges)")

    if violations:
        print("\nFAIL: separation constraint violated")
        return 1
    print("\nPASS: every goal is at least min_separation from its cube")
    return 0


if __name__ == "__main__":
    sys.exit(main())
