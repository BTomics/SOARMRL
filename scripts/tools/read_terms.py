"""Turn Episode_Reward/* numbers into physical units. Reads weights from the config.

Every Episode_Reward/* scalar is the time-averaged WEIGHTED term, so the first
step on any of them is ALWAYS divide by the weight. Gated terms need a second
division by the gate's duty cycle. Doing this by hand has produced the same
error four times in this project - most memorably grasp_top_down 0.1850 read as
1.737, which is impossible, since that quantity is a product of two kernels each
bounded by 1.

    python scripts/tools/read_terms.py --lifting_object 0.2988 \
        --grasp_top_down 0.0904 --reaching_object 0.0638 --object_at_target 4.5663

Pass whatever terms you have; weights are looked up in pickplace_env_cfg.py, so
they stay correct when the config changes. lifting_object sets the lift duty
cycle used to un-gate grasp_top_down.

WHAT THE OUTPUTS MEAN:
  reaching_object   inverts to EE-to-cube distance. ~0.85 is 7.6 mm and grasps;
                    0.64 is 19 mm and cannot close on a 3 cm cube. Gated by
                    (1 - lifted_latch), so it FALLS as the pick improves - a low
                    value late in a run is the gate, not a worse approach.
  grasp_top_down    gated on the cube being aloft, so divide by lift duty. The
                    result is E[near * down_reward], NOT a cos_down: both factors
                    are <= 1, so inverting it gives a LOWER BOUND on cos_down.
                    Never compare it to a workspace cos_down mean - different
                    quantities.
"""

import argparse
import ast
import math
import pathlib
import sys

CFG = (pathlib.Path(__file__).resolve().parent.parent.parent.parent
       / "isaac_so_arm101" / "src" / "isaac_so_arm101" / "tasks" / "pickplace"
       / "pickplace_env_cfg.py")


STEPS_PER_ITER = 24  # environment steps per training iteration


def _terms(kind: str):
    """(attr name, {keyword: value}) for every RewTerm / CurrTerm in the config."""
    tree = ast.parse(CFG.read_text(encoding="utf-8"))
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for stmt in cls.body:
            if not (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "id", "") == kind):
                continue
            kw = {}
            for k in stmt.value.keywords:
                if k.arg == "params" and isinstance(k.value, ast.Dict):
                    kw["params"] = {kk.value: ast.literal_eval(vv) for kk, vv
                                    in zip(k.value.keys, k.value.values)
                                    if isinstance(kk, ast.Constant)
                                    and not isinstance(vv, ast.Call)}
                elif k.arg == "weight":
                    kw["weight"] = ast.literal_eval(k.value)
            yield stmt.targets[0].id, kw


def weights(iteration: int | None):
    """Reward term -> EFFECTIVE weight at `iteration`, curriculum applied.

    The RewTerm weight is only the starting value. modify_reward_weight
    overwrites it once num_steps is passed, so reading the RewTerm alone gives
    the wrong divisor for any term with a curriculum - lifting_object reads 15
    when a run past iteration ~1500 is actually using 3.
    """
    base = {name: kw["weight"] for name, kw in _terms("RewTerm") if "weight" in kw}
    if iteration is None:
        return base, {}

    # apply stages in ascending num_steps, exactly as the managers do
    stages = sorted(
        ((kw["params"]["num_steps"], kw["params"]["term_name"], kw["params"]["weight"])
         for _, kw in _terms("CurrTerm")
         if {"num_steps", "term_name", "weight"} <= set(kw.get("params", {}))),
    )
    applied = {}
    for num_steps, term, weight in stages:
        if iteration >= num_steps / STEPS_PER_ITER:
            base[term] = weight
            applied[term] = (weight, int(num_steps / STEPS_PER_ITER))
    return base, applied


def invert_tanh_kernel(value: float, std: float) -> float:
    """Invert reward = 1 - tanh(d / std) back to d."""
    v = min(max(value, 1e-6), 1 - 1e-6)
    return std * math.atanh(1 - v)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for term in sorted(weights(None)[0]):
        p.add_argument(f"--{term}", type=float, default=None)
    p.add_argument("--iteration", type=int, default=None,
                   help="training iteration, so curriculum'd weights are the EFFECTIVE ones")
    p.add_argument("--resume", action="store_true",
                   help="run used the _RESUME task: all curriculum stages already applied")
    p.add_argument("--grasp_std", type=float, default=0.5, help="grasp_top_down's std")
    p.add_argument("--reach_std", type=float, default=0.05, help="reaching_object's std")
    args = vars(p.parse_args())

    it = 10**9 if args["resume"] else args["iteration"]
    w, applied = weights(it)

    given = {k: v for k, v in args.items() if k in w and v is not None}
    if not given:
        print("no terms given; base weights in the config:")
        for k, v in sorted(w.items()):
            print(f"  {k:38s} {v}")
        return 0

    if it is None:
        print("!! no --iteration and no --resume: using BASE weights. Any term with a\n"
              "   curriculum (lifting_object, action_rate, joint_vel) will be wrong.\n")
    for term, (weight, at) in sorted(applied.items()):
        print(f"curriculum applied: {term} = {weight:g} (fired iter ~{at})")
    if applied:
        print()

    print(f"{'term':38s}{'logged':>10s}{'weight':>9s}{'raw':>10s}")
    raw = {}
    for k, v in sorted(given.items()):
        raw[k] = v / w[k]
        print(f"{k:38s}{v:10.4f}{w[k]:9g}{raw[k]:10.4f}")

    duty = raw.get("lifting_object")
    print()
    if duty is not None:
        print(f"lift duty cycle                       {duty*100:.1f}% of steps with the cube aloft")
    if "reaching_object" in raw:
        # (1 - lifted_latch) * reach, so once the pick works the average is
        # diluted by zeros from every post-lift step and inverting it gives a
        # distance the gripper was never at.
        if duty is not None and duty > 0.01:
            print(f"EE to cube centre                     not invertible - the latch is "
                  f"active ({duty*100:.0f}% lift duty),\n"
                  "  so this average is mostly post-lift zeros, not distance. Only read it "
                  "as\n  a distance before the pick exists (~0.85 grasps, 0.64 cannot close).")
        else:
            d_mm = invert_tanh_kernel(raw["reaching_object"], args["reach_std"]) * 1000
            print(f"EE to cube centre                     {d_mm:.0f} mm   "
                  "(valid: no lift yet, so the gate is fully open)")
    if "grasp_top_down" in raw:
        if duty is None:
            print("grasp_top_down: pass --lifting_object too, it is gated on the cube being aloft")
        else:
            cond = raw["grasp_top_down"] / duty
            print(f"grasp_top_down per lifted step        {cond:.3f}   = E[near * down_reward], both <= 1")
            if cond >= 1.0:
                print("  !! above 1.0 is impossible - a division was skipped somewhere")
            else:
                cos = 1 - invert_tanh_kernel(cond, args["grasp_std"])
                deg = math.degrees(math.acos(min(max(cos, -1.0), 1.0)))
                print(f"cos_down                              >= {cos:.3f}  ({deg:.0f} deg off vertical at worst)")
                print("  lower bound only - the near factor is folded in. Not comparable to a")
                print("  workspace cos_down mean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
