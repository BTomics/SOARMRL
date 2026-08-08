"""Static check on the PickPlace task config. Runs WITHOUT Isaac Lab installed.

Catches the class of mistake that only shows up as a crash on the A10 an hour
after you push:

  * a params key that is not a parameter of the function it is passed to;
  * a required parameter left unsupplied;
  * a curriculum term_name that does not name a real reward term;
  * a term that indexes joint_ids without its SceneEntityCfg in params.

That last one is subtle and has bitten once. SceneEntityCfg.joint_ids defaults to
slice(None) and only becomes a list of indices when .resolve(scene) turns
joint_names into joint_ids. The managers resolve ONLY the SceneEntityCfg objects
they find in a term's params dict — one left as a function-signature default is
never resolved — so joint_ids[0] raises "TypeError: 'slice' object is not
subscriptable" on the first step. Worse is the tempting fix: indexing with
joint_ids and averaging silently gives the mean of ALL SIX joint angles, which is
arm posture rather than gripper aperture, and trains happily on garbage.

    python scripts/tools/check_task_wiring.py         # exits 1 on any problem

Isaac Lab builtins (action_rate_l2, joint_deviation_l1, ...) are listed but NOT
checked — their source is not importable here. A wrong builtin name fails at
startup within seconds, so the cheap check remains: confirm every term you expect
appears in the logged reward-term list when training starts.
"""

import ast
import pathlib
import sys

TASK = (pathlib.Path(__file__).resolve().parent.parent.parent.parent
        / "isaac_so_arm101" / "src" / "isaac_so_arm101" / "tasks" / "pickplace")
TERM_TYPES = ("RewTerm", "CurrTerm", "DoneTerm", "EventTerm")


def local_functions(mdp_dir: pathlib.Path):
    """name -> (all param names, required params, calls, touches joint_ids)."""
    sigs, calls, reads_joints = {}, {}, set()
    for path in sorted(mdp_dir.glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            a = node.args
            names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            n_req = len(a.posonlyargs) + len(a.args) - len(a.defaults)
            required = [x.arg for x in (a.posonlyargs + a.args)[:n_req]
                        if x.arg not in ("env", "env_ids")]
            sigs[node.name] = (names, required)
            calls[node.name] = {c.func.id for c in ast.walk(node)
                                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
            if any(isinstance(s, ast.Attribute) and s.attr == "joint_ids"
                   for s in ast.walk(node)):
                reads_joints.add(node.name)

    # a term also needs its cfg resolved if anything it CALLS reads a joint
    changed = True
    while changed:
        changed = False
        for fn, callees in calls.items():
            if fn not in reads_joints and (callees & reads_joints):
                reads_joints.add(fn)
                changed = True
    return sigs, reads_joints


def config_terms(cfg_path: pathlib.Path):
    """(class name, attribute name, Call node) for every wired term."""
    tree = ast.parse(cfg_path.read_text(encoding="utf-8"))
    out = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for stmt in cls.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                    and isinstance(stmt.targets[0], ast.Name):
                name, value = stmt.targets[0].id, stmt.value
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name, value = stmt.target.id, stmt.value
            else:
                continue
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                    and value.func.id in TERM_TYPES:
                out.append((cls.name, name, value))
    return out


def main():
    sigs, reads_joints = local_functions(TASK / "mdp")
    terms = config_terms(TASK / "pickplace_env_cfg.py")
    rewards = {name for _, name, node in terms if node.func.id == "RewTerm"}

    checked, skipped, problems = [], [], []
    for _, tname, node in terms:
        kw = {k.arg: k.value for k in node.keywords}
        fn = kw.get("func")
        fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "?")
        params = {}
        if isinstance(kw.get("params"), ast.Dict):
            params = {k.value: v for k, v in zip(kw["params"].keys, kw["params"].values)
                      if isinstance(k, ast.Constant)}

        if fname == "modify_reward_weight":
            ref = params.get("term_name")
            ref = ref.value if isinstance(ref, ast.Constant) else None
            steps = params.get("num_steps")
            steps = steps.value if isinstance(steps, ast.Constant) else None
            if ref not in rewards:
                problems.append(f"curriculum {tname}: term_name {ref!r} is not a reward term")
            else:
                # num_steps counts ENVIRONMENT steps, ~24 per training iteration.
                # This units trap has cost this project two runs.
                at = f"iter ~{steps // 24}" if isinstance(steps, int) else "?"
                checked.append(f"{'curriculum ' + tname:34s} -> {ref:26s} fires {at}")
            continue

        if fname not in sigs:
            skipped.append(f"{tname:34s} {fname}")
            continue

        names, required = sigs[fname]
        if bad := [p for p in params if p not in names]:
            problems.append(f"{tname}: {fname} has no parameter(s) {bad}")
        if missing := [r for r in required if r not in params]:
            problems.append(f"{tname}: {fname} missing required {missing}")

        note = ""
        if fname in reads_joints:
            cfgs = [p for p in names if p.endswith("_cfg") and "robot" in p]
            if not any(c in params for c in cfgs):
                problems.append(
                    f"{tname}: {fname} indexes joint_ids but none of {cfgs} is in params "
                    "-> unresolved SceneEntityCfg, joint_ids stays slice(None), crashes on step 1")
            else:
                note = "   [joint read, cfg passed]"
        checked.append(f"{tname:34s} {fname:34s}{note}")

    print("CHECKED")
    for line in checked:
        print("  " + line)
    print("\nSKIPPED (Isaac Lab builtins, not importable here)")
    for line in skipped:
        print("  " + line)

    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print("  ! " + p)
        return 1
    print(f"\nPASS: {len(checked)} terms wired with valid parameters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
