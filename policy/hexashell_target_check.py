"""Check that HexaShell's `target` split builds the same scenes as our evaluation.

Two sides, each dumps the scene facts of (task, seed) to JSON; `compare` lines them up.

* `ours`: `policy.sim.make_env` (the dataset's env kwargs), then for each seed in turn
  `env.rng = np.random.default_rng(seed); env.reset()`, exactly as `policy/eval.py` does
  (one env per task, seeds in order).
* `hexa`: HexaShell's `CasaSim(task, split="target")` (with `setup/patches/hexashell_target_split.patch`
  applied), `sim.reset(seed)` for each seed in turn, as its warm simulator server does.

Facts: layout and style ids, the ep_meta object configs (name, category, model file), fixture refs,
the language instruction, every object's initial body position, the robot base position, and the
task's target part (`simlib.truth.target_part`). Ground truth here is for this check only.

    python -m policy.hexashell_target_check dump --side ours --out ours.json
    python -m policy.hexashell_target_check dump --side hexa --hexashell ~/apollo-lab/HexaShell --out hexa.json
    python -m policy.hexashell_target_check compare ours.json hexa.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TASKS = ["PickPlaceCounterToCabinet", "OpenDrawer", "TurnOnMicrowave"]
SEEDS = list(range(1000, 1005))
POS_TOL = 1e-4  # metres


def _r(a, n=5):
    return [round(float(x), n) for x in np.asarray(a).reshape(-1)]


def facts(env, task: str) -> dict:
    from simlib.truth import object_position, target_part
    meta = env.get_ep_meta()
    objs = []
    for c in meta.get("object_cfgs", []):
        info = c.get("info", {}) or {}
        objs.append({"name": c.get("name"), "cat": info.get("cat"),
                     "model": Path(str(info.get("mjcf_path", ""))).parent.name})
    m = env.sim.model
    base = env.sim.data.body_xpos[m.body_name2id("mobilebase0_base")]
    out = {
        "layout_id": int(env.layout_id), "style_id": int(env.style_id),
        "lang": meta.get("lang"),
        "objects": objs,
        "fixture_refs": {k: str(v) for k, v in sorted((meta.get("fixture_refs") or {}).items())},
        "object_pos": {n: _r(object_position(env, n)) for n in sorted(env.objects)},
        "base_pos": _r(base),
    }
    try:
        p = target_part(env, task)
        out["target"] = {"fixture": p.fixture, "element": p.element, "pos": _r(p.pos)}
    except Exception as e:  # noqa: BLE001
        out["target"] = {"error": repr(e)}
    return out


def counter_fix():
    """HexaShell's `[roboshell-determinism]` RoboCasa change, as a monkeypatch: in
    `robocasa/models/fixtures/counter.py`, `list(set(valid_geoms))` iterates XML elements in memory-address
    order; `dict.fromkeys` keeps insertion order. `set` is used nowhere else in that module."""
    import robocasa.models.fixtures.counter as C
    C.set = lambda it=(): dict.fromkeys(it)


def dump(side: str, tasks, seeds, hexashell: str | None, fix: bool = False) -> dict:
    res = {"side": side, "counter_fix": fix, "rows": {}}
    import robocasa
    if fix:
        counter_fix()
    res["robocasa_file"] = robocasa.__file__
    for task in tasks:
        t0 = time.time()
        if side == "ours":
            from policy import paths, sim
            env = sim.make_env(paths.task_dir(task), cams=["left", "right"], size=256)
            for s in seeds:
                env.rng = np.random.default_rng(s)
                env.reset()
                res["rows"][f"{task}:{s}"] = facts(env, task)
        else:
            sys.path.insert(0, str(Path(hexashell).expanduser()))
            from hexashell.benches.robocasa.sim import CasaSim
            cs = CasaSim(task, size=256, split="target")
            for s in seeds:
                f = cs.reset(s)
                row = facts(cs.env, task)
                row["hexa_facts"] = {k: f[k] for k in ("layout_id", "style_id", "kitchen", "instruction")}
                res["rows"][f"{task}:{s}"] = row
        print(f"{task}: {len(seeds)} seeds in {time.time() - t0:.0f} s", flush=True)
    return res


def compare(a: dict, b: dict) -> list[dict]:
    out = []
    for key in sorted(set(a["rows"]) | set(b["rows"])):
        x, y = a["rows"].get(key), b["rows"].get(key)
        task, seed = key.split(":")
        row = {"task": task, "seed": int(seed)}
        if x is None or y is None:
            row["result"] = "missing"
            out.append(row)
            continue
        diffs = []
        for k in ("layout_id", "style_id", "lang", "objects", "fixture_refs"):
            if x[k] != y[k]:
                diffs.append(k)
        if set(x["object_pos"]) != set(y["object_pos"]):
            diffs.append("object_names")
        dpos = max([float(np.max(np.abs(np.subtract(x["object_pos"][n], y["object_pos"][n]))))
                    for n in set(x["object_pos"]) & set(y["object_pos"])] or [0.0])
        dbase = float(np.max(np.abs(np.subtract(x["base_pos"], y["base_pos"]))))
        dtgt = (float(np.max(np.abs(np.subtract(x["target"]["pos"], y["target"]["pos"]))))
                if "pos" in x["target"] and "pos" in y["target"] else float("nan"))
        if dpos > POS_TOL:
            diffs.append("object_pos")
        if dbase > POS_TOL:
            diffs.append("base_pos")
        if not dtgt <= POS_TOL:
            diffs.append("target_pos")
        row.update(layout=f"{x['layout_id']}/{y['layout_id']}", style=f"{x['style_id']}/{y['style_id']}",
                   max_obj_dpos_m=round(dpos, 5), base_dpos_m=round(dbase, 5), target_dpos_m=round(dtgt, 5),
                   result="match" if not diffs else "MISMATCH: " + ",".join(diffs))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dump")
    d.add_argument("--side", choices=["ours", "hexa"], required=True)
    d.add_argument("--hexashell", default="~/apollo-lab/HexaShell")
    d.add_argument("--tasks", nargs="+", default=TASKS)
    d.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    d.add_argument("--counter-fix", action="store_true", help="ours: apply HexaShell's determinism fix")
    d.add_argument("--out", required=True)
    c = sp.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--csv", default=None)
    args = ap.parse_args()
    if args.cmd == "dump":
        res = dump(args.side, args.tasks, args.seeds, args.hexashell, args.counter_fix)
        Path(args.out).write_text(json.dumps(res, indent=1))
        return
    a, b = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    rows = compare(a, b)
    print(f"A: {args.a} ({a['robocasa_file']})\nB: {args.b} ({b['robocasa_file']})")
    for r in rows:
        print("  ".join(f"{k}={v}" for k, v in r.items()))
    n = sum(r["result"] == "match" for r in rows)
    print(f"{n}/{len(rows)} match")
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
