"""PROPOSALS P-1: which drawer did the policy open? Rescore saved OpenDrawer rollouts, zero GPU.

For each saved trajectory (<eval_media>/<note>/OpenDrawer_epNNN_traj.npz, written with --save-traj), rebuild the
scene from its seed (seed0 + episode, determinism fix on, as policy/eval.py), load the saved final simulator
state, and read how open every Drawer fixture in the kitchen is (get_door_state, 0 closed to 1 open).
Per episode: number of drawers, target open fraction, the most-opened other drawer, and the outcome class:
  target      target opened >= 0.95 (task success)
  partial     target opened 0.1 to 0.95, no other drawer more open
  wrong       another drawer opened more than the target and >= 0.1
  none        nothing opened >= 0.1
    python -m policy.rescore_drawers <eval_media_dir> [--seed0 1000]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

import numpy as np

from policy import paths, sim


def drawers(env):
    from robocasa.models.fixtures.cabinets import Drawer
    return {name: fx for name, fx in env.fixtures.items() if isinstance(fx, Drawer)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    env = sim.make_env(paths.task_dir("OpenDrawer"), cams=None)
    rows = []
    for d in a.dirs:
        for f in sorted(glob.glob(f"{d}/OpenDrawer_ep*_traj.npz")):
            ep = int(re.search(r"_ep(\d+)_traj", f).group(1))
            z = np.load(f)
            env.rng = np.random.default_rng(a.seed0 + ep)
            env.reset()
            env.sim.set_state_from_flattened(z["final_state"])
            env.sim.forward()
            dr = drawers(env)
            state = {n: float(np.mean(list(fx.get_door_state(env=env).values()))) for n, fx in dr.items()}
            target = env.drawer.name
            t_open = state.get(target, float("nan"))
            others = {n: v for n, v in state.items() if n != target}
            o_name, o_open = max(others.items(), key=lambda x: x[1]) if others else ("", 0.0)
            if t_open >= 0.95:
                cls = "target"
            elif o_open >= 0.1 and o_open > t_open:
                cls = "wrong"
            elif t_open >= 0.1:
                cls = "partial"
            else:
                cls = "none"
            rows.append({"run": Path(d).name, "episode": ep, "success": int(z["success"]), "n_drawers": len(dr),
                         "target_open": round(t_open, 3), "other_max_open": round(o_open, 3), "class": cls})
    env.close()
    by = collections.defaultdict(collections.Counter)
    nd = collections.defaultdict(list)
    for r in rows:
        by[r["run"]][r["class"]] += 1
        nd[r["run"]].append(r["n_drawers"])
    for run, c in sorted(by.items()):
        print("RESCORE", run, dict(c), "n=", sum(c.values()), "drawers per kitchen: median",
              int(np.median(nd[run])), "min", min(nd[run]), "max", max(nd[run]), flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
