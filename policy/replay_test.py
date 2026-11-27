"""Replay test: a demo's recorded actions must succeed in the simulator.

For each demo: build the env from the dataset's saved kwargs, reset to the saved
scene and initial state, play the recorded actions open loop through
`policy.sim.lerobot_to_env_action`, and check `env._check_success()` after every
step. Also reports how far the simulator state drifts from the recorded states.

    python -m policy.replay_test --tasks OpenDrawer TurnOnMicrowave --n 10
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from policy import paths, sim


def replay(env, ds_dir: Path, ep: int, binarize: bool = True, extra_steps: int = 0) -> dict:
    t0 = time.time()
    states, _, ep_meta = sim.episode_extras(ds_dir, ep)
    sim.reset_to_episode(env, ds_dir, ep)
    acts = sim.episode_actions(ds_dir, ep)
    env_acts = sim.lerobot_to_env_action(acts, binarize=binarize)
    first_success, max_err = -1, 0.0
    for t, a in enumerate(env_acts):
        env.step(a)
        if t + 1 < len(states):
            max_err = max(max_err, float(np.linalg.norm(states[t + 1] - env.sim.get_state().flatten())))
        if first_success < 0 and env._check_success():
            first_success = t
    for k in range(extra_steps):           # hold still: zero motion, keep the last gripper command
        a = env_acts[-1].copy()
        a[:6] = 0
        a[7:11] = 0
        env.step(a)
        if first_success < 0 and env._check_success():
            first_success = len(env_acts) + k
    return {"episode": ep, "steps": len(acts), "success": int(first_success >= 0),
            "first_success_step": first_success, "final_success": int(bool(env._check_success())),
            "max_state_err": round(max_err, 5), "seconds": round(time.time() - t0, 2),
            "lang": ep_meta.get("lang", "")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--no-binarize", action="store_true")
    ap.add_argument("--extra-steps", type=int, default=0)
    ap.add_argument("--out", default=None, help="CSV of per-demo results")
    args = ap.parse_args()
    rows = []
    for task in args.tasks:
        ds = paths.task_dir(task)
        env = sim.make_env(ds, cams=None)
        for ep in range(args.start, args.start + args.n):
            r = {"task": task, **replay(env, ds, ep, binarize=not args.no_binarize, extra_steps=args.extra_steps)}
            rows.append(r)
            print(r, flush=True)
        env.close()
        ok = sum(r["success"] for r in rows if r["task"] == task)
        print(f"REPLAY {task}: {ok}/{args.n} succeeded", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
