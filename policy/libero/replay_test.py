"""LIBERO replay test: a demo's recorded actions must succeed in the simulator.

For each demo: build the task's env (no cameras), reset, set the demo's first saved MuJoCo state,
play the recorded 7-number actions open loop, check `env.check_success()` after every step.
Loads the demo's own scene XML first (as LIBERO's create_dataset.py), unless --no-xml.
Also reports the state error at t=0 (env state8 against the demo's ee_states + gripper_states)
and the largest drift of the MuJoCo state from the recorded states.

    python -m policy.libero.replay_test --suites libero_spatial libero_object --tasks 0 1 2 --n 10
Pass: at least 9 of 10 demos succeed per tested task.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from policy.libero import sim


def replay(env, d, extra_steps: int = 0, xml: bool = True) -> dict:
    t0 = time.time()
    acts, states = d["actions"][()], d["states"][()]
    rec0 = np.concatenate([d["obs/ee_states"][0], d["obs/gripper_states"][0]])
    obs = sim.reset_to_demo(env, d) if xml else sim.reset_to_state(env, states[0])
    err0 = float(np.abs(sim.state8(obs) - rec0).max())
    first, max_err, err1 = -1, 0.0, float("nan")
    for t, a in enumerate(acts):
        obs, _, _, _ = env.step(a)
        if t == 0:   # LIBERO's recorded obs[i] is taken after action i; this checks that
            err1 = float(np.abs(sim.state8(obs) - rec0).max())
        if t + 1 < len(states):
            max_err = max(max_err, float(np.linalg.norm(states[t + 1] - env.get_sim_state())))
        if first < 0 and env.check_success():
            first = t
    for k in range(extra_steps):
        env.step(sim.NOOP if acts[-1][6] < 0 else np.r_[np.zeros(6), 1.0])
        if first < 0 and env.check_success():
            first = len(acts) + k
    return {"steps": len(acts), "success": int(0 <= first < len(acts)), "success_hold": int(first >= 0),
            "first_success_step": first,
            "final_success": int(bool(env.check_success())), "state8_err0": round(err0, 6),
            "state8_err_after1": round(err1, 6),
            "max_state_err": round(max_err, 5), "seconds": round(time.time() - t0, 2)}


def main() -> None:
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object"])
    ap.add_argument("--tasks", nargs="+", type=int, default=[0, 1, 2], help="task ids within each suite")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--extra-steps", type=int, default=0, help="hold still this long after the last action")
    ap.add_argument("--no-xml", action="store_true", help="skip loading the demo's own scene XML")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows, summary = [], []
    for suite in args.suites:
        for tid in args.tasks:
            info = sim.task_info(suite, tid)
            env = sim.make_env(info["bddl"], render=False)
            with h5py.File(info["demo"], "r") as f:
                keys = sim.demo_keys(f)[args.start:args.start + args.n]
                for k in keys:
                    r = {"suite": suite, "task_id": tid, "task": info["name"], "demo": k,
                         **replay(env, f["data"][k], args.extra_steps, xml=not args.no_xml)}
                    rows.append(r)
                    print(r, flush=True)
            env.close()
            mine = [r for r in rows if r["suite"] == suite and r["task_id"] == tid]
            ok = sum(r["success"] for r in mine)
            hold = sum(r["success_hold"] for r in mine)
            line = f"REPLAY {suite} task {tid} ({info['name']}): {ok}/{len(keys)} succeeded"
            if args.extra_steps:
                line += f" ({hold}/{len(keys)} with {args.extra_steps} hold steps after the last action)"
            summary.append((ok, len(keys), line))
            print(line, flush=True)
    print("\n".join(s[2] for s in summary))
    passed = all(ok >= 0.9 * n for ok, n, _ in summary)
    print("REPLAY TEST", "PASS" if passed else "FAIL", "(need >= 90% per task)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
