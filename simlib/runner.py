"""Parallel episode runner: one process and one log file per episode, results CSV rows.

An episode function has the signature

    def my_episode(task: str, seed: int, episode: int, **kwargs) -> dict

and returns at least {"success": bool}. Optional keys: llm_calls, input_tokens,
output_tokens, cost_usd (default 0), seconds. If the function does not return `seconds`, the
runner uses the wall time of the whole child process (env build included). It fills `code_commit`.
It is named as "package.module:function" so each child process can import it fresh.

CLI:
    PYTHONPATH=. python -m simlib.runner --fn simlib.episodes:random_actions \
        --tasks OpenDrawer TurnOnMicrowave --episodes 5 --workers 4 --timeout 900 \
        --out results/2026-11-27_policy_only_zero-test.csv

Each episode i of task T uses seed = seed0 + i. Logs go to <out without .csv>_logs/T_epNNN.log.
A crashed or timed-out episode is written with success=False and its log says why.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COLUMNS = ["task", "episode", "seed", "success", "seconds", "llm_calls", "input_tokens",
           "output_tokens", "cost_usd", "code_commit"]
REPO = Path(__file__).resolve().parent.parent


def code_commit(repo: Path = REPO) -> str:
    """Short commit hash of this repo, with '-dirty' if there are uncommitted changes."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=repo, capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_row(task, episode, seed, result: dict | None, seconds: float, commit: str) -> dict:
    r = result or {}
    return {
        "task": task, "episode": int(episode), "seed": int(seed),
        "success": bool(r.get("success", False)),
        "seconds": round(float(r.get("seconds", seconds)), 2),
        "llm_calls": int(r.get("llm_calls", 0)), "input_tokens": int(r.get("input_tokens", 0)),
        "output_tokens": int(r.get("output_tokens", 0)),
        "cost_usd": float(r.get("cost_usd", 0.0)), "code_commit": commit,
    }


def load_fn(spec: str):
    mod, _, name = spec.partition(":")
    return getattr(importlib.import_module(mod), name)


def _child(fn_spec, task, seed, episode, kwargs_json, result_path):
    """Runs inside the child process."""
    fn = load_fn(fn_spec)
    t = time.time()
    res = fn(task=task, seed=seed, episode=episode, **json.loads(kwargs_json))
    res = dict(res)
    res.setdefault("seconds_inside", time.time() - t)
    Path(result_path).write_text(json.dumps(res, default=float))


def run_one(fn_spec: str, task: str, seed: int, episode: int, log_dir: Path,
            timeout: float, kwargs: dict | None = None, env_vars: dict | None = None) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{task}_ep{episode:03d}.log"
    res_path = log_dir / f"{task}_ep{episode:03d}.json"
    res_path.unlink(missing_ok=True)
    cmd = [sys.executable, "-m", "simlib.runner", "--child", fn_spec, task, str(seed),
           str(episode), json.dumps(kwargs or {}), str(res_path)]
    env = {**os.environ, **(env_vars or {})}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    t = time.time()
    status = "ok"
    with open(log, "w") as f:
        f.write(f"# {' '.join(cmd)}\n")
        f.flush()
        try:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               cwd=REPO, env=env)
            if p.returncode != 0:
                status = f"crashed (exit {p.returncode})"
        except subprocess.TimeoutExpired:
            status = f"timeout after {timeout:.0f}s"
    secs = time.time() - t
    result = json.loads(res_path.read_text()) if status == "ok" and res_path.exists() else None
    if result is None and status == "ok":
        status = "no result file"
    with open(log, "a") as f:
        f.write(f"\n# runner: status={status} seconds={secs:.1f} result={result}\n")
    return {"result": result, "seconds": secs, "status": status}


def run_episodes(fn_spec: str, tasks, episodes: int, out_csv: str | Path, seed0: int = 0,
                 workers: int = 4, timeout: float = 1800, kwargs: dict | None = None,
                 env_vars: dict | None = None, log_dir: str | Path | None = None) -> list[dict]:
    """Run `episodes` episodes of each task in parallel, append rows to out_csv as they finish."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir else out_csv.with_name(out_csv.stem + "_logs")
    commit = code_commit()
    jobs = [(t, i, seed0 + i) for t in tasks for i in range(episodes)]
    new = not out_csv.exists()
    rows = []
    with open(out_csv, "a", newline="") as f, ThreadPoolExecutor(workers) as pool:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        futs = {pool.submit(run_one, fn_spec, t, s, i, log_dir, timeout, kwargs, env_vars):
                (t, i, s) for t, i, s in jobs}
        for fut in futs:  # rows written in submission order; each waits for its own job
            t, i, s = futs[fut]
            r = fut.result()
            row = make_row(t, i, s, r["result"], r["seconds"], commit)
            w.writerow(row)
            f.flush()
            rows.append(row)
            print(f"{t} ep{i} seed{s}: success={row['success']} {row['seconds']}s [{r['status']}]",
                  flush=True)
    return rows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--child":
        fn_spec, task, seed, episode, kw, res = argv[1:7]
        _child(fn_spec, task, int(seed), int(episode), kw, res)
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--fn", required=True, help="module:function episode function")
    ap.add_argument("--tasks", nargs="+", required=True, help="task names, or 'atomic_seen'")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kwargs", default="{}", help="JSON dict passed to the episode function")
    a = ap.parse_args(argv)
    from simlib.tasks import ATOMIC_SEEN
    tasks = list(ATOMIC_SEEN) if a.tasks == ["atomic_seen"] else a.tasks
    rows = run_episodes(a.fn, tasks, a.episodes, a.out, a.seed0, a.workers, a.timeout,
                        json.loads(a.kwargs))
    ok = sum(r["success"] for r in rows)
    print(f"{ok}/{len(rows)} succeeded. CSV: {a.out}")


if __name__ == "__main__":
    main()
