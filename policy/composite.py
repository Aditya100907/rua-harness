"""Part Q: run composite tasks by chaining atomic commands from ground truth.

The target composite datasets ship per-frame subtask labels (`annotation.human.subtask`
-> instruction, `annotation.human.subtask_name`, `subtask_idx`). For a demo we take its
subtask sequence, reset to the demo's scene, and hand each subtask's instruction to the
policy in turn with `run_skill` (time budget: 1.5 x the demo's segment length). The
composite task succeeds if `env._check_success()` turns true at any step.

Where chains break: after each segment we evaluate RoboCasa's own subtask labeler
(`robocasa.utils.subtask_labeling`, privileged state, grading only) on the current state
and record which stage the scene has reached.

Normalization: the policy was trained on atomic tasks with per-task statistics, so each
subtask borrows the statistics of the atomic task its name maps to (`ATOMIC_FOR`).

    python -m policy.composite --ckpt CKPT --tasks PrepareCoffee --episodes 10
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np

from policy import paths, sim

# subtask name keywords -> the atomic task whose normalization statistics to borrow
ATOMIC_FOR = [
    (r"drawer.*open|open.*drawer", "OpenDrawer"), (r"open.*(cab|door)|(cab|door).*open", "OpenCabinet"),
    (r"close.*fridge|fridge.*close", "CloseFridge"), (r"close.*(toaster|oven)", "CloseToasterOvenDoor"),
    (r"microwave|press|button", "TurnOnMicrowave"), (r"stove|burner|knob", "TurnOffStove"),
    (r"faucet|water|sink.*(on|off)", "TurnOnSinkFaucet"), (r"kettle", "TurnOnElectricKettle"),
    (r"mug|coffee", "CoffeeSetupMug"), (r"rack|dishwasher", "SlideDishwasherRack"),
    (r"blender|lid", "CloseBlenderLid"), (r"mixer", "OpenStandMixerHead"), (r"navigate|go to", "NavigateKitchen"),
    (r"toaster", "PickPlaceToasterToCounter"), (r"sink", "PickPlaceSinkToCounter"), (r"stove|pan|pot", "PickPlaceCounterToStove"),
    (r"drawer", "PickPlaceDrawerToCounter"), (r".*", "PickPlaceCounterToCabinet"),
]


def atomic_for(name: str, lang: str) -> str:
    s = f"{name} {lang}".lower()
    for pat, task in ATOMIC_FOR:
        if re.search(pat, s):
            return task
    return "PickPlaceCounterToCabinet"


def composite_dir(task: str) -> Path:
    base = paths.data_base()
    hits = sorted((base / "v1.0" / "target" / "composite" / task).glob("*/lerobot"))
    if not hits:
        raise FileNotFoundError(f"no target composite demos for {task} under {base}")
    return hits[-1]


def subtask_segments(ds: Path, ep: int) -> list[dict]:
    import pandas as pd
    with open(ds / "meta" / "tasks.jsonl") as f:
        strings = {d["task_index"]: d["task"] for d in map(json.loads, filter(str.strip, f))}
    df = pd.read_parquet(next(ds.glob(f"data/*/episode_{ep:06d}.parquet")))
    if "annotation.human.subtask" not in df.columns:
        raise KeyError(f"{ds}: no per-frame subtask labels (annotation.human.subtask)")
    lang = df["annotation.human.subtask"].to_numpy()
    name = df["annotation.human.subtask_name"].to_numpy() if "annotation.human.subtask_name" in df else lang
    segs, start = [], 0
    for t in range(1, len(df) + 1):
        if t == len(df) or lang[t] != lang[start]:
            segs.append({"lang": strings.get(int(lang[start]), ""), "name": strings.get(int(name[start]), ""),
                         "start": start, "len": t - start})
            start = t
    return segs


def labeler_stage(labeler, env, t: int, hist: list) -> str:
    try:
        hist.append(labeler.per_step(env, t))
        res = labeler.resolve(hist)
        idx = getattr(res, "subtask_idx", None)
        return str(int(np.asarray(idx)[-1])) if idx is not None and len(np.asarray(idx)) else ""
    except Exception as e:  # labelers expect a full replay; report rather than fail
        return f"labeler_error:{type(e).__name__}"


def run_chain(env, ds: Path, ep: int, ckpt: str, slack: float = 1.5) -> dict:
    from policy.run_skill import run_skill
    obs = sim.reset_to_episode(env, ds, ep)
    segs = subtask_segments(ds, ep)
    labeler = None
    try:
        from robocasa.utils.subtask_labeling import get_labeler
        labeler = get_labeler(type(env).__name__)
        if labeler is not None:
            labeler.setup(env, env.get_ep_meta())
    except Exception:
        labeler = None
    steps, hist, rows, success = 0, [], [], False
    for k, sgm in enumerate(segs):
        at = atomic_for(sgm["name"], sgm["lang"])
        r = run_skill(env, sgm["lang"], max_seconds=slack * sgm["len"] / 20, checkpoint=ckpt, task=at, obs=obs)
        obs, steps = r["obs"], steps + r["steps"]
        success = success or r["grading_only"]["success"]
        stage = labeler_stage(labeler, env, steps, hist) if labeler else ""
        rows.append({"k": k, "subtask": sgm["lang"], "name": sgm["name"], "atomic_stats": at, "steps": r["steps"],
                     "stage_after": stage, "composite_success_so_far": success})
        if success:
            break
    return {"success": int(success), "steps": steps, "n_subtasks": len(segs), "segments_run": len(rows),
            "chain": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--note", default="composite_chain")
    args = ap.parse_args()
    out = paths.REPO / "results" / f"{dt.date.today()}_policy_only_{args.note}.csv"
    cols = ["task", "episode", "seed", "success", "seconds", "llm_calls", "input_tokens", "output_tokens", "cost_usd",
            "code_commit", "checkpoint_step", "format", "run_id", "steps", "n_subtasks", "segments_run", "chain"]
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    if new:
        w.writeheader()
    import time
    for task in args.tasks:
        ds = composite_dir(task)
        env = sim.make_env(ds, cams=["left", "right"])
        for ep in range(args.start, args.start + args.episodes):
            t0 = time.time()
            r = run_chain(env, ds, ep, args.ckpt)
            w.writerow(dict(task=task, episode=ep, seed=-1, success=r["success"], seconds=round(time.time() - t0, 1),
                            llm_calls=0, input_tokens=0, output_tokens=0, cost_usd=0, code_commit=paths.code_commit(),
                            checkpoint_step=Path(args.ckpt).stem.split("_")[-1], format="sentence",
                            run_id=Path(args.ckpt).parent.name, steps=r["steps"], n_subtasks=r["n_subtasks"],
                            segments_run=r["segments_run"], chain=json.dumps(r["chain"])))
            f.flush()
            print(task, ep, r["success"], r["segments_run"], "/", r["n_subtasks"], flush=True)
        env.close()
    f.close()


if __name__ == "__main__":
    main()
