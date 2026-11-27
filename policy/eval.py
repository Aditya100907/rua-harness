"""Evaluate a DEM checkpoint on RoboCasa target-split episodes.

The environment comes from `policy.sim.make_env` (the dataset's own env kwargs,
as in the replay test), reset with a per-episode seed, so kitchens come from the
target layouts 1..10 and the target object split. Each control cycle feeds the
left and right agent views (flipped to dataset orientation) and the 16-number
state, predicts 16 actions and executes the first 8, until success or the task's
horizon from `dataset_registry.py`.

One CSV row per episode in `results/` (columns of results/README.md, plus
checkpoint_step, format, run_id, steps, instruction).

    python -m policy.eval --ckpt runs/x/ckpt_0020000.pt --tasks OpenDrawer --episodes 50
    python -m policy.eval --wiring --tasks OpenDrawer --episodes 10      # D1 test 1: demo actions, no policy
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from policy import paths, sim

COLS = ["task", "episode", "seed", "success", "seconds", "llm_calls", "input_tokens", "output_tokens",
        "cost_usd", "code_commit", "checkpoint_step", "format", "run_id", "steps", "instruction"]


class PolicyRunner:
    """Wraps a checkpoint + norm_stats: raw obs in, raw LeRobot-order actions out."""

    def __init__(self, ckpt: str, device: str = "cuda", cams=("left", "right"), nfe: int = 1):
        import torch
        from policy.data import load_norm_stats
        from policy.model import load_policy
        self.torch, self.device, self.cams, self.nfe = torch, device, list(cams), nfe
        self.policy = load_policy(ckpt).to(device).eval()
        ck_dir = Path(ckpt).parent
        self.stats = load_norm_stats(ck_dir / "norm_stats.json")
        cfg_p = ck_dir / "config.json"
        self.cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        if self.cfg.get("cams"):
            self.cams = list(self.cfg["cams"])
        self.format = self.cfg.get("format", "sentence")
        self.quant = None

    def set_task(self, task: str, instruction: str):
        from policy.data import task_quantiles
        if task not in self.stats["per_task"]:
            raise KeyError(f"{task} was not in this checkpoint's training tasks")
        self.quant = task_quantiles(self.stats, task)
        self.policy.set_instruction(instruction)

    def act(self, obs: dict, extra: dict | None = None) -> np.ndarray:
        torch = self.torch
        from dem import preprocess_images
        qs, qa = self.quant
        imgs = preprocess_images(sim.images(obs, self.cams), size=256, device=self.device)[None]
        st = torch.as_tensor(qs.normalize(sim.state16(obs)), dtype=torch.float32, device=self.device)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            vis = self.policy.encode_vision(imgs)
        if extra:
            a = self.policy.act(None, st, vis_tokens=vis.float(), nfe=self.nfe, **extra)
        else:
            a = self.policy.act(None, st, vis_tokens=vis.float(), nfe=self.nfe)
        return qa.unnormalize(a[0].float().cpu().numpy())          # [16, 12] raw LeRobot order


def run_episode(env, task: str, seed: int, act_fn, horizon: int, execute: int = 8, video=None,
                reset_obs: dict | None = None, record: bool = False) -> dict:
    """act_fn(obs, t) -> [H, 12] raw LeRobot actions. Returns a result dict."""
    t0 = time.time()
    if reset_obs is None:
        env.rng = np.random.default_rng(seed)
        obs = env.reset()
    else:
        obs = reset_obs
    traj = {"eef": [], "grip_cmd": [], "obj": []} if record else None
    success, t, model_s = False, 0, 0.0
    while t < horizon and not success:
        tm = time.time()
        chunk = act_fn(obs, t)
        model_s += time.time() - tm
        if chunk is None or len(chunk) == 0:
            break
        for a in sim.lerobot_to_env_action(np.asarray(chunk)[:execute]):
            obs, _, _, _ = env.step(a)
            t += 1
            if video is not None:
                video.append(np.concatenate(list(sim.images(obs, ["left", "right"])), axis=1))
            if record:
                traj["eef"].append(np.asarray(obs.get("robot0_eef_pos", np.zeros(3))))
                traj["grip_cmd"].append(a[6])
                o = env.objects.get("obj") if hasattr(env, "objects") else None
                traj["obj"].append(np.asarray(env.sim.data.body_xpos[env.obj_body_id["obj"]])
                                   if o is not None and "obj" in getattr(env, "obj_body_id", {}) else np.zeros(3))
            if env._check_success():
                success = True
                break
            if t >= horizon:
                break
    out = {"success": int(success), "steps": t, "seconds": round(time.time() - t0, 2),
           "model_seconds": round(model_s, 2)}
    if record:
        out["traj"] = {k: np.asarray(v) for k, v in traj.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=1000, help="episode i uses seed seed0 + i")
    ap.add_argument("--execute", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--wiring", action="store_true", help="D1 test 1: play demo actions through this loop")
    ap.add_argument("--video", type=int, default=2, help="save videos of the first N episodes per task")
    ap.add_argument("--save-traj", action="store_true", help="save per-episode trajectories (failure sorting)")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=None, help="results CSV (default results/<date>_policy_only_<note>.csv)")
    ap.add_argument("--outdir", default=None, help="videos and trajectories")
    ap.add_argument("--horizon", type=int, default=0, help="override the task horizon")
    ap.add_argument("--shard", type=int, default=0, help="run episodes shard, shard + nshards, ...")
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    runner, step, run_id, fmt = None, "", "", "sentence"
    if not args.wiring:
        runner = PolicyRunner(args.ckpt, device=dev)
        fmt = runner.format
        ck = Path(args.ckpt)
        step = int(ck.stem.split("_")[-1]) if ck.stem.startswith("ckpt_") else ""
        run_id = ck.parent.name
    note = args.note or ("wiring" if args.wiring else f"{run_id}_{step}")
    out = Path(args.out) if args.out else paths.REPO / "results" / f"{dt.date.today()}_policy_only_{note}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    outdir = Path(args.outdir) if args.outdir else out.with_suffix("")
    commit = paths.code_commit()
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS + ["model_seconds", "state_err0", "image_err0"], extrasaction="ignore")
    if new:
        w.writeheader()
    cams = runner.cams if runner else ["left", "right"]
    for task in args.tasks:
        ds = paths.task_dir(task)
        H = args.horizon or paths.horizon(task)
        env = sim.make_env(ds, cams=sorted(set(cams) | {"left", "right"}), size=256)
        n_ok = 0
        for i in range(args.shard, args.episodes, args.nshards):
            seed = args.seed0 + i
            frames = [] if i < args.video else None
            extra_row = {}
            if args.wiring:
                # demo i: reset to its saved state, check the obs pipeline against the recording,
                # then play its actions through the same chunked loop
                obs0 = sim.reset_to_episode(env, ds, i)
                acts = sim.episode_actions(ds, i)
                from policy.data import load_task_table  # noqa: F401  (dataset state for comparison)
                import pandas as pd
                pq = next(ds.glob(f"data/*/episode_{i:06d}.parquet"))
                rec_state0 = np.asarray(pd.read_parquet(pq, columns=["observation.state"]).iloc[0, 0])
                extra_row["state_err0"] = float(np.abs(sim.state16(obs0) - rec_state0).max())
                from policy.extract_frames import decode_video, video_path
                v0 = decode_video(video_path(ds, "left", i))[0].astype(int)
                extra_row["image_err0"] = float(np.abs(sim.images(obs0, ["left"])[0].astype(int) - v0).mean())
                instr = sim.episode_extras(ds, i)[2].get("lang", "")
                r = run_episode(env, task, seed, lambda o, t: acts[t:t + 16], len(acts) + 50,
                                execute=args.execute, video=frames, reset_obs=obs0, record=args.save_traj)
            else:
                env.rng = np.random.default_rng(seed)
                obs0 = env.reset()
                instr = env.get_ep_meta().get("lang", "")
                runner.set_task(task, instr)
                r = run_episode(env, task, seed, lambda o, t: runner.act(o), H, execute=args.execute,
                                video=frames, reset_obs=obs0, record=args.save_traj)
            n_ok += r["success"]
            row = dict(task=task, episode=i, seed=seed, success=r["success"], seconds=r["seconds"], llm_calls=0,
                       input_tokens=0, output_tokens=0, cost_usd=0, code_commit=commit, checkpoint_step=step,
                       format=fmt, run_id=run_id or "wiring", steps=r["steps"], instruction=instr,
                       model_seconds=r["model_seconds"], **extra_row)
            w.writerow(row)
            f.flush()
            print(json.dumps({k: row[k] for k in ("task", "episode", "success", "steps", "seconds", "instruction")}
                             | extra_row), flush=True)
            if frames:
                import imageio
                outdir.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(outdir / f"{task}_ep{i:03d}_{'ok' if r['success'] else 'fail'}.mp4", frames, fps=20)
            if args.save_traj and "traj" in r:
                outdir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(outdir / f"{task}_ep{i:03d}_traj.npz", success=r["success"],
                                    final_state=env.sim.get_state().flatten(), **r["traj"])
        env.close()
        n_run = len(range(args.shard, args.episodes, args.nshards))
        print(f"EVAL {task}: {n_ok}/{n_run} succeeded", flush=True)
    f.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
