"""Evaluate a DEM checkpoint on LIBERO, mirroring `policy.eval`.

Standard LIBERO protocol: each task's 50 fixed benchmark init states (`get_task_init_states`), episode i
uses init state i; 10 no-op steps first so objects settle; then each control cycle feeds agentview +
wrist (rendered at 128 as in the demos, flipped upright, resized to 256) and the 8-number state,
predicts 16 actions and executes the first 8, until success or the suite's step limit
(`sim.MAX_STEPS`). One CSV row per episode with the columns of results/README.md.

    python -m policy.libero.eval --ckpt CK --suite libero_spatial --episodes 50
    python -m policy.libero.eval --wiring --suite libero_spatial --tasks 0 --episodes 5   # demo actions, no policy
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

from policy import paths
from policy.libero import sim

COLS = ["task", "episode", "seed", "success", "seconds", "llm_calls", "input_tokens", "output_tokens",
        "cost_usd", "code_commit", "checkpoint_step", "format", "run_id", "steps", "instruction"]


class LiberoPolicyRunner:
    """Checkpoint + norm_stats: raw LIBERO obs in, raw 7-number actions out."""

    def __init__(self, ckpt: str, device: str = "cuda", nfe: int = 1):
        import torch
        from policy.data import load_norm_stats
        from policy.model import load_policy
        self.torch, self.device, self.nfe = torch, device, nfe
        self.policy = load_policy(ckpt).to(device).eval()
        d = Path(ckpt).parent
        self.stats = load_norm_stats(d / "norm_stats.json")
        cfg_p = d / "config.json"
        self.cfg = json.loads(cfg_p.read_text()) if cfg_p.exists() else {}
        self.cams = list(self.cfg.get("cams") or sim.DEFAULT_CAMS)
        c = self.policy.config
        assert (c.action_dim, c.proprio_dim) == (sim.ACTION_DIM, sim.STATE_DIM), \
            f"checkpoint has action/proprio dims {(c.action_dim, c.proprio_dim)}, LIBERO needs (7, 8)"
        self.quant = None

    def set_task(self, task: str, instruction: str) -> None:
        from policy.data import task_quantiles
        if task not in self.stats["per_task"]:
            raise KeyError(f"{task} was not in this checkpoint's training tasks")
        self.quant = task_quantiles(self.stats, task)
        self.policy.set_instruction(instruction)

    def act(self, obs: dict) -> np.ndarray:
        torch = self.torch
        from dem import preprocess_images
        qs, qa = self.quant
        imgs = preprocess_images(sim.images(obs, self.cams), size=256, device=self.device)[None]
        st = torch.as_tensor(qs.normalize(sim.state8(obs)), dtype=torch.float32, device=self.device)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            vis = self.policy.encode_vision(imgs)
            a = self.policy.act(None, st, vis_tokens=vis.float(), nfe=self.nfe)
        return qa.unnormalize(a[0].float().cpu().numpy())        # [16, 7]


def run_episode(env, init_state, act_fn, horizon: int, execute: int = 8, video=None,
                wait: int = sim.WAIT_STEPS) -> dict:
    """act_fn(obs, t) -> [H, 7] raw actions (t counts policy steps, after the wait). Returns a result dict."""
    t0 = time.time()
    obs = init_state() if callable(init_state) else sim.reset_to_state(env, init_state)
    for _ in range(wait):
        obs, _, _, _ = env.step(sim.NOOP)
    success, t, model_s = False, 0, 0.0
    while t < horizon and not success:
        tm = time.time()
        chunk = act_fn(obs, t)
        model_s += time.time() - tm
        if chunk is None or len(chunk) == 0:
            break
        for a in np.asarray(chunk, np.float64)[:execute]:
            a = a.copy()
            a[6] = 1.0 if a[6] > 0 else -1.0          # gripper is binary in the demos
            obs, _, _, _ = env.step(np.clip(a, -1, 1))
            t += 1
            if video is not None:
                video.append(np.concatenate(list(sim.images(obs, size=None)), axis=1))
            if env.check_success():
                success = True
                break
            if t >= horizon:
                break
    return {"success": int(success), "steps": t, "seconds": round(time.time() - t0, 2),
            "model_seconds": round(model_s, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", nargs="+", type=int, default=None, help="task ids (default: all in the suite)")
    ap.add_argument("--episodes", type=int, default=50, help="per task; init state i for episode i")
    ap.add_argument("--execute", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--wiring", action="store_true", help="play demo i's actions through this loop (no policy)")
    ap.add_argument("--video", type=int, default=2)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    runner, step, run_id, fmt = None, "", "", "sentence"
    if not args.wiring:
        runner = LiberoPolicyRunner(args.ckpt, device=dev)
        ck = Path(args.ckpt)
        step = int(ck.stem.split("_")[-1]) if ck.stem.startswith("ckpt_") else ""
        run_id = ck.parent.name
    note = args.note or ("wiring" if args.wiring else f"{run_id}_{step}")
    out = Path(args.out) if args.out else paths.REPO / "results" / f"{dt.date.today()}_libero_{args.suite}_{note}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = paths.code_commit()
    new = not out.exists()
    f = open(out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=COLS + ["suite", "model_seconds"], extrasaction="ignore")
    if new:
        w.writeheader()
    tids = args.tasks if args.tasks is not None else list(range(sim.n_tasks(args.suite)))
    H = sim.MAX_STEPS[args.suite]
    for tid in tids:
        info = sim.task_info(args.suite, tid)
        key = f"{args.suite}/{info['name']}"
        env = sim.make_env(info["bddl"], size=sim.RAW_SIZE)
        if runner:
            runner.set_task(key, info["language"])
        n_ok = 0
        for i in range(args.episodes):
            frames = [] if i < args.video else None
            torch.manual_seed(i)
            if args.wiring:
                import h5py
                with h5py.File(info["demo"], "r") as h:
                    d = h["data"][sim.demo_keys(h)[i]]
                    acts = d["actions"][()]
                    # the demo's own scene XML and first state; no settle wait, the actions start there
                    r = run_episode(env, lambda: sim.reset_to_demo(env, d), lambda o, t: acts[t:t + 16],
                                    len(acts) + 50, args.execute, frames, wait=0)
            else:
                init = info["init_states"][i % len(info["init_states"])]
                r = run_episode(env, init, lambda o, t: runner.act(o), H, args.execute, frames)
            n_ok += r["success"]
            row = dict(task=key, episode=i, seed=i, success=r["success"], seconds=r["seconds"], llm_calls=0,
                       input_tokens=0, output_tokens=0, cost_usd=0, code_commit=commit, checkpoint_step=step,
                       format=fmt, run_id=run_id or "wiring", steps=r["steps"], instruction=info["language"],
                       suite=args.suite, model_seconds=r["model_seconds"])
            w.writerow(row)
            f.flush()
            print(json.dumps({k: row[k] for k in ("task", "episode", "success", "steps", "seconds")}), flush=True)
            if frames:
                import imageio
                vd = out.with_suffix("")
                vd.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(vd / f"task{tid:02d}_ep{i:03d}_{'ok' if r['success'] else 'fail'}.mp4", frames, fps=20)
        env.close()
        print(f"EVAL {key}: {n_ok}/{args.episodes} succeeded", flush=True)
    f.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
