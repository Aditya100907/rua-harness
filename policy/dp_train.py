"""Train the Diffusion Policy baseline (policy/diffusion_policy.py) on exactly the DEM training samples.

Reached through `python -m policy.train --format diffusion_policy ...`, so policy/slurm/train.sbatch, the
watcher's resubmit and its scoring all work unchanged. Same output folder as policy/train.py:
  config.json, norm_stats.json, ckpt_XXXXXXX.pt (EMA weights, rua_format "diffusion_policy"),
  train_state.pt (raw weights, EMA, optimizer, scheduler, step, RNG), train_log.csv.

Optimization follows LeRobot's DiffusionConfig presets, which are the original DP recipe: AdamW lr 1e-4,
betas (0.95, 0.999), weight decay 1e-6, cosine schedule with 500 warmup steps, gradient clip (LeRobot's
train script default 10), EMA of the weights (original DP; LeRobot has none), fp32.

    python -m policy.train --format diffusion_policy --tasks OpenDrawer --steps 20 --batch 4 --video \
        --episodes-json od2.json --out ~/apollo-lab/runs/dp_test --dp-down-dims 64 128 256
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from policy import paths
from policy.data import RoboCasaDataset, collate, load_norm_stats, save_norm_stats
from policy.diffusion_policy import EMA, FORMAT, DPPolicy, lerobot_version


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default=FORMAT, choices=[FORMAT])
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--all-tasks", action="store_true")
    ap.add_argument("--cams", nargs="+", default=["left", "wrist"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--episodes-json", default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--quantiles", default="recompute", choices=["recompute", "dataset"])
    ap.add_argument("--clip-norm", type=float, default=1.0,
                    help="clip stage-1 state/actions to +-this; 1.0 so targets fit DP's sample clipping")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--dp-scheduler", default="DDIM", choices=["DDPM", "DDIM"])
    ap.add_argument("--dp-train-timesteps", type=int, default=100)
    ap.add_argument("--dp-infer-steps", type=int, default=10)
    ap.add_argument("--dp-resize", type=int, default=128)
    ap.add_argument("--dp-crop-ratio", type=float, default=0.9)
    ap.add_argument("--dp-down-dims", type=int, nargs="+", default=[512, 1024, 2048])
    a = ap.parse_args(argv)
    if a.all_tasks:
        a.tasks = list(paths.ATOMIC_SEEN)
    return a


def lr_lambda(step: int, total: int, warmup: int) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def build(args) -> DPPolicy:
    return DPPolicy(args.tasks, args.cams, scheduler=args.dp_scheduler, train_timesteps=args.dp_train_timesteps,
                    infer_steps=args.dp_infer_steps, resize=args.dp_resize, crop_ratio=args.dp_crop_ratio,
                    down_dims=tuple(args.dp_down_dims), clip_sample_range=args.clip_norm or 1.0)


def save_ckpt(model: DPPolicy, out: Path, step: int) -> Path:
    p = out / f"ckpt_{step:07d}.pt"
    tmp = p.with_suffix(".tmp")
    model.save(tmp)
    os.replace(tmp, p)
    return p


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device
    if dev.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    state_path = out / "train_state.pt"
    resuming = args.resume and state_path.exists()
    stats = load_norm_stats(out / "norm_stats.json") if resuming else None
    episodes = json.loads(Path(args.episodes_json).read_text()) if args.episodes_json else None
    ds = RoboCasaDataset(args.tasks, cams=tuple(args.cams), norm_stats=stats, episodes=episodes,
                         use_video=args.video, clip=args.clip_norm, quantiles=args.quantiles)
    if not resuming:
        save_norm_stats(ds.norm_stats, out / "norm_stats.json")
    model = build(args).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    cfg = dict(vars(args), code_commit=paths.code_commit(), n_samples=len(ds), world_size=1,
               global_batch=args.batch, n_episodes=int(sum(len(t["lens"]) for t in ds.tables)),
               n_train_episodes=int(len(np.unique(ds.index[:, :2], axis=0))), instructions=len(ds.instructions()),
               objective="diffusion", task_conditioning="one-hot task id appended to the state",
               lerobot_version=lerobot_version(), dp_settings=model.settings, n_params=n_params)
    (out / ("config_resume.json" if resuming else "config.json")).write_text(json.dumps(cfg, indent=1))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, args.steps, args.warmup))
    ema = None if args.no_ema else EMA(model)
    step = 0
    if resuming:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        if ema is not None and st.get("ema") is not None:
            ema.model.load_state_dict(st["ema"])
            ema.step = int(st["ema_step"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["np_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        step = int(st["step"])
        print(f"resumed at step {step}", flush=True)
    if ema is not None:
        ema.model.to(dev)
    model.train()

    gen = torch.Generator().manual_seed(args.seed + step)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                                     num_workers=args.workers, collate_fn=collate, generator=gen,
                                     persistent_workers=args.workers > 0, pin_memory=dev.startswith("cuda"),
                                     prefetch_factor=4 if args.workers > 0 else None)
    log_f = open(out / "train_log.csv", "a", newline="")
    log = csv.writer(log_f)
    if log_f.tell() == 0:
        log.writerow(["step", "loss", "raw_mse", "lr_head", "steps_per_sec", "data_frac", "gpu_mem_gb", "wall"])
    t_start = time.time()
    t_win, data_t, n_win, losses = time.time(), 0.0, 0, []
    done = False
    while not done:
        t_d = time.time()
        for batch in dl:
            data_t += time.time() - t_d
            loss = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update(model)
            step += 1
            n_win += 1
            losses.append(loss.item())
            if step % args.log_every == 0 or step == args.steps:
                el = time.time() - t_win
                mem = torch.cuda.max_memory_allocated() / 1e9 if dev.startswith("cuda") else 0
                row = [step, np.mean(losses), np.mean(losses), opt.param_groups[0]["lr"], n_win / el,
                       data_t / el, round(mem, 2), round(time.time() - t_start, 1)]
                log.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
                log_f.flush()
                print(f"step {step} loss {row[1]:.4f} lr {row[3]:.2e} {row[4]:.2f} it/s data {row[5]:.0%} "
                      f"mem {mem:.1f}G", flush=True)
                t_win, data_t, n_win, losses = time.time(), 0.0, 0, []
            out_of_time = bool(args.max_hours and (time.time() - t_start) > args.max_hours * 3600)
            if step % args.save_every == 0 or step >= args.steps or out_of_time:
                ck = save_ckpt(ema.model if ema is not None else model, out, step)
                torch.save({"step": step, "model": model.state_dict(),
                            "ema": ema.model.state_dict() if ema is not None else None,
                            "ema_step": ema.step if ema is not None else 0,
                            "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            "ckpt": str(ck)}, state_path.with_suffix(".tmp"))
                os.replace(state_path.with_suffix(".tmp"), state_path)
                print(f"saved {ck}", flush=True)
            if step >= args.steps or out_of_time:
                done = True
                break
            t_d = time.time()
    log_f.close()
    print("TRAIN DONE" if step >= args.steps else f"TRAIN STOPPED at {step} (time limit)", flush=True)


if __name__ == "__main__":
    main()
