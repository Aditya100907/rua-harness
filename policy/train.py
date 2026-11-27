"""Train DEM on RoboCasa demos.

Recipe from the DEM README / examples/train_minimal.py: AdamW (head 1e-4, vision
5e-5, weight decay 1e-4, betas 0.9/0.95), cosine schedule, gradient clip 1.0,
frozen NeoBERT with cached tokens, fine-tuned DINOv3 ConvNeXt-B under bf16
autocast (CUDA only), head and loss in fp32.

Output folder:
  config.json         every argument, the code commit, the dataset size
  norm_stats.json     per-task quantiles + global z-score (see policy/data.py)
  ckpt_0005000.pt     DEM.save() files (head + vision; frozen NeoBERT is re-downloaded on load)
  train_state.pt      optimizer, scheduler, step, RNG, path of the matching ckpt (for --resume)
  train_log.csv       step, loss, raw_mse, lr_head, steps_per_sec, data_frac, gpu_mem_gb

    python -m policy.train --tasks OpenDrawer --steps 50 --batch 4 --device mps --out ~/apollo-lab/runs/train_test
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
from policy.variants import collate_variant
from policy.data import RoboCasaDataset, collate, load_norm_stats, save_norm_stats, zscore_normalizers


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--all-tasks", action="store_true", help="the 18 atomic-seen tasks")
    ap.add_argument("--cams", nargs="+", default=["left", "right"])
    ap.add_argument("--format", default="sentence", help="command format (policy/model.py)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-vision", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--freeze-vision", action="store_true")
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/train_state.pt if it exists")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--offline", action="store_true", help="random-init towers, no Hub downloads (tests)")
    ap.add_argument("--max-hours", type=float, default=0, help="save and exit after this long (0 = no limit)")
    ap.add_argument("--episodes-json", default=None, help="task -> training episode ids (held-out splits)")
    ap.add_argument("--point-noise-cm", type=float, default=2.0, help="training noise on command points")
    ap.add_argument("--labels-dir", default=None, help="hindsight command labels (policy/label_commands.py)")
    ap.add_argument("--video", action="store_true", help="decode mp4 directly (slow, no frame store needed)")
    ap.add_argument("--done-head", action="store_true", help="also train a progress / done output (DoneHead)")
    ap.add_argument("--dispersive-weight", type=float, default=0.25, help="DEM recipe: 0.25")
    ap.add_argument("--clip-norm", type=float, default=0.0, help="clip stage-1 normalized state/actions to +-this (0 = off)")
    a = ap.parse_args(argv)
    if a.all_tasks:
        a.tasks = list(paths.ATOMIC_SEEN)
    return a


def lr_lambda(step: int, total: int, warmup: int) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def build(args, ds_stats: dict, device: str):
    from policy.model import build_policy
    policy = build_policy(args.format, cams=len(args.cams), offline=args.offline,
                          freeze_vision=args.freeze_vision,
                          dispersive_weight=getattr(args, "dispersive_weight", 0.25))
    a_norm, p_norm = zscore_normalizers(ds_stats)
    policy.action_norm, policy.proprio_norm = a_norm, p_norm
    return policy.to(device)


def save_ckpt(policy, out: Path, step: int, fmt: str = "sentence") -> Path:
    p = out / f"ckpt_{step:07d}.pt"
    tmp = p.with_suffix(".tmp")
    policy.save(tmp)
    ck = torch.load(tmp, map_location="cpu", weights_only=True)
    ck.setdefault("rua_format", fmt)          # every checkpoint names its command format
    torch.save(ck, tmp)
    os.replace(tmp, p)
    return p


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device

    state_path = out / "train_state.pt"
    resuming = args.resume and state_path.exists()
    stats = load_norm_stats(out / "norm_stats.json") if resuming else None
    episodes = json.loads(Path(args.episodes_json).read_text()) if args.episodes_json else None
    from policy.model import wrap_dataset
    ds = RoboCasaDataset(args.tasks, cams=tuple(args.cams), norm_stats=stats, episodes=episodes,
                         use_video=args.video, clip=args.clip_norm)
    ds = wrap_dataset(ds, args)
    stats = ds.norm_stats
    if not resuming:
        save_norm_stats(stats, out / "norm_stats.json")
    cfg = dict(vars(args), code_commit=paths.code_commit(), n_samples=len(ds),
               n_episodes=int(sum(len(t["lens"]) for t in ds.tables)), instructions=len(ds.instructions()))
    (out / ("config_resume.json" if resuming else "config.json")).write_text(json.dumps(cfg, indent=1))

    if resuming:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        from policy.model import load_policy
        policy = load_policy(st["ckpt"], offline=args.offline).to(dev)
        step = int(st["step"])
    else:
        policy = build(args, stats, dev)
        step = 0
    policy.train()

    # Language tokens once per distinct instruction (frozen NeoBERT caches them inside DEM).
    if policy.language.tokenizer is None:               # offline test mode: fixed random ids per string
        g = torch.Generator().manual_seed(0)
        for text in ds.instructions():
            ids = torch.randint(1, 100, (1, 8), generator=g).to(dev)
            tok, mask = policy.language.encode_ids(ids, torch.ones_like(ids))
            policy.cache_instruction(text, tok[0], mask[0])
    else:
        with torch.no_grad():
            policy.encode_language(ds.instructions())

    done_head = None
    groups = policy.parameter_groups(args.lr_head, args.lr_vision)
    if getattr(args, "done_head", False):
        from policy.model import DoneHead
        done_head = DoneHead(policy.vision.embed_dim, 16).to(dev)
        if resuming and Path(str(st["ckpt"]).replace(".pt", ".done.pt")).exists():
            done_head.load_state_dict(torch.load(str(st["ckpt"]).replace(".pt", ".done.pt"), map_location=dev))
        groups.append({"params": list(done_head.parameters()), "lr": args.lr_head})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, args.steps, args.warmup))
    if resuming:
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["torch_rng"])
        if st.get("np_rng") is not None:
            np.random.set_state(st["np_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        print(f"resumed from {st['ckpt']} at step {step}", flush=True)
    trainable = [p for g_ in opt.param_groups for p in g_["params"]]

    gen = torch.Generator().manual_seed(args.seed + step)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
        collate_fn=collate if args.format == "sentence" else collate_variant, generator=gen, persistent_workers=args.workers > 0,
        pin_memory=dev.startswith("cuda"), prefetch_factor=4 if args.workers > 0 else None)
    use_bf16 = dev.startswith("cuda")
    from policy.model import forward_loss

    log_f = open(out / "train_log.csv", "a", newline="")
    log = csv.writer(log_f)
    if log_f.tell() == 0:
        log.writerow(["step", "loss", "raw_mse", "lr_head", "steps_per_sec", "data_frac", "gpu_mem_gb", "wall"])
    t_start = time.time()
    t_win, data_t, n_win = time.time(), 0.0, 0
    losses, mses = [], []
    done = False
    while not done:
        t_d = time.time()
        for batch in dl:
            data_t += time.time() - t_d
            loss, aux = forward_loss(policy, batch, dev, use_bf16, done_head=done_head)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            step += 1
            n_win += 1
            losses.append(loss.item())
            mses.append(float(aux.get("raw_mse", float("nan"))))
            if step % args.log_every == 0 or step == args.steps:
                el = time.time() - t_win
                mem = torch.cuda.max_memory_allocated() / 1e9 if dev.startswith("cuda") else 0
                row = [step, np.mean(losses), np.mean(mses), opt.param_groups[0]["lr"], n_win / el,
                       data_t / el, round(mem, 2), round(time.time() - t_start, 1)]
                log.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in row])
                log_f.flush()
                print(f"step {step} loss {row[1]:.4f} mse {row[2]:.4f} lr {row[3]:.2e} "
                      f"{row[4]:.2f} it/s data {row[5]:.0%} mem {mem:.1f}G", flush=True)
                t_win, data_t, n_win, losses, mses = time.time(), 0.0, 0, [], []
            out_of_time = args.max_hours and (time.time() - t_start) > args.max_hours * 3600
            if step % args.save_every == 0 or step >= args.steps or out_of_time:
                ck = save_ckpt(policy, out, step, args.format)
                if done_head is not None:
                    torch.save(done_head.state_dict(), str(ck).replace(".pt", ".done.pt"))
                torch.save({"step": step, "opt": opt.state_dict(), "sched": sched.state_dict(),
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
