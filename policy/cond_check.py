"""Part K: does the policy's output depend on its inputs? (fixed noise across inputs vs fixed input across noise)

    python -m policy.cond_check --ckpt CKPT --task OpenDrawer
Prints: spread of predictions across 32 different frames with the same noise (input effect), spread across
32 noise draws on the same frame (noise effect), MAE to the recorded actions with z0 = 0 and with random z0,
and the same after zeroing the images / the state / the language to see which input the head uses.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="OpenDrawer")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--nfe", type=int, default=1)
    a = ap.parse_args()
    from dem import preprocess_images
    from policy.data import RoboCasaDataset, load_norm_stats, task_quantiles
    from policy.model import load_policy
    ck = Path(a.ckpt)
    stats = load_norm_stats(ck.parent / "norm_stats.json")
    cfg = json.loads((ck.parent / "config.json").read_text())
    pol = load_policy(a.ckpt).eval()
    ds = RoboCasaDataset([a.task], cams=tuple(cfg.get("cams", ["left", "right"])), norm_stats=stats)
    _, qa = task_quantiles(stats, a.task)
    idx = np.random.default_rng(1).choice(len(ds), a.n, replace=False)
    S = [ds[int(i)] for i in idx]
    imgs = torch.stack([preprocess_images(s["images"], size=256) for s in S])
    st = torch.as_tensor(np.stack([s["state"] for s in S]))
    texts = [s["text"] for s in S]
    true = np.stack([ds.raw(int(i))["actions"] for i in idx])
    torch.manual_seed(0)
    z_fixed = torch.randn(1, 16, 12).expand(a.n, -1, -1).contiguous()

    def run(img, s, txt, z0):
        with torch.no_grad():
            vis = pol.encode_vision(img)
            out = pol.act(None, s, texts=txt, vis_tokens=vis, z0=z0, nfe=a.nfe)
        return qa.unnormalize(out.float().numpy())

    def mae(p):
        return float(np.abs(p - true)[:, :8, 5:11].mean())
    p_fixed = run(imgs, st, texts, z_fixed)
    p_zero = run(imgs, st, texts, torch.zeros(a.n, 16, 12))
    p_rand = run(imgs, st, texts, torch.randn(a.n, 16, 12))
    one = run(imgs[:1].expand(a.n, -1, -1, -1, -1), st[:1].expand(a.n, -1), [texts[0]] * a.n, torch.randn(a.n, 16, 12))
    out = {
        "input_spread_fixed_noise": float(p_fixed[:, :8, 5:11].std(0).mean()),
        "noise_spread_fixed_input": float(one[:, :8, 5:11].std(0).mean()),
        "true_spread_across_inputs": float(true[:, :8, 5:11].std(0).mean()),
        "mae_z0_zero": mae(p_zero), "mae_z0_random": mae(p_rand),
        "mae_mean_action": float(np.abs(true[:, :8, 5:11].mean(0) - true[:, :8, 5:11]).mean()),
        "mae_zero_images": mae(run(torch.zeros_like(imgs), st, texts, torch.zeros(a.n, 16, 12))),
        "mae_zero_state": mae(run(imgs, torch.zeros_like(st), texts, torch.zeros(a.n, 16, 12))),
        "mae_other_text": mae(run(imgs, st, ["Close the fridge."] * a.n, torch.zeros(a.n, 16, 12))),
    }
    print("CONDCHECK", json.dumps({k: round(v, 4) for k, v in out.items()}), flush=True)


if __name__ == "__main__":
    main()
