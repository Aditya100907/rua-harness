"""Part K diagnostic: does a checkpoint predict the demo's own actions on training frames?

For N random training samples: run the checkpoint exactly as eval does (stage-1 state from the frame,
images from the frame store, instruction), unnormalize to raw actions, and compare with the recorded
16-step chunk. Baselines: the per-task mean action and "repeat the current action".
If the model beats the baselines clearly here but fails in the simulator, the problem is at eval time
(inputs differ from training). If it does not, the problem is in training or in how act() is used.

    python -m policy.openloop_check --ckpt CKPT --task OpenDrawer --n 200 --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="OpenDrawer")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nfe", type=int, default=1)
    ap.add_argument("--episodes", default=None, help="JSON task -> episode ids to sample frames from")
    a = ap.parse_args()
    from dem import preprocess_images
    from policy.data import RoboCasaDataset, load_norm_stats, task_quantiles
    from policy.model import load_policy
    ck = Path(a.ckpt)
    stats = load_norm_stats(ck.parent / "norm_stats.json")
    cfg = json.loads((ck.parent / "config.json").read_text())
    pol = load_policy(a.ckpt).to(a.device).eval()
    eps = json.loads(Path(a.episodes).read_text()) if a.episodes else None
    ds = RoboCasaDataset([a.task], cams=tuple(cfg.get("cams", ["left", "right"])), norm_stats=stats, episodes=eps)
    qs, qa = task_quantiles(stats, a.task)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), min(a.n, len(ds)), replace=False)
    all_raw = ds.tables[0]["actions"]
    mean_a = all_raw.mean(0)
    err_m, err_mean, err_rep, err_m1 = [], [], [], []
    per_dim = []
    torch.manual_seed(0)
    for i in idx:
        s = ds[int(i)]
        raw = ds.raw(int(i))
        true = raw["actions"]                                      # [16, 12] raw
        img = preprocess_images(s["images"], size=256, device=a.device)[None]
        st = torch.as_tensor(s["state"], device=a.device)[None]  # stage-1 normalized, as eval passes
        pol.set_instruction(s["text"])
        with torch.no_grad():
            pred = pol.act(img, st, nfe=a.nfe)[0].float().cpu().numpy()
        pred = qa.unnormalize(pred)
        err_m.append(np.abs(pred - true)[:8, 5:11].mean())        # arm deltas, executed part
        err_m1.append(np.abs(pred[0] - true[0])[5:11].mean())
        err_mean.append(np.abs(mean_a - true)[:8, 5:11].mean())
        err_rep.append(np.abs(true[0][None] - true)[:8, 5:11].mean())
        per_dim.append(np.abs(pred - true)[:8].mean(0))
        grip_ok = np.mean(np.sign(pred[:8, 11]) == np.sign(true[:8, 11]))
        per_dim[-1] = np.append(per_dim[-1], grip_ok)
    pd_ = np.mean(per_dim, 0)
    out = {"ckpt": str(ck), "task": a.task, "n": a.n,
           "model_mae_arm_first8": round(float(np.mean(err_m)), 4),
           "model_mae_arm_step0": round(float(np.mean(err_m1)), 4),
           "mean_action_mae": round(float(np.mean(err_mean)), 4),
           "repeat_current_mae": round(float(np.mean(err_rep)), 4),
           "per_dim_mae": [round(float(x), 3) for x in pd_[:12]], "gripper_sign_agree": round(float(pd_[12]), 3)}
    print("OPENLOOP", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
