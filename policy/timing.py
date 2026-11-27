"""H1: policy timing. Milliseconds per `act` call (vision + head, language cached) and per robot second.

    python -m policy.timing --ckpt CKPT --device cuda --n 200

The language encoder runs once per command (cached), so it is timed separately. One act call
covers 8 executed steps at 20 Hz = 0.4 s of robot time, so ms per robot second = ms per call * 2.5.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bf16", action="store_true", help="vision under bf16 autocast, as in training")
    a = ap.parse_args()
    from dem import preprocess_images
    from policy.model import load_policy
    pol = load_policy(a.ckpt).to(a.device).eval()
    dev = a.device
    frames = np.random.default_rng(0).integers(0, 255, (2, 256, 256, 3), dtype=np.uint8)
    st = torch.zeros(1, 16, device=dev)
    texts = [f"open the drawer number {i}" for i in range(20)]
    sync(dev); t = time.time()
    for s in texts:
        pol.encode_language(s)
    sync(dev); lang_ms = (time.time() - t) / len(texts) * 1000
    pol.set_instruction(texts[0])
    extra = {}
    if hasattr(pol, "geo"):
        extra = {"points": torch.tensor([[[0.5, 0, 0.9, 1], [0.4, 0.3, 1.2, 1]]], device=dev)} \
            if pol.rua_format != "pose_path" else {"path": torch.zeros(1, 8, 7, device=dev)}

    def call():
        img = preprocess_images(frames, size=256, device=dev)[None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16 and dev.startswith("cuda")):
            vis = pol.encode_vision(img)
        return pol.act(None, st, vis_tokens=vis.float(), **extra)

    for _ in range(20):
        call()
    times = []
    for _ in range(a.n):
        sync(dev); t = time.time(); call(); sync(dev)
        times.append((time.time() - t) * 1000)
    times = np.array(times)
    name = torch.cuda.get_device_name(0) if dev.startswith("cuda") else dev
    out = {"device": name, "bf16_vision": a.bf16, "act_ms_median": round(float(np.median(times)), 2),
           "act_ms_p90": round(float(np.percentile(times, 90)), 2), "lang_ms_per_new_command": round(lang_ms, 2),
           "ms_per_robot_second": round(float(np.median(times)) * 2.5, 2), "n": a.n,
           "format": getattr(pol, "rua_format", "sentence")}
    print("TIMING", json.dumps(out))


if __name__ == "__main__":
    main()
