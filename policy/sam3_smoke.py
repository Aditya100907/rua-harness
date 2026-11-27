"""Smoke test for the SAM 3 service (HexaShell's `services/sam3_server.py`): segment one RoboCasa frame.

Takes one frame from our frame store (`$RUA_FRAMES/<task>/<camera>.bin`, JPEG, see
`policy/extract_frames.py`), sends it to the service with a text prompt, and prints the masks it
returns (score, box, area, fraction of the picture). Stdlib, numpy and PIL only, so it runs in the
sam3 env or the dem env.

    python -m policy.sam3_smoke --url http://127.0.0.1:18300 --task TurnOnMicrowave --text microwave
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


def post(url: str, path: str, body: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def frame_jpeg(frames: Path, task: str, cam: str, i: int) -> bytes:
    idx = np.load(frames / task / f"{cam}.idx.npy")
    with open(frames / task / f"{cam}.bin", "rb") as f:
        f.seek(int(idx[i]))
        return f.read(int(idx[i + 1] - idx[i]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18300")
    ap.add_argument("--frames", default=os.environ.get("RUA_FRAMES", "/nfs/roberts/scratch/pi_dr877/ak3288/frames"))
    ap.add_argument("--task", default="TurnOnMicrowave")
    ap.add_argument("--camera", default="left")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--text", nargs="+", default=["microwave", "robot gripper"])
    ap.add_argument("--save", default=None, help="folder for the frame and mask PNGs")
    args = ap.parse_args()

    jpg = frame_jpeg(Path(args.frames), args.task, args.camera, args.frame)
    img = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
    print(f"frame: {args.task} {args.camera} #{args.frame}, {img.shape[1]}x{img.shape[0]}")
    with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=10) as r:
        print("health:", r.read().decode()[:300])
    up = post(args.url, "/image", {"jpeg_b64": base64.b64encode(jpg).decode()})
    print("image:", up)
    ok = True
    for text in args.text:
        r = post(args.url, "/segment", {"image_id": up["image_id"], "text": text, "max_masks": 8})
        if "error" in r:
            print(f"{text!r}: error {r['error']}")
            ok = False
            continue
        print(f"{text!r}: {len(r['masks'])} masks in {r.get('ms')} ms")
        for k, m in enumerate(r["masks"]):
            mask = np.asarray(Image.open(io.BytesIO(base64.b64decode(m["png_b64"]))).convert("L")) > 0
            print(f"  mask {k}: score {m['score']:.3f} box {m['box']} area {m['area']} "
                  f"({mask.mean():.3%} of the picture, decoded {int(mask.sum())} px)")
            if args.save:
                out = Path(args.save)
                out.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(out / "frame.png")
                Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{text.replace(' ', '_')}_{k}.png")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
