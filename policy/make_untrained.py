"""Save an untrained checkpoint (pretrained towers, random head) with real norm stats, for the zero test.

    python -m policy.make_untrained --tasks OpenDrawer TurnOnMicrowave PickPlaceCounterToCabinet --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from policy import paths
from policy.data import RoboCasaDataset, save_norm_stats
from policy.train import build, save_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--cams", nargs="+", default=["left", "right"])
    ap.add_argument("--format", default="sentence")
    ap.add_argument("--freeze-vision", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = RoboCasaDataset(args.tasks, cams=tuple(args.cams))
    save_norm_stats(ds.norm_stats, out / "norm_stats.json")
    (out / "config.json").write_text(json.dumps(dict(vars(args), untrained=True,
                                                     code_commit=paths.code_commit()), indent=1))
    policy = build(args, ds.norm_stats, "cpu")
    print("saved", save_ckpt(policy, out, 0))


if __name__ == "__main__":
    main()
