"""Turn the DEM paper's research files into one of our checkpoint folders (priority 5, Nov 27).

Research files (per the DEM README, `DEM.load_research_checkpoints`): a head state_dict (`mf_final.pt`,
or `uf_final.pt` with a "head" entry), vision encoder weights, `norm.pt` (a_mean, a_std, p_mean, p_std:
the z-score stage) and the per-task quantiles (`norm_stats.json` with "action"/"state" q01/q99). Heads
trained in that code may use mmBERT-small text tokens (384-d): pass `--lang mmbert`.

The output folder looks like a run of ours, so `policy/eval.py`, `run_skill`, the commander and
`policy/train.py --init-from` work on it unchanged:
  ckpt_0000000.pt   DEM.save() with only the z-score stage in the policy's normalizers
  norm_stats.json   our schema: per-task q01/q99 (stage 1) + the research z-score (stage 2)
  config.json       format, cameras, tasks, language backend, source files

Quantiles: by default each task's from `--norm-stats` if given (one file for all tasks, or a folder with
<task>.json), else the dataset's shipped meta/stats.json (identical to recomputed ones on our data).

    python -m policy.research_ckpt --head mf_final.pt --vision tower.pt --norm norm.pt \
        [--norm-stats norm_stats.json | DIR] [--lang neobert|mmbert] --cams left wrist --out DIR
    python -m policy.research_ckpt --fake --lang mmbert --out DIR      # random-init stand-ins, for testing
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from policy import paths

LANG = {"neobert": dict(language_backend="neobert", language_model="chandar-lab/NeoBERT"),
        "mmbert": dict(language_backend="hf", language_model="jhu-clsp/mmBERT-small")}


def make_config(lang: str, cams: int):
    from dem import DEMConfig
    return DEMConfig(action_dim=12, proprio_dim=16, chunk=16, num_cameras=cams, **LANG[lang])


def task_quantiles(task: str, norm_stats: str | None) -> dict:
    if norm_stats:
        p = Path(norm_stats)
        f = p / f"{task}.json" if p.is_dir() else p
        st = json.loads(f.read_text())
        if task in st:                        # a file keyed by task
            st = st[task]
        return {"action": {k: list(map(float, st["action"][k])) for k in ("q01", "q99")},
                "state": {k: list(map(float, st["state"][k])) for k in ("q01", "q99")}}
    from policy.data import shipped_quantiles
    (s01, s99), (a01, a99) = shipped_quantiles(task)
    return {"state": {"q01": s01.tolist(), "q99": s99.tolist()}, "action": {"q01": a01.tolist(), "q99": a99.tolist()}}


def write_fake(d: Path, lang: str, cams: int) -> dict:
    """Random-init stand-ins in the research layouts, to test the loader before the real files arrive."""
    from dem import DEM
    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    ref = DEM(make_config(lang, cams))
    torch.save(ref.head.state_dict(), d / "mf_final.pt")
    torch.save({"tower_timm": ref.vision.export_weights()}, d / "tower.pt")
    torch.save({"a_mean": torch.zeros(12), "a_std": torch.ones(12), "p_mean": torch.zeros(16), "p_std": torch.ones(16)},
               d / "norm.pt")
    return {"head": str(d / "mf_final.pt"), "vision": str(d / "tower.pt"), "norm": str(d / "norm.pt")}


def convert(head, vision, norm, norm_stats, lang, cams, tasks, out: Path) -> Path:
    from dem import DEM, Normalizer
    out.mkdir(parents=True, exist_ok=True)
    pol = DEM(make_config(lang, len(cams)))
    pol.load_research_checkpoints(head=head, vision=vision)       # weights only; normalization handled below
    n = torch.load(norm, map_location="cpu", weights_only=True)
    pol.action_norm = Normalizer.from_mean_std(n["a_mean"], n["a_std"])
    pol.proprio_norm = Normalizer.from_mean_std(n["p_mean"], n["p_std"])
    stats = {"per_task": {t: task_quantiles(t, norm_stats) for t in tasks},
             "zscore": {"action": {"mean": n["a_mean"].tolist(), "std": n["a_std"].tolist()},
                        "state": {"mean": n["p_mean"].tolist(), "std": n["p_std"].tolist()}},
             "clip": 0.0, "quantiles": "research" if norm_stats else "dataset"}
    (out / "norm_stats.json").write_text(json.dumps(stats, indent=1))
    ck = out / "ckpt_0000000.pt"
    pol.save(ck, include_frozen=False)
    c = torch.load(ck, map_location="cpu", weights_only=True)
    c["rua_format"] = "sentence"
    torch.save(c, ck)
    (out / "config.json").write_text(json.dumps({
        "format": "sentence", "cams": list(cams), "tasks": list(tasks), "steps": 0, "language": lang,
        "source": {"head": str(head), "vision": str(vision), "norm": str(norm), "norm_stats": norm_stats},
        "research_checkpoint": True, "code_commit": paths.code_commit()}, indent=1))
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head")
    ap.add_argument("--vision")
    ap.add_argument("--norm")
    ap.add_argument("--norm-stats", default=None)
    ap.add_argument("--lang", default="neobert", choices=list(LANG))
    ap.add_argument("--cams", nargs="+", default=["left", "wrist"])
    ap.add_argument("--tasks", nargs="+", default=paths.ATOMIC_SEEN)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", action="store_true", help="make random-init research files first (test mode)")
    a = ap.parse_args()
    out = Path(a.out)
    if a.fake:
        f = write_fake(out / "fake_research_files", a.lang, len(a.cams))
        a.head, a.vision, a.norm = f["head"], f["vision"], f["norm"]
    ck = convert(a.head, a.vision, a.norm, a.norm_stats, a.lang, a.cams, a.tasks, out)
    # round trip through our own loader
    from policy.model import load_policy
    pol = load_policy(str(ck))
    print("RESEARCH_CKPT OK", ck, "language", pol.config.language_backend, pol.language.embed_dim,
          "cams", pol.config.num_cameras)


if __name__ == "__main__":
    main()
