"""RoboCasa LeRobot demos -> DEM training samples.

One sample is one frame t of one demo:
  images  uint8 [K, 256, 256, 3]   cameras in `cams` order (default left, right agent view)
  state   float32 [16]             stage-1 normalized (per-task quantiles to [-1, 1])
  actions float32 [16, 12]         actions t .. t+15, stage-1 normalized; past the episode end
                                   the last action is repeated
  mask    bool [16]                False where the chunk was padded. The loss does NOT use it (DEM.loss has
                                   no mask): padded steps train on the repeated last action, a deliberate
                                   choice that teaches the policy to hold still at the end
  text    str                      the episode's instruction
  task    int                      index into `tasks`

Normalization follows the DEM research code (`DEM.load_research_checkpoints`):
stage 1 maps each task's q01..q99 to [-1, 1] (constant dims map to -1 and are
restored exactly), stage 2 z-scores with statistics over all training tasks.
The dataset applies stage 1; the policy's own normalizers hold stage 2.
Both are saved in `norm_stats.json`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from policy import paths
from policy.extract_frames import FrameStore, decode_video, episode_lengths, video_path

CHUNK = 16


# ---------------------------------------------------------------- normalization
@dataclass
class Quantile:
    """numpy twin of `dem.Normalizer.from_quantiles`."""
    lo: np.ndarray
    hi: np.ndarray
    eps: float = 1e-6

    def __post_init__(self):
        self.lo = np.asarray(self.lo, np.float64)
        self.hi = np.asarray(self.hi, np.float64)
        self.valid = (self.hi - self.lo) > self.eps
        rng = np.where(self.valid, self.hi - self.lo, 1.0)
        self.offset, self.scale = self.lo + rng / 2, rng / 2

    def normalize(self, x):
        return (np.asarray(x, np.float64) - self.offset) / self.scale

    def unnormalize(self, y):
        x = np.asarray(y, np.float64) * self.scale + self.offset
        return np.where(self.valid, x, self.lo)


def norm_stats_from_arrays(per_task: dict[str, tuple[np.ndarray, np.ndarray]], clip: float = 0.0) -> dict:
    """per_task: task -> (states [N, 16], actions [N, 12]) over every frame. `clip` clips stage-1 values."""
    out = {"per_task": {}, "zscore": {}, "clip": clip}
    s1_s, s1_a = [], []
    for task, (s, a) in per_task.items():
        qs = (np.quantile(s, 0.01, 0), np.quantile(s, 0.99, 0))
        qa = (np.quantile(a, 0.01, 0), np.quantile(a, 0.99, 0))
        out["per_task"][task] = {"state": {"q01": qs[0].tolist(), "q99": qs[1].tolist()},
                                 "action": {"q01": qa[0].tolist(), "q99": qa[1].tolist()}}
        ns, na = Quantile(*qs).normalize(s), Quantile(*qa).normalize(a)
        if clip:
            ns, na = np.clip(ns, -clip, clip), np.clip(na, -clip, clip)
        s1_s.append(ns)
        s1_a.append(na)
    S, A = np.concatenate(s1_s), np.concatenate(s1_a)
    out["zscore"] = {"state": {"mean": S.mean(0).tolist(), "std": S.std(0).tolist()},
                     "action": {"mean": A.mean(0).tolist(), "std": A.std(0).tolist()}}
    return out


def task_quantiles(stats: dict, task: str) -> tuple[Quantile, Quantile]:
    t = stats["per_task"][task]
    return Quantile(t["state"]["q01"], t["state"]["q99"]), Quantile(t["action"]["q01"], t["action"]["q99"])


def zscore_normalizers(stats: dict):
    """Stage-2 normalizers to put on the policy (`policy.action_norm`, `policy.proprio_norm`)."""
    from dem import Normalizer
    z = stats["zscore"]
    return (Normalizer.from_mean_std(np.asarray(z["action"]["mean"]), np.asarray(z["action"]["std"])),
            Normalizer.from_mean_std(np.asarray(z["state"]["mean"]), np.asarray(z["state"]["std"])))


# ---------------------------------------------------------------- episodes
def load_task_table(task: str) -> dict:
    """All per-frame low-dim data of a task, concatenated in episode order."""
    import pandas as pd
    ds = paths.task_dir(task)
    lens = episode_lengths(ds)
    with open(ds / "meta" / "tasks.jsonl") as f:
        texts = {d["task_index"]: d["task"] for d in map(json.loads, filter(str.strip, f))}
    states, actions, instr = [], [], []
    for ep, n in enumerate(lens):
        f = next(ds.glob(f"data/*/episode_{ep:06d}.parquet"))
        df = pd.read_parquet(f, columns=["observation.state", "action", "annotation.human.task_description"])
        assert len(df) == n, f"{task} ep {ep}: parquet has {len(df)} rows, episodes.jsonl says {n}"
        states.append(np.stack(df["observation.state"].to_list()))
        actions.append(np.stack(df["action"].to_list()))
        instr.append(texts[int(df["annotation.human.task_description"].iloc[0])])
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return {"task": task, "dir": ds, "lens": np.asarray(lens), "starts": starts,
            "states": np.concatenate(states), "actions": np.concatenate(actions), "instr": instr}


class RoboCasaDataset:
    """Map-style dataset over (task, episode, frame). Works as a torch Dataset."""

    def __init__(self, tasks: list[str], cams: tuple[str, ...] = ("left", "right"), chunk: int = CHUNK,
                 norm_stats: dict | None = None, frames_base: Path | None = None, episodes: dict | None = None,
                 use_video: bool = False, clip: float = 0.0):
        """`episodes`: optional task -> list of episode ids to keep (for held-out splits)."""
        self.tasks, self.cams, self.chunk = list(tasks), list(cams), chunk
        self.tables = [load_task_table(t) for t in self.tasks]
        def kept(tb):   # statistics only over the episodes this dataset trains on (held-out splits)
            keep = episodes.get(tb["task"]) if episodes else None
            if keep is None:
                return tb["states"], tb["actions"]
            rows = np.concatenate([np.arange(tb["starts"][e], tb["starts"][e] + tb["lens"][e]) for e in keep])
            return tb["states"][rows], tb["actions"][rows]
        self.clip = clip if not norm_stats else norm_stats.get("clip", 0.0)
        self.norm_stats = norm_stats or norm_stats_from_arrays({t["task"]: kept(t) for t in self.tables}, clip)
        self.quant = [task_quantiles(self.norm_stats, t) for t in self.tasks]
        self.frames_base, self.use_video = frames_base, use_video
        self._stores: dict = {}
        idx = []
        for ti, tb in enumerate(self.tables):
            keep = episodes.get(tb["task"]) if episodes else None
            for ep, n in enumerate(tb["lens"]):
                if keep is not None and ep not in keep:
                    continue
                g0 = tb["starts"][ep]
                idx.append(np.stack([np.full(n, ti), np.full(n, ep), np.arange(n), g0 + np.arange(n)], 1))
        self.index = np.concatenate(idx).astype(np.int64)    # rows: task, episode, frame, global frame

    def __len__(self) -> int:
        return len(self.index)

    def instructions(self) -> list[str]:
        return sorted({s for tb in self.tables for s in tb["instr"]})

    def _store(self, ti: int):
        s = self._stores.get(ti)
        if s is None:
            s = self._stores[ti] = FrameStore(self.tasks[ti], self.cams, self.frames_base)
        return s

    def _images(self, ti: int, ep: int, t: int, g: int) -> np.ndarray:
        if self.use_video:   # slow path for tests without extracted frames
            ds = self.tables[ti]["dir"]
            return np.stack([decode_video(video_path(ds, c, ep))[t] for c in self.cams])
        st = self._store(ti)
        return np.stack([st.get(c, g) for c in self.cams])

    def __getitem__(self, i: int) -> dict:
        ti, ep, t, g = (int(v) for v in self.index[i])
        tb = self.tables[ti]
        n = int(tb["lens"][ep])
        g0 = int(tb["starts"][ep])
        steps = np.arange(t, t + self.chunk)
        mask = steps < n
        rows = g0 + np.minimum(steps, n - 1)
        qs, qa = self.quant[ti]
        st_n, ac_n = qs.normalize(tb["states"][g]), qa.normalize(tb["actions"][rows])
        c = getattr(self, "clip", 0)
        if c:
            st_n, ac_n = np.clip(st_n, -c, c), np.clip(ac_n, -c, c)
        return {"images": self._images(ti, ep, t, g),
                "state": st_n.astype(np.float32),
                "actions": ac_n.astype(np.float32),
                "mask": mask, "text": tb["instr"][ep], "task": ti, "episode": ep, "frame": t,
                "progress": np.float32(t / max(1, n - 1))}

    # raw (un-normalized) access, used by tests and labeling
    def raw(self, i: int) -> dict:
        ti, ep, t, g = (int(v) for v in self.index[i])
        tb = self.tables[ti]
        n = int(tb["lens"][ep])
        rows = tb["starts"][ep] + np.minimum(np.arange(t, t + self.chunk), n - 1)
        return {"state": tb["states"][g], "actions": tb["actions"][rows]}


def collate(batch: list[dict]) -> dict:
    import torch
    out = {k: torch.as_tensor(np.stack([b[k] for b in batch])) for k in ("images", "state", "actions", "mask")}
    out["task"] = torch.as_tensor([b["task"] for b in batch])
    if "progress" in batch[0]:
        out["progress"] = torch.as_tensor(np.stack([b["progress"] for b in batch]))
    out["text"] = [b["text"] for b in batch]
    return out


def save_norm_stats(stats: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(stats, indent=1))


def load_norm_stats(path: Path) -> dict:
    return json.loads(Path(path).read_text())
