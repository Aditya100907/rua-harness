"""LIBERO hdf5 demos -> DEM training samples, mirroring `policy.data.RoboCasaDataset`.

One sample is one frame t of one demo:
  images  uint8 [2, 256, 256, 3]   agentview, wrist; flipped upright, resized from 128 (bilinear)
  state   float32 [8]              stage-1 normalized (per-task quantiles to [-1, 1]); layout in policy/libero/sim.py
  actions float32 [16, 7]          actions t .. t+15, stage-1 normalized; past the end the last action repeats
  mask    bool [16]                False where padded (not used by the loss, as for RoboCasa)
  text    str                      the task's language instruction
  task    int                      index into `tasks`

"Task" here is `<suite>/<task_name>` (for example `libero_spatial/pick_up_the_black_bowl_..._plate`),
so norm_stats keys never collide across suites. Normalization reuses `policy.data` (stage 1 here,
stage 2 z-score on the policy's normalizers).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from policy.data import CHUNK, norm_stats_from_arrays, task_quantiles
from policy.libero import sim


def suite_tasks(suite: str, root: Path | None = None) -> list[str]:
    """All `<suite>/<task_name>` keys with a demo file, in file-name order (no LIBERO import needed)."""
    d = Path(root or sim.data_root()) / suite
    return [f"{suite}/{p.name[:-len('_demo.hdf5')]}" for p in sorted(d.glob("*_demo.hdf5"))]


def demo_path(task: str, root: Path | None = None) -> Path:
    return Path(root or sim.data_root()) / f"{task}_demo.hdf5"


def task_language(path: Path) -> str:
    """Instruction from the hdf5's problem_info (falls back to the file name)."""
    import json
    import h5py
    with h5py.File(path, "r") as f:
        pi = f["data"].attrs.get("problem_info")
        if pi is not None:
            lang = json.loads(pi).get("language_instruction", "")
            if lang:
                return " ".join(lang.split()) if isinstance(lang, str) else " ".join(lang)
    name = path.name[:-len("_demo.hdf5")]
    words = [w for w in name.split("_") if not (w.isupper() or w.startswith("SCENE"))]
    return " ".join(words)


def load_task_table(task: str, root: Path | None = None) -> dict:
    p = demo_path(task, root)
    low = sim.demo_lowdim(p)
    lens = np.asarray([len(a) for a in low["action"]])
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    return {"task": task, "path": p, "keys": low["keys"], "lens": lens, "starts": starts,
            "states": np.concatenate(low["state"]), "actions": np.concatenate(low["action"]),
            "instr": [task_language(p)] * len(lens)}


class LiberoDataset:
    """Map-style dataset over (task, demo, frame). Works as a torch Dataset; hdf5 files open lazily per worker."""

    def __init__(self, tasks: list[str], cams: tuple[str, ...] = sim.DEFAULT_CAMS, chunk: int = CHUNK,
                 norm_stats: dict | None = None, episodes: dict | None = None, root: Path | None = None,
                 size: int = 256):
        self.tasks, self.cams, self.chunk, self.size = list(tasks), list(cams), chunk, size
        self.tables = [load_task_table(t, root) for t in self.tasks]

        def kept(tb):
            keep = episodes.get(tb["task"]) if episodes else None
            if keep is None:
                return tb["states"], tb["actions"]
            rows = np.concatenate([np.arange(tb["starts"][e], tb["starts"][e] + tb["lens"][e]) for e in keep])
            return tb["states"][rows], tb["actions"][rows]
        self.norm_stats = norm_stats or norm_stats_from_arrays({t["task"]: kept(t) for t in self.tables})
        self.quant = [task_quantiles(self.norm_stats, t) for t in self.tasks]
        idx = []
        for ti, tb in enumerate(self.tables):
            keep = episodes.get(tb["task"]) if episodes else None
            for ep, n in enumerate(tb["lens"]):
                if keep is not None and ep not in keep:
                    continue
                g0 = tb["starts"][ep]
                idx.append(np.stack([np.full(n, ti), np.full(n, ep), np.arange(n), g0 + np.arange(n)], 1))
        self.index = np.concatenate(idx).astype(np.int64)
        self._fh: dict = {}

    def __len__(self) -> int:
        return len(self.index)

    def instructions(self) -> list[str]:
        return sorted({s for tb in self.tables for s in tb["instr"]})

    def _file(self, ti: int):
        f = self._fh.get(ti)
        if f is None:
            import h5py
            f = self._fh[ti] = h5py.File(self.tables[ti]["path"], "r")
        return f

    def _images(self, ti: int, ep: int, t: int) -> np.ndarray:
        d = self._file(ti)["data"][self.tables[ti]["keys"][ep]]["obs"]
        return np.stack([sim.upright(d[sim.CAMERAS[c][1]][t], self.size) for c in self.cams])

    def __getitem__(self, i: int) -> dict:
        ti, ep, t, g = (int(v) for v in self.index[i])
        tb = self.tables[ti]
        n = int(tb["lens"][ep])
        steps = np.arange(t, t + self.chunk)
        mask = steps < n
        rows = int(tb["starts"][ep]) + np.minimum(steps, n - 1)
        qs, qa = self.quant[ti]
        return {"images": self._images(ti, ep, t),
                "state": qs.normalize(tb["states"][g]).astype(np.float32),
                "actions": qa.normalize(tb["actions"][rows]).astype(np.float32),
                "mask": mask, "text": tb["instr"][ep], "task": ti, "episode": ep, "frame": t,
                "progress": np.float32(t / max(1, n - 1))}

    def raw(self, i: int) -> dict:
        ti, ep, t, g = (int(v) for v in self.index[i])
        tb = self.tables[ti]
        n = int(tb["lens"][ep])
        rows = tb["starts"][ep] + np.minimum(np.arange(t, t + self.chunk), n - 1)
        return {"state": tb["states"][g], "actions": tb["actions"][rows]}

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_fh"] = {}
        return d
