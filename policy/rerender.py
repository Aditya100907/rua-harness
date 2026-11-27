"""Re-render demos with depth, so every dataset frame gets a metric point map.

The RoboCasa videos have RGB only. The saved simulator states (`extras/.../states.npz`) let us
rebuild each frame exactly: reset to the demo scene, then for every frame t set states[t],
call forward() and render RGB + depth from the dataset cameras.

One file per episode:

    <out>/<task>/episode_NNNNNN.npz      written last (tmp + rename), so a file means "done"

Keys (all images top-row-first, the dataset video orientation):

    cams                [K] str        camera short names, e.g. ["left", "right"]
    size                int            image height = width
    n_frames            int            T
    K_<cam>             [3, 3] f64     intrinsics (fixed per camera)
    T_world_cam_<cam>   [T, 4, 4] f32  camera pose per frame, OpenCV axes (+z forward, +y down).
                                       The agentview cameras ride on the mobile base, so they move.
    T_world_base        [T, 4, 4] f32  robot base pose per frame (from the parquet state)
    group               int            G, frames per compressed group (8)
    dz_<cam>            [B] uint8      depth in millimetres (uint16, 0 = invalid), compressed in
                                       groups of G frames: frame 0 of a group is differenced along
                                       rows, the rest against the previous frame, then zlib
    off_<cam>           [ceil(T/G) + 1] i64   byte offsets of the groups in dz_<cam>
    rgb_mae_<cam>       [T] f32        mean abs error vs the dataset video (0..255), if checked

Grouped time differences are 2.5x smaller than per-frame 16-bit PNG (9 vs 22 KB per frame on
PickPlaceCounterToCabinet ep 0) and still give random access: one read decodes G frames.
The outer npz is NOT deflated (np.savez), because the groups are already compressed. That lets
`PointMapStore` seek straight to one group without reading the whole episode.

    python -m policy.rerender --task PickPlaceCounterToCabinet --episodes 0 1 2 3 4 --check-rgb
    python -m policy.rerender --task OpenDrawer --chunk 3 --chunk-size 50     # episodes 150..199
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np

from policy import paths, sim
from policy.extract_frames import decode_video, episode_lengths, video_path
from simlib import camera as C

MM_MAX = 65535           # uint16 limit: 65.535 m; farther (or no hit) is stored as 0 = invalid


def depth_base() -> Path:
    if os.environ.get("RUA_DEPTH"):
        return Path(os.environ["RUA_DEPTH"])
    if paths.BOUCHET_SCRATCH.exists():
        return paths.BOUCHET_SCRATCH / "depth"
    return Path.home() / "apollo-lab" / "runs" / "depth"


def episode_file(base: Path, task: str, ep: int) -> Path:
    return Path(base) / task / f"episode_{ep:06d}.npz"


# ---------------------------------------------------------------- encoding

def depth_to_mm(depth_m: np.ndarray) -> np.ndarray:
    mm = np.round(np.asarray(depth_m, dtype=np.float64) * 1000.0)
    mm[~np.isfinite(mm) | (mm <= 0) | (mm > MM_MAX)] = 0
    return mm.astype(np.uint16)


GROUP = 8


def encode_group(mm: np.ndarray, level: int = 6) -> bytes:
    """[g, H, W] uint16 mm -> bytes. Differences wrap modulo 2**16, so decoding is exact."""
    a = np.asarray(mm, dtype=np.uint16)
    d = np.empty_like(a)
    d[0, :, 0] = a[0, :, 0]
    d[0, :, 1:] = a[0, :, 1:] - a[0, :, :-1]
    d[1:] = a[1:] - a[:-1]
    return zlib.compress(d.tobytes(), level)


def decode_group(b: bytes, g: int, H: int, W: int) -> np.ndarray:
    d = np.frombuffer(zlib.decompress(b), dtype=np.uint16).reshape(g, H, W)
    a = np.empty_like(d)
    a[0] = np.cumsum(d[0], axis=1, dtype=np.uint16)
    for i in range(1, g):
        a[i] = a[i - 1] + d[i]
    return a


def pose_from_state(state16: np.ndarray) -> np.ndarray:
    """[T, 16] dataset state -> [T, 4, 4] T_world_base. robosuite quats are xyzw."""
    from scipy.spatial.transform import Rotation as R
    s = np.asarray(state16, dtype=np.float64)
    T = np.tile(np.eye(4), (len(s), 1, 1))
    T[:, :3, :3] = R.from_quat(s[:, 3:7]).as_matrix()
    T[:, :3, 3] = s[:, 0:3]
    return T


def episode_states16(ds: Path, ep: int) -> np.ndarray:
    import pandas as pd
    f = next(Path(ds).glob(f"data/*/episode_{ep:06d}.parquet"))
    return np.stack(pd.read_parquet(f, columns=["observation.state"])["observation.state"].to_list())


# ---------------------------------------------------------------- rendering

def rerender_episode(env, ds: Path, ep: int, cams: list[str], size: int = 256,
                     check_rgb: bool = False) -> dict:
    """Render every frame of one demo. Returns the arrays to save (see the module docstring)."""
    t0 = time.time()
    states, _, _ = sim.episode_extras(ds, ep)
    T = len(states)
    sim.reset_to_episode(env, ds, ep)
    t_reset = time.time() - t0
    s = env.sim
    near, far = C.near_far(s)
    names = {c: sim.CAMERAS.get(c, c) for c in cams}
    out: dict = {"cams": np.array(cams), "size": np.int64(size), "n_frames": np.int64(T)}
    mm = {c: np.zeros((T, size, size), np.uint16) for c in cams}
    poses = {c: np.zeros((T, 4, 4), np.float32) for c in cams}
    rgbs = {c: np.zeros((T, size, size, 3), np.uint8) for c in cams} if check_rgb else None
    for c in cams:
        out[f"K_{c}"] = C.camera_K(s, names[c], size, size)
    for t in range(T):
        s.set_state_from_flattened(states[t])
        s.forward()
        for c in cams:
            rgb, d = s.render(camera_name=names[c], height=size, width=size, depth=True)
            depth = C.flip_rows(C.metric_depth(d, near, far))
            mm[c][t] = depth_to_mm(depth)
            poses[c][t] = C.camera_T(s, names[c])
            if check_rgb:
                rgbs[c][t] = C.flip_rows(rgb)
    t_render = time.time() - t0 - t_reset
    out["group"] = np.int64(GROUP)
    for c in cams:
        blobs = [encode_group(mm[c][s0:s0 + GROUP]) for s0 in range(0, T, GROUP)]
        off = np.zeros(len(blobs) + 1, np.int64)
        off[1:] = np.cumsum([len(b) for b in blobs])
        out[f"dz_{c}"] = np.frombuffer(b"".join(blobs), dtype=np.uint8)
        out[f"off_{c}"] = off
        out[f"T_world_cam_{c}"] = poses[c]
    s16 = episode_states16(ds, ep)
    if len(s16) != T:
        raise ValueError(f"episode {ep}: {T} saved states but {len(s16)} parquet rows")
    out["T_world_base"] = pose_from_state(s16).astype(np.float32)
    if check_rgb:
        for c in cams:
            vid = decode_video(video_path(ds, c, ep))
            n = min(len(vid), T)
            mae = np.full(T, np.nan, np.float32)
            mae[:n] = np.abs(vid[:n].astype(np.int16) - rgbs[c][:n].astype(np.int16)).mean(axis=(1, 2, 3))
            out[f"rgb_mae_{c}"] = mae
    out["seconds_reset_render_total"] = np.array([t_reset, t_render, time.time() - t0])
    return out


def save_episode(path: Path, arrays: dict) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)          # stored, not deflated: the groups are compressed already
    tmp.rename(path)
    return path.stat().st_size


def run(task: str, episodes: list[int], cams: list[str], out_base: Path, size: int = 256,
        check_rgb_every: int = 0, force: bool = False) -> list[dict]:
    """Render `episodes`, skipping ones already on disk. Compares RGB with the dataset video on
    episodes whose index is a multiple of `check_rgb_every` (0 = never). Appends one JSON row per
    episode to <out>/<task>/rerender_log.jsonl."""
    ds = paths.task_dir(task)
    todo = [e for e in episodes if force or not episode_file(out_base, task, e).exists()]
    print(f"{task}: {len(episodes) - len(todo)} done already, {len(todo)} to render", flush=True)
    if not todo:
        return []
    t0 = time.time()
    env = sim.make_env(ds, cams=cams, size=size, depth=True)
    print(f"env built in {time.time() - t0:.1f}s", flush=True)
    rows = []
    for ep in todo:
        t1 = time.time()
        try:
            check = bool(check_rgb_every) and ep % check_rgb_every == 0
            arrays = rerender_episode(env, ds, ep, cams, size, check)
        except Exception as e:     # a bad episode must not kill the whole chunk
            print(f"{task} ep {ep}: FAILED {type(e).__name__}: {e}", flush=True)
            rows.append({"task": task, "episode": ep, "error": str(e)})
            continue
        nbytes = save_episode(episode_file(out_base, task, ep), arrays)
        row = {"task": task, "episode": ep, "frames": int(arrays["n_frames"]),
               "seconds": round(time.time() - t1, 2), "bytes": nbytes,
               "reset_s": round(float(arrays["seconds_reset_render_total"][0]), 2),
               "render_s": round(float(arrays["seconds_reset_render_total"][1]), 2)}
        for c in cams:
            if f"rgb_mae_{c}" in arrays:
                row[f"rgb_mae_{c}"] = round(float(np.nanmean(arrays[f"rgb_mae_{c}"])), 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
        with open(Path(out_base) / task / "rerender_log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    env.close()
    return rows


# ---------------------------------------------------------------- reading

class _Episode:
    """Lazy view of one episode file. Reads small arrays once, seeks to one group per read."""

    def __init__(self, path: Path):
        self.path = path
        z = np.load(path, allow_pickle=False)
        self.cams = [str(c) for c in z["cams"]]
        self.n_frames = int(z["n_frames"])
        self.size = int(z["size"])
        self.group = int(z["group"])
        self.K = {c: z[f"K_{c}"] for c in self.cams}
        self.T_world_cam = {c: z[f"T_world_cam_{c}"].astype(np.float64) for c in self.cams}
        self.off = {c: z[f"off_{c}"] for c in self.cams}
        self.T_world_base = z["T_world_base"].astype(np.float64)
        z.close()
        self._data_start: dict = {}
        with zipfile.ZipFile(path) as zf:
            for c in self.cams:
                info = zf.getinfo(f"dz_{c}.npy")
                assert info.compress_type == zipfile.ZIP_STORED, "episode file must be np.savez, not compressed"
                with zf.open(info) as f:     # npy header length inside the member
                    ver = np.lib.format.read_magic(f)
                    if ver == (1, 0):
                        np.lib.format.read_array_header_1_0(f)
                    else:
                        np.lib.format.read_array_header_2_0(f)
                    hdr = f.tell()
                self._data_start[c] = (info.header_offset, hdr)
        with open(path, "rb") as fh:            # zip local file header: 30 bytes + name + extra
            for c, (local_off, hdr) in self._data_start.items():
                fh.seek(local_off)
                lh = fh.read(30)
                n_name, n_extra = int.from_bytes(lh[26:28], "little"), int.from_bytes(lh[28:30], "little")
                self._data_start[c] = local_off + 30 + n_name + n_extra + hdr
        self._fh = None
        self._cache: dict = {}      # cam -> (group index, decoded [g, H, W] uint16)

    def depth_mm(self, cam: str, t: int) -> np.ndarray:
        if not 0 <= t < self.n_frames:
            raise IndexError(f"frame {t} out of range 0..{self.n_frames - 1}")
        gi = t // self.group
        hit = self._cache.get(cam)
        if hit is None or hit[0] != gi:
            g = min(self.group, self.n_frames - gi * self.group)
            hit = self._cache[cam] = (gi, decode_group(self._read(cam, gi), g, self.size, self.size))
        return hit[1][t - gi * self.group]

    def _read(self, cam: str, gi: int) -> bytes:
        if self._fh is None:
            self._fh = open(self.path, "rb")
        start = self._data_start[cam]
        s, e = int(self.off[cam][gi]), int(self.off[cam][gi + 1])
        self._fh.seek(start + s)
        return self._fh.read(e - s)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._cache = {}


class PointMapStore:
    """Read re-rendered depth as metric depth or point maps. Safe in DataLoader workers.

        store = PointMapStore()                      # base: $RUA_DEPTH, scratch/depth or runs/depth
        pm = store.point_map("OpenDrawer", ep=3, t=40, cam="left", frame="base")   # [3, 256, 256] f32

    frame="world": world coordinates. frame="base": the robot base frame at that same frame t.
    Invalid pixels (no depth) are NaN.
    """

    def __init__(self, base: Path | None = None, max_open: int = 64):
        self.base = Path(base) if base else depth_base()
        self.max_open = max_open
        self._eps: dict = {}

    def episode(self, task: str, ep: int) -> _Episode:
        key = (task, ep)
        e = self._eps.get(key)
        if e is None:
            if len(self._eps) >= self.max_open:
                old = next(iter(self._eps))
                self._eps.pop(old).close()
            e = self._eps[key] = _Episode(episode_file(self.base, task, ep))
        return e

    def has(self, task: str, ep: int) -> bool:
        return episode_file(self.base, task, ep).exists()

    def depth(self, task: str, ep: int, t: int, cam: str) -> np.ndarray:
        """[H, W] float32 metres, NaN where invalid."""
        mm = self.episode(task, ep).depth_mm(cam, t).astype(np.float32)
        mm[mm == 0] = np.nan
        return mm / 1000.0

    def camera(self, task: str, ep: int, t: int, cam: str, frame: str = "world") -> tuple[np.ndarray, np.ndarray]:
        """(K, T_frame_cam) for this frame."""
        e = self.episode(task, ep)
        T = e.T_world_cam[cam][t]
        if frame == "base":
            T = np.linalg.inv(e.T_world_base[t]) @ T
        elif frame != "world":
            raise ValueError(frame)
        return e.K[cam], T

    def point_map(self, task: str, ep: int, t: int, cam: str, frame: str = "world") -> np.ndarray:
        """[3, H, W] float32 point map in the world or robot base frame. NaN where invalid."""
        d = self.depth(task, ep, t, cam)
        K, T = self.camera(task, ep, t, cam, frame)
        H, W = d.shape
        u, v = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)
        x = (u - K[0, 2]) * d / K[0, 0]
        y = (v - K[1, 2]) * d / K[1, 1]
        cam_pts = np.stack([x, y, d], 0).reshape(3, -1)
        R, p = T[:3, :3].astype(np.float32), T[:3, 3].astype(np.float32)
        return (np.einsum("ij,jn->in", R, cam_pts) + p[:, None]).reshape(3, H, W)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_eps"] = {}
        return d


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="explicit episode indices")
    ap.add_argument("--chunk", type=int, default=None, help="render episodes [chunk*size, (chunk+1)*size)")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--cams", nargs="+", default=["left", "right"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-rgb", action="store_true", help="compare every episode with the dataset video")
    ap.add_argument("--check-rgb-every", type=int, default=0, help="compare episodes with index %% k == 0")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    n = len(episode_lengths(paths.task_dir(args.task)))
    if args.episodes:
        eps = args.episodes
    elif args.chunk is not None:
        eps = list(range(args.chunk * args.chunk_size, min(n, (args.chunk + 1) * args.chunk_size)))
    else:
        eps = list(range(n))
    out = Path(args.out) if args.out else depth_base()
    if not eps:
        print(f"{args.task}: no episodes in this range (task has {n}), nothing to do", flush=True)
        return
    every = 1 if args.check_rgb else args.check_rgb_every
    rows = run(args.task, eps, args.cams, out, args.size, every, args.force)
    bad = [r for r in rows if "error" in r]
    print(f"done: {len(rows) - len(bad)} rendered, {len(bad)} failed", flush=True)


if __name__ == "__main__":
    main()
