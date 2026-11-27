"""Pre-extract camera frames from the LeRobot mp4 files, once, for fast training.

Random access into h264 mp4 files is slow (each sample seeks to a keyframe and
decodes forward), so we decode every video once and store each frame as a JPEG
(quality 95) in one flat file per task and camera:

    <frames_base>/<task>/<camera>.bin       concatenated JPEG bytes
    <frames_base>/<task>/<camera>.idx.npy   int64 offsets, length n_frames + 1,
                                            frames in episode order, then frame order
    <frames_base>/<task>/DONE               written last, with frame counts

    python -m policy.extract_frames --tasks OpenDrawer --cams left right
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from policy import paths
from policy.sim import CAMERAS


def episode_lengths(ds_dir: Path) -> list[int]:
    with open(ds_dir / "meta" / "episodes.jsonl") as f:
        eps = [json.loads(l) for l in f if l.strip()]
    eps.sort(key=lambda e: e["episode_index"])
    assert [e["episode_index"] for e in eps] == list(range(len(eps))), "episode indices are not 0..N-1"
    return [int(e["length"]) for e in eps]


def video_path(ds_dir: Path, cam: str, ep: int, chunk_size: int = 1000) -> Path:
    key = f"observation.images.{CAMERAS.get(cam, cam)}"
    return ds_dir / "videos" / f"chunk-{ep // chunk_size:03d}" / key / f"episode_{ep:06d}.mp4"


def decode_video(path: Path) -> np.ndarray:
    """All frames of an mp4 as uint8 [T, H, W, 3]."""
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        return np.stack([fr.to_ndarray(format="rgb24") for fr in c.decode(s)])


def encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


def decode_jpeg(b: bytes | np.ndarray) -> np.ndarray:
    import cv2
    arr = np.frombuffer(b, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)[:, :, ::-1]


def extract_task(task: str, cams: list[str], out_base: Path, quality: int = 95,
                 max_episodes: int | None = None) -> dict:
    """`max_episodes` keeps only the first episodes (for tests); global frame indices stay valid."""
    ds = paths.task_dir(task)
    lens = episode_lengths(ds)[:max_episodes]
    out = out_base / task
    out.mkdir(parents=True, exist_ok=True)
    info = {"task": task, "episodes": len(lens), "frames": int(sum(lens)), "cams": {}, "quality": quality}
    t0 = time.time()
    for cam in cams:
        offsets = [0]
        mismatched = []
        tmp = out / f"{cam}.bin.tmp"
        with open(tmp, "wb") as f:
            for ep, n in enumerate(lens):
                frames = decode_video(video_path(ds, cam, ep))
                if len(frames) != n:               # keep the index aligned with the parquet rows
                    mismatched.append((ep, len(frames), n))
                    frames = frames[:n] if len(frames) > n else np.concatenate(
                        [frames, np.repeat(frames[-1:], n - len(frames), 0)])
                for fr in frames:
                    b = encode_jpeg(fr, quality)
                    f.write(b)
                    offsets.append(offsets[-1] + len(b))
        tmp.rename(out / f"{cam}.bin")
        np.save(out / f"{cam}.idx.npy", np.asarray(offsets, dtype=np.int64))
        info["cams"][cam] = {"bytes": offsets[-1], "mismatched_episodes": mismatched[:20],
                             "n_mismatched": len(mismatched)}
        print(task, cam, f"{offsets[-1] / 1e9:.2f} GB", f"{len(mismatched)} length mismatches",
              f"{time.time() - t0:.0f}s", flush=True)
    info["seconds"] = round(time.time() - t0, 1)
    (out / "DONE").write_text(json.dumps(info, indent=1))
    return info


class FrameStore:
    """Read pre-extracted frames. Opens files lazily, so it is safe in DataLoader workers."""

    def __init__(self, task: str, cams: list[str], base: Path | None = None):
        self.dir = Path(base or paths.frames_base()) / task
        if not (self.dir / "DONE").exists():
            raise FileNotFoundError(f"frames not extracted for {task} at {self.dir}")
        self.cams = cams
        self.idx = {c: np.load(self.dir / f"{c}.idx.npy") for c in cams}
        self._fh: dict = {}

    def get(self, cam: str, i: int) -> np.ndarray:
        fh = self._fh.get(cam)
        if fh is None:
            fh = self._fh[cam] = open(self.dir / f"{cam}.bin", "rb")
        s, e = self.idx[cam][i], self.idx[cam][i + 1]
        fh.seek(int(s))
        return decode_jpeg(fh.read(int(e - s)))

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_fh"] = {}
        return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=paths.SMOKE_TASKS)
    ap.add_argument("--cams", nargs="+", default=["left", "right", "wrist"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else paths.frames_base()
    for t in args.tasks:
        if (out / t / "DONE").exists() and not args.force:
            print("skip, already done:", t)
            continue
        print(json.dumps(extract_task(t, args.cams, out, args.quality)))


if __name__ == "__main__":
    main()
