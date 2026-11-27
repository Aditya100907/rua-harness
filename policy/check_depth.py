"""Spot-check re-rendered depth against the hindsight grasp points (F1 labels).

For each sampled episode: project the label's grasp point (world frame, end-effector position at
`grasp_frame`) into a camera at that frame, round to a pixel, read the point map there and
measure the distance to the grasp point. At the grasp frame the arm often sits between a camera
and the gripper (on PickPlaceCounterToCabinet the left camera sees robot0_link5 at the grasp
pixel in 7 of 10 demos), so we also report the nearest point-map point within a
(2r+1) x (2r+1) window (r = 5 pixels), and the best of the cameras per episode. A task passes
when the best-camera window distance is <= 5 cm on >= 90% of sampled episodes.

Reads only the depth store (no simulator), so it runs anywhere the files are.

    python -m policy.check_depth --tasks PickPlaceCounterToCabinet --episodes 0 1 2 3 4
    python -m policy.check_depth --all-tasks --n 20 --out results/depth_spotcheck.json   # F2 test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from policy import paths
from policy.label_commands import load_labels
from policy.rerender import PointMapStore
from simlib.camera import world_to_pixel


PASS_FRAC = 0.9      # F2 pass: best-camera window distance <= 5 cm on >= 90% of sampled episodes


def check_one(store: PointMapStore, task: str, lb: dict, cam: str, r: int = 5) -> dict:
    ep, t = lb["episode"], lb["grasp_frame"]
    g = np.asarray(lb["grasp_world"], dtype=np.float64)
    K, T = store.camera(task, ep, t, cam, "world")
    uv, z = world_to_pixel(g, K, T)
    u, v = int(np.round(uv[0])), int(np.round(uv[1]))
    row = {"task": task, "episode": ep, "frame": t, "cam": cam, "u": u, "v": v, "z_grasp_m": round(float(z), 4)}
    pm = store.point_map(task, ep, t, cam, "world")          # [3, H, W]
    H, W = pm.shape[1:]
    if z <= 0 or not (0 <= u < W and 0 <= v < H):
        row["in_view"] = False
        return row
    row["in_view"] = True
    p = pm[:, v, u]
    row["pixel_cm"] = round(100 * float(np.linalg.norm(p - g)), 2)
    row["depth_at_pixel_m"] = round(float(store.depth(task, ep, t, cam)[v, u]), 4)
    win = pm[:, max(0, v - r):v + r + 1, max(0, u - r):u + r + 1].reshape(3, -1).T
    d = np.linalg.norm(win - g, axis=1)
    row["window_cm"] = round(100 * float(np.nanmin(d)), 2)
    # base-frame point map must give the same point after mapping back to the world
    e = store.episode(task, ep)
    pb = store.point_map(task, ep, t, cam, "base")[:, v, u]
    pw = e.T_world_base[t][:3, :3] @ pb + e.T_world_base[t][:3, 3]
    row["base_roundtrip_mm"] = round(1000 * float(np.linalg.norm(pw - p)), 3)
    return row


def summarize(rows: list[dict]) -> dict:
    vis = [r for r in rows if r.get("in_view")]
    out = {"n": len(rows), "in_view": len(vis)}
    for k in ("pixel_cm", "window_cm"):
        a = np.array([r[k] for r in vis])
        if len(a):
            out[k] = {"median": round(float(np.median(a)), 2), "p90": round(float(np.percentile(a, 90)), 2),
                      "max": round(float(a.max()), 2), "within_3cm": int((a <= 3).sum()),
                      "within_5cm": int((a <= 5).sum())}
    if vis:
        out["base_roundtrip_mm_max"] = max(r["base_roundtrip_mm"] for r in vis)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["PickPlaceCounterToCabinet"])
    ap.add_argument("--all-tasks", action="store_true")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="explicit episodes (else random)")
    ap.add_argument("--n", type=int, default=20, help="random episodes per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cams", nargs="+", default=["left", "right"])
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--depth", default=None, help="depth store base")
    ap.add_argument("--labels", default=None, help="labels dir, default <frames base>/../labels")
    ap.add_argument("--out", default=None, help="write rows + summary as JSON")
    args = ap.parse_args()
    store = PointMapStore(args.depth)
    labels_dir = Path(args.labels) if args.labels else paths.frames_base().parent / "labels"
    tasks = paths.ATOMIC_SEEN if args.all_tasks else args.tasks
    rng = np.random.default_rng(args.seed)
    report = {"tasks": {}, "rows": []}
    for task in tasks:
        try:
            labels = {lb["episode"]: lb for lb in load_labels(task, labels_dir)}
        except FileNotFoundError:
            print(f"{task}: no labels, skipped", flush=True)
            continue
        ok = [e for e, lb in labels.items() if lb.get("grasp_frame") is not None and lb.get("grasp_world")
              and store.has(task, e)]
        if args.episodes:
            eps = [e for e in args.episodes if e in ok]
        else:
            eps = sorted(rng.choice(ok, size=min(args.n, len(ok)), replace=False).tolist()) if ok else []
        if not eps:
            print(f"{task}: no rendered episodes with a grasp label, skipped", flush=True)
            continue
        rows = [check_one(store, task, labels[e], c, args.window) for e in eps for c in args.cams]
        for r in rows:
            print(json.dumps(r), flush=True)
        summ = {c: summarize([r for r in rows if r["cam"] == c]) for c in args.cams}
        best = [min((r["window_cm"] for r in rows if r["episode"] == e and r.get("in_view")), default=None)
                for e in eps]
        best = np.array([b for b in best if b is not None])
        summ["best_cam_window_cm"] = {
            "n": int(len(best)), "median": round(float(np.median(best)), 2) if len(best) else None,
            "max": round(float(best.max()), 2) if len(best) else None,
            "within_5cm_frac": round(float((best <= 5).mean()), 3) if len(best) else None}
        summ["pass"] = bool(len(best) and (best <= 5).mean() >= PASS_FRAC)
        report["tasks"][task] = summ
        report["rows"] += rows
        print(task, json.dumps(report["tasks"][task]), flush=True)
    failed = [t for t, v in report["tasks"].items() if not v["pass"]]
    print("SPOTCHECK", "PASS" if report["tasks"] and not failed else f"FAIL {failed}",
          f"({len(report['tasks'])} tasks checked)", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
