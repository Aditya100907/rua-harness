"""E5: sort failed pick-and-place episodes by how they failed, from eval trajectories (--save-traj).

Each `<task>_epNNN_traj.npz` holds per-step eef position, gripper command (+1 close) and the target
object's position. Categories, checked in order:
  never_reached   the gripper never came within 6 cm of the object
  missed_grasp    it came close but the object never rose 3 cm above its start
  dropped         the object rose 3 cm or more, then ended within 3 cm of its start height
                  or fell while the gripper was still commanded closed
  wrong_place     the object was carried and released, but the task did not succeed
  timed_out       none of the above (still carrying or moving at the horizon)
    python -m policy.failures <eval_media_dir> [--task PickPlaceCounterToCabinet]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json

import numpy as np


def classify(z) -> str:
    eef, obj, grip = z["eef"], z["obj"], z["grip_cmd"]
    if len(eef) == 0 or not np.any(obj):
        return "no_data"
    d = np.linalg.norm(eef - obj, axis=1)
    lift = obj[:, 2] - obj[0, 2]
    if d.min() > 0.06:
        return "never_reached"
    if lift.max() < 0.03:
        return "missed_grasp"
    t_peak = int(np.argmax(lift))
    closed_after = grip[t_peak:] > 0
    fell = lift[t_peak:] < lift.max() - 0.05
    if np.any(fell & closed_after) or lift[-1] < 0.03 and np.all(grip[t_peak:] > 0):
        return "dropped"
    if np.any(grip[t_peak:] < 0):
        return "wrong_place"
    return "timed_out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--task", default="PickPlaceCounterToCabinet")
    a = ap.parse_args()
    counts, rows = collections.Counter(), []
    for f in sorted(glob.glob(f"{a.dir}/{a.task}_ep*_traj.npz")):
        z = np.load(f)
        if int(z["success"]):
            counts["success"] += 1
            continue
        c = classify(z)
        counts[c] += 1
        rows.append((f.split("/")[-1], c))
    print(json.dumps(dict(counts)))
    for r in rows:
        print(*r)


if __name__ == "__main__":
    main()
