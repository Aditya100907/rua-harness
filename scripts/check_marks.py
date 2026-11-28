"""P-2 check: the training-side mark position (camera poses stored with the re-rendered depth) equals the
eval-side position (live simulator camera poses) for every camera, at several frames of several demos.
    python scripts/check_marks.py [--depth-dir DIR]
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--depth-dir", default=None)
a = ap.parse_args()
if a.depth_dir:
    os.environ["RUA_DEPTH"] = a.depth_dir
from policy import paths, sim
from policy.label_commands import load_labels
from policy.rerender import PointMapStore
from simlib.camera import camera_K, camera_T, world_to_pixel

task = "PickPlaceCounterToCabinet"
ds = paths.task_dir(task)
store = PointMapStore()
labels = {l["episode"]: l for l in load_labels(task, paths.frames_base().parent / "labels_center")}
env = sim.make_env(ds, cams=["left", "right", "wrist"])
worst = 0.0
for ep in (0, 7, 31):
    states, _, _ = sim.episode_extras(ds, ep)
    sim.reset_to_episode(env, ds, ep)
    p = np.asarray(labels[ep]["grasp_world"])
    for t in (0, len(states) // 3, 2 * len(states) // 3):
        env.sim.set_state_from_flattened(states[t]); env.sim.forward()
        for cam in ("left", "right", "wrist"):
            K1, T1 = store.camera(task, ep, t, cam, frame="world")
            name = sim.CAMERAS[cam]
            (u1, v1), z1 = world_to_pixel(p, K1, T1)
            (u2, v2), z2 = world_to_pixel(p, camera_K(env.sim, name, 256, 256), camera_T(env.sim, name))
            d = float(np.hypot(u1 - u2, v1 - v2))
            worst = max(worst, d)
            print(f"ep {ep} t {t} {cam:5s} store ({u1:7.1f},{v1:7.1f}) live ({u2:7.1f},{v2:7.1f}) diff {d:.2f} px")
print("MARKCHECK worst pixel difference", round(worst, 3))
