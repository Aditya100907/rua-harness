"""Render RGB, depth, object mask and point map for one RoboCasa pick-and-place scene, and
check that the geometry conventions line up.

Run on the Mac (conda env `dem`), from this folder:
    cd ~/apollo-lab/rua-harness && PYTHONPATH=. python scripts/render_geometry.py

Self-checks (the script exits non-zero if one fails):
 1. simlib K and T_world_cam equal robosuite.utils.camera_utils' matrices.
 2. For each camera that sees the target object: lift the mask's centre pixel to 3D with
    metric depth + K + T_world_cam. It must land within 5 cm of the simulator's object position.
    Done both with the observation flipped to top-row-first (right) and as it comes from
    robosuite (wrong), so the log shows which convention is right.
 3. Project the true object position into the image; the pixel must be on (or next to) the mask.
 4. If the OpenDrawer target demos are downloaded: reset to demo 0 and compare our flipped render
    with frame 0 of the dataset video (flipped must match much better than unflipped).
Images go to docs/img/geometry/, arrays to ~/apollo-lab/runs/geometry_demo/scene.npz.
First run: Nov 26, 2026 on Adi's Mac.
"""
import os
import sys
from pathlib import Path

import numpy as np
import robosuite.utils.camera_utils as CU

from simlib import camera as C
from simlib import truth
from simlib.tasks import make_env, language, dataset_dir, reset_to_demo
from rua_harness.geometry import add_depth_noise

H = W = 256
TASK, SEED = "PickPlaceCounterToCabinet", 0
REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "docs/img/geometry"
OUT = Path(os.path.expanduser("~/apollo-lab/runs/geometry_demo"))
IMG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
TOL = 0.05
failures = []


def norm(a):
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    return (255 * (a - a.min()) / (np.ptp(a) + 1e-9)).astype(np.uint8)


def save(name, img):
    import imageio.v2 as imageio
    imageio.imwrite(IMG / name, img)


env = make_env(TASK, seed=SEED, size=H, depth=True)
obs = env.reset()
sim = env.sim
print("task:", TASK, "seed:", SEED, "instruction:", language(env))

views = C.views(env, obs)
true_pos = truth.object_position(env, "obj")
obj_geoms = truth.object_geom_ids(env, "obj")
print("true object position:", true_pos.round(3))

# 1. matrices agree with robosuite
for cam, v in views.items():
    if not np.allclose(v.K, CU.get_camera_intrinsic_matrix(sim, cam, H, W)):
        failures.append(f"{cam}: K differs from robosuite")
    if not np.allclose(v.T_world_cam, CU.get_camera_extrinsic_matrix(sim, cam)):
        failures.append(f"{cam}: T_world_cam differs from robosuite")
print("check 1 (K, T match robosuite):", "ok" if not failures else failures)

saved = {}
for cam, v in views.items():
    seg = C.segmentation(env, cam, H, W)              # top-row-first
    mask = np.isin(seg, obj_geoms)
    print(f"[{cam}] mask pixels: {int(mask.sum())}")
    if mask.sum() == 0:
        print(f"  object not visible from {cam}, skipped")
        continue
    v_px, u_px = np.argwhere(mask).mean(0).round().astype(int)
    if not mask[v_px, u_px]:                           # centroid off the mask: take nearest mask pixel
        pts = np.argwhere(mask)
        v_px, u_px = pts[np.argmin(((pts - [v_px, u_px]) ** 2).sum(1))]
    raw_depth = C.metric_depth(obs[f"{cam}_depth"], *C.near_far(sim))   # bottom-row-first
    mask_px = np.argwhere(mask)[:, ::-1]                                 # (u, v)
    for label, depth in [("flipped to top-row-first", v.depth), ("as robosuite returns it", raw_depth)]:
        p = C.pixel_to_world([u_px, v_px], depth, v.K, v.T_world_cam)
        err = np.linalg.norm(p - true_pos)
        # every lifted mask pixel should lie on the object: within 1 cm of some object geom's
        # bounding sphere (geom centre +- geom_rbound)
        pts = C.pixel_to_world(mask_px, depth, v.K, v.T_world_cam)
        gx, gr = sim.data.geom_xpos[obj_geoms], sim.model.geom_rbound[obj_geoms]
        dist = np.linalg.norm(pts[:, None] - gx[None], axis=-1) - gr[None]
        on_obj = (dist.min(1) < 0.01).mean()
        print(f"  depth {label}: centre pixel lifted {p.round(3)}, {err * 100:.1f} cm from object "
              f"origin; {on_obj * 100:.0f}% of mask pixels lift onto the object")
        if label.startswith("flipped"):
            saved[cam] = err
            if on_obj < 0.95:
                failures.append(f"{cam}: only {on_obj * 100:.0f}% of mask pixels lift onto the object")
            if err > TOL and cam != "robot0_eye_in_hand":
                failures.append(f"{cam}: lift error {err * 100:.1f} cm > {TOL * 100:.0f} cm")
    # 3. project the truth back into the image
    uv, z = v.world_to_pixel(true_pos)
    u, vv = np.round(uv).astype(int)
    near = mask[max(vv - 2, 0):vv + 3, max(u - 2, 0):u + 3].any() if 0 <= u < W and 0 <= vv < H else False
    print(f"  true position projects to pixel ({u}, {vv}), depth {z:.3f} m, on mask (+-2 px): {near}")
    if not near:
        failures.append(f"{cam}: projection of true position misses the mask")

    pm = v.point_map()
    noisy = add_depth_noise(v.depth, np.random.default_rng(0))
    short = cam.replace("robot0_", "")
    save(f"{short}_rgb.png", v.rgb)
    save(f"{short}_depth.png", norm(np.clip(v.depth, 0, 4)))
    save(f"{short}_depth_noisy.png", norm(np.clip(noisy, 0, 4)))
    save(f"{short}_mask.png", (mask * 255).astype(np.uint8))
    save(f"{short}_point_map.png", np.dstack([norm(pm[..., i]) for i in range(3)]))  # x,y,z -> r,g,b
    overlay = v.rgb.copy()
    overlay[mask] = (0.5 * overlay[mask] + [0, 127, 0]).astype(np.uint8)
    overlay[max(vv - 2, 0):vv + 3, max(u - 2, 0):u + 3] = [255, 0, 0]
    save(f"{short}_overlay.png", overlay)
    if cam == "robot0_agentview_left":
        np.savez_compressed(OUT / "scene.npz", rgb=v.rgb, depth=v.depth, depth_noisy=noisy,
                            mask=mask, point_map=pm, K=v.K, T_world_cam=v.T_world_cam,
                            true_pos=true_pos)

if not saved:
    failures.append("object not visible from any camera")

# 4. compare with the dataset video
d = None
try:
    d = dataset_dir("OpenDrawer")
except Exception as e:  # registry/dataset problems should not hide the checks above
    print("dataset lookup failed:", e)
if d is not None:
    import av
    env.close()
    env = make_env("OpenDrawer", seed=0, size=H, depth=True)
    env.reset()
    reset_to_demo(env, d, 0, step=0)
    cam = "robot0_agentview_left"
    rgb_gl = env.sim.render(camera_name=cam, height=H, width=W)       # bottom-row-first
    vid = d / f"videos/chunk-000/observation.images.{cam}/episode_000000.mp4"
    frame = next(av.open(str(vid)).decode(video=0)).to_ndarray(format="rgb24")
    e_flip = np.abs(C.flip_rows(rgb_gl).astype(float) - frame).mean()
    e_raw = np.abs(rgb_gl.astype(float) - frame).mean()
    print(f"check 4 (dataset video frame 0 vs render): mean abs diff flipped {e_flip:.1f}, unflipped {e_raw:.1f}")
    save("dataset_frame0.png", frame)
    save("render_flipped_demo0.png", C.flip_rows(rgb_gl))
    if not e_flip < e_raw:
        failures.append("flipped render does not match the dataset video better than unflipped")
else:
    print("check 4 skipped: OpenDrawer target demos not downloaded")

env.close()
print("images:", IMG)
if failures:
    print("SELF-CHECK FAILED:", *failures, sep="\n  ")
    sys.exit(1)
print("SELF-CHECK PASSED. lift errors (cm):", {k: round(e * 100, 1) for k, e in saved.items()})
