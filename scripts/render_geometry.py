"""Render RGB, depth, object mask and point map for one RoboCasa pick-and-place scene.

Run on the Mac (conda env `dem`), from this folder:
    cd ~/apollo-lab/rua-harness && PYTHONPATH=. python scripts/render_geometry.py

Self-check: lifts the mask's centre pixel to 3D and compares it with the simulator's true
object position. A small error (a few cm) means depth, intrinsics and extrinsics all line up.
NOT YET RUN on the Mac. Written against robosuite 5ce6643 / robocasa 425684d source.
"""
import os
import numpy as np
import robocasa  # noqa: F401  (registers environments)
import robosuite.utils.camera_utils as CU
from robocasa.utils.env_utils import create_env
from rua_harness.geometry import lift_pixel, depth_to_point_map, add_depth_noise

CAM, H, W = "robot0_agentview_left", 256, 256
OUT = os.path.expanduser("~/apollo-lab/runs/geometry_demo")
os.makedirs(OUT, exist_ok=True)

env = create_env(env_name="PickPlaceCounterToCabinet", split="pretrain", seed=0,
                 camera_heights=H, camera_widths=W, camera_depths=True)
obs = env.reset()
sim = env.sim

rgb = obs[f"{CAM}_image"]
depth = CU.get_real_depth_map(sim, obs[f"{CAM}_depth"][..., 0])        # metres
seg = CU.get_camera_segmentation(sim, CAM, H, W)[..., 1]               # geom id per pixel
K = CU.get_camera_intrinsic_matrix(sim, CAM, H, W)
T_world_cam = CU.get_camera_extrinsic_matrix(sim, CAM)

# mask of the target object: geoms whose body sits under the object's root body
root = env.obj_body_id["obj"]
def under_root(b):
    while b > 0:
        if b == root:
            return True
        b = sim.model.body_parentid[b]
    return False
obj_geoms = [g for g in range(sim.model.ngeom) if under_root(sim.model.geom_bodyid[g])]
mask = np.isin(seg, obj_geoms)

true_pos = sim.data.body_xpos[root]
print("instruction:", env.get_ep_meta()["lang"])
print("mask pixels:", int(mask.sum()))
best = None
for name, m, d in [("as rendered", mask, depth), ("depth flipped vertically", mask, depth[::-1])]:
    if m.sum() == 0:
        continue
    v, u = np.argwhere(m).mean(0).round().astype(int)
    p = lift_pixel([u, v], d, K, T_world_cam)
    err = np.linalg.norm(p - true_pos)
    print(f"  {name}: lifted {p.round(3)} true {true_pos.round(3)} error {err*100:.1f} cm")
    if best is None or err < best[0]:
        best = (err, name, d)
if best:
    print("use convention:", best[1])
    depth = best[2]
    if best[1] != "as rendered":
        rgb = rgb[::-1]          # keep RGB in the same top-down convention as the mask

pm = depth_to_point_map(depth, K, T_world_cam)
noisy = add_depth_noise(depth, np.random.default_rng(0))
np.savez_compressed(f"{OUT}/scene.npz", rgb=rgb, depth=depth, depth_noisy=noisy, mask=mask,
                    point_map=pm, K=K, T_world_cam=T_world_cam, true_pos=true_pos)
try:
    import imageio.v2 as imageio
    norm = lambda a: (255 * (a - a.min()) / (np.ptp(a) + 1e-9)).astype(np.uint8)
    imageio.imwrite(f"{OUT}/rgb.png", rgb)
    imageio.imwrite(f"{OUT}/depth.png", norm(depth))
    imageio.imwrite(f"{OUT}/depth_noisy.png", norm(noisy))
    imageio.imwrite(f"{OUT}/mask.png", (mask * 255).astype(np.uint8))
    imageio.imwrite(f"{OUT}/point_map.png", norm(pm))
except ImportError:
    print("imageio missing: saved scene.npz only")
print("saved to", OUT)
env.close()
