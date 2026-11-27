"""Cameras: images, metric depth, intrinsics, extrinsics, pixel <-> world.

Conventions (checked in the simulator, see docs/geometry_notes.md):

* Image rows: simlib images are TOP-ROW-FIRST (row 0 = top), like the dataset mp4 videos
  and OpenCV. robosuite observations (`obs["<cam>_image"]`, `obs["<cam>_depth"]`) are
  bottom-row-first (OpenGL), so `views()` flips them with `flip_rows`.
* Pixels: `uv = (u, v)` = (column, row) in integer pixel indices, top-row-first. The centre
  of pixel (u, v) is at continuous image coordinate (u + 0.5, v + 0.5).
* Depth: MuJoCo returns depth normalised to [0, 1]. `metric_depth` converts it to metres along
  the optical axis (z-depth, not ray length) with near/far = model.vis.map.znear/zfar * model.stat.extent.
* Camera frame: MuJoCo cameras look down their -z axis with +y up. simlib uses the OpenCV frame
  (+z forward, +x right, +y down), so T_world_cam = [R_mujoco @ diag(1, -1, -1), p].
  Same as robosuite.utils.camera_utils.get_camera_extrinsic_matrix.
* Intrinsics: fy = fx = 0.5 * H / tan(fovy / 2), cx = W / 2, cy = H / 2 (square pixels).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


# ---------------------------------------------------------------- pure numpy (no simulator)

def flip_rows(img: np.ndarray) -> np.ndarray:
    """Bottom-row-first (OpenGL, robosuite obs) <-> top-row-first (dataset videos). Self-inverse."""
    return np.ascontiguousarray(img[::-1])


def intrinsics(fovy_deg: float, height: int, width: int) -> np.ndarray:
    """3x3 pinhole matrix for a MuJoCo camera with vertical field of view `fovy_deg`."""
    f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2)
    return np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


def extrinsics_from_mujoco(cam_xpos: np.ndarray, cam_xmat: np.ndarray) -> np.ndarray:
    """4x4 pose of the camera in the world, OpenCV axes (+z forward, +y down)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(cam_xmat).reshape(3, 3) @ MUJOCO_TO_OPENCV
    T[:3, 3] = cam_xpos
    return T


def metric_depth(depth01: np.ndarray, near: float, far: float) -> np.ndarray:
    """Normalised MuJoCo depth in [0, 1] -> metres along the optical axis."""
    d = np.asarray(depth01, dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    return near / (1.0 - d * (1.0 - near / far))


def pixel_to_world(uv, depth, K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Pixel(s) -> world point(s).

    uv: (2,) or (N, 2) integer pixel indices (u = column, v = row, top-row-first).
    depth: either a metric depth IMAGE [H, W] (top-row-first; looked up at uv), or the metric
      z-depth value(s) of the pixel(s) directly (scalar or (N,)).
    Returns (3,) or (N, 3).
    """
    uv = np.asarray(uv, dtype=np.float64)
    single = uv.ndim == 1
    uv = np.atleast_2d(uv)
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 2:
        z = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    else:
        z = np.broadcast_to(depth, (len(uv),)).astype(np.float64)
    x = (uv[:, 0] + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (uv[:, 1] + 0.5 - K[1, 2]) * z / K[1, 1]
    cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    # einsum, not @: numpy 2 + Apple Accelerate prints spurious matmul warnings on big arrays
    world = np.einsum("ij,nj->ni", T_world_cam, cam)[:, :3]
    return world[0] if single else world


def world_to_pixel(p, K: np.ndarray, T_world_cam: np.ndarray):
    """World point(s) -> (uv, z).

    uv: continuous pixel-index coordinates (u = column, v = row, top-row-first); round to get the
      pixel index. z: depth in metres along the optical axis (negative = behind the camera).
    Shapes: (3,) -> ((2,), scalar); (N, 3) -> ((N, 2), (N,)).
    """
    p = np.asarray(p, dtype=np.float64)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    T_cam_world = np.linalg.inv(T_world_cam)
    cam = np.einsum("ij,nj->ni", T_cam_world, np.c_[p, np.ones(len(p))])
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * cam[:, 0] / z + K[0, 2] - 0.5
        v = K[1, 1] * cam[:, 1] / z + K[1, 2] - 0.5
    uv = np.stack([u, v], axis=-1)
    return (uv[0], z[0]) if single else (uv, z)


def depth_to_points(depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Metric depth [H, W] (top-row-first) -> world point map [H, W, 3]."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pts = pixel_to_world(np.stack([u.ravel(), v.ravel()], -1), depth.ravel(), K, T_world_cam)
    return pts.reshape(H, W, 3)


# ---------------------------------------------------------------- simulator helpers

def near_far(sim) -> tuple[float, float]:
    """Clip planes in metres, as MuJoCo uses them for the depth buffer."""
    ext = sim.model.stat.extent
    return sim.model.vis.map.znear * ext, sim.model.vis.map.zfar * ext


def camera_K(sim, cam: str, height: int, width: int) -> np.ndarray:
    return intrinsics(sim.model.cam_fovy[sim.model.camera_name2id(cam)], height, width)


def camera_T(sim, cam: str) -> np.ndarray:
    """4x4 T_world_cam (OpenCV axes) at the current simulator state. Moves with the robot."""
    i = sim.model.camera_name2id(cam)
    return extrinsics_from_mujoco(sim.data.cam_xpos[i], sim.data.cam_xmat[i])


@dataclass
class View:
    """One camera at one instant. All arrays top-row-first."""
    name: str
    rgb: np.ndarray            # [H, W, 3] uint8
    depth: np.ndarray | None   # [H, W] metres, or None if the env has no depth
    K: np.ndarray              # 3x3
    T_world_cam: np.ndarray    # 4x4, OpenCV axes

    def pixel_to_world(self, uv):
        return pixel_to_world(uv, self.depth, self.K, self.T_world_cam)

    def world_to_pixel(self, p):
        return world_to_pixel(p, self.K, self.T_world_cam)

    def point_map(self):
        return depth_to_points(self.depth, self.K, self.T_world_cam)


def views(env, obs: dict, cams=None) -> dict[str, View]:
    """Turn a robosuite observation dict into top-row-first Views with metric depth.

    Build the env with camera_depths=True to get depth. Call right after reset()/step(), so the
    camera poses match the images.
    """
    sim = env.sim
    near, far = near_far(sim)
    out = {}
    for cam in cams or [c for c in CAMERAS if f"{c}_image" in obs]:
        rgb = flip_rows(obs[f"{cam}_image"])
        H, W = rgb.shape[:2]
        d = obs.get(f"{cam}_depth")
        depth = flip_rows(metric_depth(d, near, far)) if d is not None else None
        out[cam] = View(cam, rgb, depth, camera_K(sim, cam, H, W), camera_T(sim, cam))
    return out


def render_view(env, cam: str, height: int = 256, width: int = 256) -> View:
    """Render a camera directly (no obs needed). Top-row-first, metric depth."""
    sim = env.sim
    rgb, d = sim.render(camera_name=cam, height=height, width=width, depth=True)
    near, far = near_far(sim)
    return View(cam, flip_rows(rgb), flip_rows(metric_depth(d, near, far)),
                camera_K(sim, cam, height, width), camera_T(sim, cam))


def segmentation(env, cam: str, height: int = 256, width: int = 256) -> np.ndarray:
    """[H, W] MuJoCo geom id per pixel (-1 = background), top-row-first. Grading/debug only."""
    import mujoco
    seg = env.sim.render(camera_name=cam, height=height, width=width, segmentation=True)
    ids = np.where(seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 1], -1)
    return flip_rows(ids)
