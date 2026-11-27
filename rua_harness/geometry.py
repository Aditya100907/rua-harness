"""Geometry helpers: pixel -> 3D point, depth -> point map, and sensor-like depth noise."""
from __future__ import annotations

import numpy as np


def lift_pixel(uv, depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """Pixel [u, v] + metric depth image -> [x, y, z] in the robot base frame.

    depth: [H, W] metres along the camera's optical axis.
    K: 3x3 intrinsics. T_base_cam: 4x4 pose of the camera in the base frame
    (camera looks along +z, x right, y down).
    """
    u, v = int(uv[0]), int(uv[1])
    z = float(depth[v, u])
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return (T_base_cam @ np.array([x, y, z, 1.0]))[:3]


def depth_to_point_map(depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """[H, W] depth -> [H, W, 3] point map in the base frame. Same shape as an RGB image."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    x = (u - K[0, 2]) * depth / K[0, 0]
    y = (v - K[1, 2]) * depth / K[1, 1]
    cam = np.stack([x, y, depth, np.ones_like(depth)], axis=-1)
    return (cam @ T_base_cam.T)[..., :3]


def add_depth_noise(depth: np.ndarray, rng: np.random.Generator, sigma_at_1m: float = 0.002,
                    hole_prob: float = 0.01, edge_drop: bool = True) -> np.ndarray:
    """Make clean simulated depth look more like a stereo depth camera.

    * error grows with distance squared (how stereo cameras behave)
    * random missing pixels, written as 0
    * missing pixels at depth edges, where real sensors fail most
    """
    d = depth + rng.normal(0.0, sigma_at_1m, depth.shape) * depth ** 2
    if edge_drop:
        gy, gx = np.gradient(depth)
        d[np.hypot(gx, gy) > 0.05] = 0.0
    d[rng.random(depth.shape) < hole_prob] = 0.0
    return d
