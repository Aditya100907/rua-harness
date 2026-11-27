"""Stereo depth camera noise for clean simulated depth. Pure numpy, no simulator.

Why: `rua_harness.geometry.add_depth_noise` adds Gaussian noise scaled by depth squared,
drops pixels at depth edges and drops random pixels. A real stereo camera (RealSense D435,
ZED 2) differs in ways that matter for 3D points handed to a policy:

1. The error lives in DISPARITY (pixel shift between the two imagers). Depth z = f * B / d,
   so a constant disparity error sigma_d gives depth error sigma_z = z^2 * sigma_d / (f * B).
   This is Intel's own RMS formula (see docs/sim_to_real.md, references).
2. Disparity is quantized (sub-pixel steps) and depth is stored in fixed units (1 mm).
3. The noise is spatially correlated: block matching uses a window, so neighbours err together.
4. At depth discontinuities the camera returns either nothing or "flying pixels" whose depth
   lies between the foreground and the background.
5. Next to a near object there is an occlusion shadow: background that one imager sees and
   the other does not. Width in pixels = the disparity jump. No depth there.
6. Holes on dark, shiny or textureless surfaces, where matching fails.
7. Out of range: nearer than min-Z or farther than max-Z gives no depth.

Missing depth is written as 0.0, as the RealSense SDK does.

Conventions: depth [H, W] in metres along the optical axis (simlib.camera.metric_depth),
top-row-first. `rgb` [H, W, 3] uint8 with the same rows, if given.

The disparity noise uses the REAL camera's focal length and baseline, so sigma_z(z) matches
the real sensor whatever the render size. The occlusion-shadow width uses the RENDERED image's
focal length `fx` (pass K[0, 0]), because it is measured in rendered pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class StereoCamera:
    name: str
    baseline_m: float          # distance between the two imagers
    hfov_deg: float            # horizontal field of view of the depth stream
    native_width: int          # depth stream width the noise numbers refer to
    subpixel_rms: float        # disparity noise, pixels RMS, at native_width
    subpixel_step: float       # disparity quantization step, pixels (0 = off)
    depth_unit_m: float        # depth storage step, metres (0 = off)
    min_z: float               # nearer than this: no depth
    max_z: float               # farther than this: no depth
    corr_px: int = 3           # box size (rendered px) of spatially correlated noise; 1 = iid
    edge_rel: float = 0.05     # edge threshold: relative 2nd difference of inverse depth
    edge_drop_prob: float = 0.5   # edge pixel -> 0
    edge_fly_prob: float = 0.3    # edge pixel -> random depth between near and far neighbour
    shadow: bool = True        # occlusion shadow beside near objects
    shadow_side: str = "left"  # depth aligned to the left imager: shadow left of near objects
    dark_lum: float = 25.0     # luminance (0-255) below which matching tends to fail
    dark_hole_prob: float = 0.3
    bright_lum: float = 250.0  # near-saturated (specular) pixels
    bright_hole_prob: float = 0.3
    hole_prob: float = 0.002   # independent random holes

    @property
    def focal_px(self) -> float:
        """Native focal length in pixels: 0.5 * width / tan(HFOV / 2)."""
        return focal_px(self.native_width, self.hfov_deg)


def focal_px(width: int, hfov_deg: float) -> float:
    return 0.5 * width / np.tan(np.deg2rad(hfov_deg) / 2)


def depth_rms(z, cam: StereoCamera):
    """Expected depth RMS error (m) at depth z (m): z^2 * subpixel / (f * B)."""
    z = np.asarray(z, dtype=np.float64)
    return z ** 2 * cam.subpixel_rms / (cam.focal_px * cam.baseline_m)


# ---------------------------------------------------------------- presets
# Numbers and where they come from: docs/sim_to_real.md, "Noise model". Edge, dark and hole
# probabilities are chosen by hand, not measured.

# Intel RealSense D435 at 848x480 (Intel's recommended depth resolution).
D435 = StereoCamera(
    name="d435", baseline_m=0.050, hfov_deg=87.0, native_width=848,
    subpixel_rms=0.08, subpixel_step=1 / 32, depth_unit_m=0.001,
    min_z=0.195, max_z=3.0,
)

# Stereolabs ZED 2 / 2i, 2.1 mm lens, HD720. Stereolabs publishes no sub-pixel figure;
# 0.18 px reproduces the "1% at 3 m" accuracy claim, so this is a pessimistic setting.
# Its neural depth fills dark areas and smooths more, but leaves more flying pixels at edges.
ZED2 = StereoCamera(
    name="zed2", baseline_m=0.120, hfov_deg=110.0, native_width=1280,
    subpixel_rms=0.18, subpixel_step=0.0, depth_unit_m=0.0,
    min_z=0.30, max_z=20.0, corr_px=5,
    edge_drop_prob=0.2, edge_fly_prob=0.5,
    dark_hole_prob=0.05, bright_hole_prob=0.1, hole_prob=0.0005,
)

PRESETS = {"d435": D435, "zed2": ZED2}


# ---------------------------------------------------------------- pieces

def _box_mean(x: np.ndarray, k: int) -> np.ndarray:
    """k x k mean filter with edge padding (k odd)."""
    if k <= 1:
        return x
    r = k // 2
    p = np.pad(x, r, mode="edge")
    c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    H, W = x.shape
    s = c[k:k + H, k:k + W] - c[:H, k:k + W] - c[k:k + H, :W] + c[:H, :W]
    return s / (k * k)


def correlated_normal(shape, rng: np.random.Generator, k: int) -> np.ndarray:
    """Unit-variance Gaussian field, correlated over about k pixels."""
    n = rng.normal(size=shape)
    if k <= 1:
        return n
    sm = _box_mean(n, k)
    sd = sm.std()
    return sm / sd if sd > 0 else n


def _neighbour_minmax(z: np.ndarray):
    """Min and max over the 3x3 neighbourhood (edge padded)."""
    p = np.pad(z, 1, mode="edge")
    H, W = z.shape
    stack = np.stack([p[i:i + H, j:j + W] for i in range(3) for j in range(3)])
    return stack.min(0), stack.max(0)


def edge_mask(z: np.ndarray, rel: float) -> np.ndarray:
    """Pixels at a depth discontinuity.

    Uses the second difference of inverse depth along rows and columns, relative to the pixel's
    inverse depth. Inverse depth is linear in pixel coordinates on any plane, so slanted
    surfaces (a counter seen at a grazing angle) are NOT edges; only real jumps are.
    """
    with np.errstate(divide="ignore"):
        inv = np.where(z > 0, 1.0 / np.maximum(z, 1e-9), 0.0)
    p = np.pad(inv, 1, mode="reflect", reflect_type="odd")  # linear extrapolation at borders
    d2x = np.abs(p[1:-1, :-2] - 2 * inv + p[1:-1, 2:])
    d2y = np.abs(p[:-2, 1:-1] - 2 * inv + p[2:, 1:-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.maximum(d2x, d2y) / np.maximum(inv, 1e-9)) > rel


def occlusion_shadow(z: np.ndarray, fx: float, baseline_m: float, side: str = "left",
                     max_px: int = 64) -> np.ndarray:
    """Pixels the second imager cannot see, beside nearer surfaces.

    Disparity in rendered pixels d = fx * B / z. With depth aligned to the left imager and the
    right imager at +B, pixel u is hidden when some pixel k columns to its right is nearer by
    at least k pixels of disparity: d[u + k] - d[u] >= k. side="right" mirrors this.
    """
    with np.errstate(divide="ignore"):
        d = np.where(z > 0, fx * baseline_m / np.maximum(z, 1e-9), 0.0)
    if side == "right":
        d = d[:, ::-1]
    H, W = d.shape
    kmax = int(min(max_px, W - 1, np.ceil(d.max() - d.min()) if d.size else 0))
    hidden = np.zeros_like(d, dtype=bool)
    for k in range(1, kmax + 1):
        hidden[:, :W - k] |= (d[:, k:] - d[:, :W - k]) >= k
    return hidden[:, ::-1] if side == "right" else hidden


def luminance(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


# ---------------------------------------------------------------- main entry points

def stereo_noise(depth: np.ndarray, rng: np.random.Generator, cam: StereoCamera = D435,
                 rgb: np.ndarray | None = None, fx: float | None = None,
                 scale: float = 1.0) -> np.ndarray:
    """Clean metric depth -> depth as `cam` would report it. 0.0 = no depth.

    rgb: enables dark and shiny holes. fx: rendered focal length in pixels (K[0, 0]); enables
    the occlusion shadow. scale: multiplies the disparity noise (sweep knob; 0 = no Gaussian part).
    """
    z = np.asarray(depth, dtype=np.float64)
    if z.ndim == 3:
        z = z[..., 0]
    valid_in = z > 0
    f, B = cam.focal_px, cam.baseline_m
    with np.errstate(divide="ignore"):
        disp = np.where(valid_in, f * B / np.maximum(z, 1e-9), 0.0)

    # 1-3. correlated disparity noise, then quantization
    disp = disp + scale * cam.subpixel_rms * correlated_normal(z.shape, rng, cam.corr_px)
    if cam.subpixel_step > 0:
        disp = np.round(disp / cam.subpixel_step) * cam.subpixel_step
    with np.errstate(divide="ignore"):
        out = np.where(disp > 0, f * B / np.maximum(disp, 1e-12), 0.0)
    out[~valid_in] = 0.0
    if cam.depth_unit_m > 0:
        out = np.round(out / cam.depth_unit_m) * cam.depth_unit_m

    # 4. edges: drop or fly (decided on the clean depth, as matching fails on true geometry)
    edges = edge_mask(z, cam.edge_rel) & valid_in
    if edges.any():
        u = rng.random(z.shape)
        drop = edges & (u < cam.edge_drop_prob)
        fly = edges & (u >= cam.edge_drop_prob) & (u < cam.edge_drop_prob + cam.edge_fly_prob)
        lo, hi = _neighbour_minmax(z)
        t = rng.random(z.shape)
        out = np.where(fly, lo + t * (hi - lo), out)
        out[drop] = 0.0

    # 5. occlusion shadow
    if cam.shadow and fx is not None:
        out[occlusion_shadow(z, fx, B, cam.shadow_side)] = 0.0

    # 6. dark and shiny surfaces
    if rgb is not None:
        lum = luminance(rgb)
        u = rng.random(z.shape)
        out[(lum < cam.dark_lum) & (u < cam.dark_hole_prob)] = 0.0
        out[(lum > cam.bright_lum) & (u < cam.bright_hole_prob)] = 0.0

    # 7. random holes and range limits
    out[rng.random(z.shape) < cam.hole_prob] = 0.0
    out[(z < cam.min_z) | (z > cam.max_z)] = 0.0
    return out


def gaussian_noise(depth: np.ndarray, rng: np.random.Generator, sigma_m: float) -> np.ndarray:
    """GAE-style setting: i.i.d. Gaussian depth noise with a fixed std in metres. Holes stay 0."""
    z = np.asarray(depth, dtype=np.float64)
    out = z + rng.normal(0.0, sigma_m, z.shape)
    out[z <= 0] = 0.0
    return np.maximum(out, 0.0)


# Evaluation sweep levels (docs/sim_to_real.md, "Evaluation-time noise sweep").
SWEEP = {
    "clean": None,
    "gauss_1cm": ("gauss", 0.01),
    "gauss_10cm": ("gauss", 0.10),
    "d435": ("stereo", D435, 1.0),
    "d435_x2": ("stereo", D435, 2.0),
    "d435_x4": ("stereo", D435, 4.0),
    "zed2": ("stereo", ZED2, 1.0),
    "d435_no_holes": ("stereo", replace(D435, edge_drop_prob=0.0, edge_fly_prob=0.0, shadow=False,
                                        dark_hole_prob=0.0, bright_hole_prob=0.0, hole_prob=0.0), 1.0),
}


def apply_level(level: str, depth: np.ndarray, rng: np.random.Generator,
                rgb: np.ndarray | None = None, fx: float | None = None) -> np.ndarray:
    """Apply one named sweep level to a clean depth image."""
    spec = SWEEP[level]
    if spec is None:
        return np.array(depth, dtype=np.float64, copy=True)
    if spec[0] == "gauss":
        return gaussian_noise(depth, rng, spec[1])
    _, cam, scale = spec
    return stereo_noise(depth, rng, cam, rgb=rgb, fx=fx, scale=scale)
