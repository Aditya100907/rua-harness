"""Depth noise model tests. Pure numpy, no simulator."""
from dataclasses import replace

import numpy as np
import pytest

from policy.noise import depth_noise as N

QUIET = dict(edge_drop_prob=0.0, edge_fly_prob=0.0, shadow=False, dark_hole_prob=0.0,
             bright_hole_prob=0.0, hole_prob=0.0, corr_px=1, subpixel_step=0.0, depth_unit_m=0.0)


def test_focal_and_rms_formula():
    f = N.focal_px(848, 87.0)
    assert f == pytest.approx(424 / np.tan(np.deg2rad(43.5)))
    # D435 at 1 m: a few millimetres; at 2 m: four times that (z^2) and under 2% of range
    r1, r2 = N.depth_rms(1.0, N.D435), N.depth_rms(2.0, N.D435)
    assert 0.002 < r1 < 0.006
    assert r2 == pytest.approx(4 * r1)
    assert r2 < 0.02 * 2.0
    # ZED 2 preset reproduces about 1% at 3 m
    assert N.depth_rms(3.0, N.ZED2) == pytest.approx(0.03, rel=0.1)


@pytest.mark.parametrize("z", [0.5, 1.0, 2.0])
def test_measured_std_matches_formula(z):
    cam = replace(N.D435, **QUIET)
    rng = np.random.default_rng(0)
    out = N.stereo_noise(np.full((200, 200), z), rng, cam)
    assert abs(out.mean() - z) < 0.002
    assert out.std() == pytest.approx(N.depth_rms(z, cam), rel=0.05)


def test_scale_zero_is_clean_and_scale_doubles_std():
    cam = replace(N.D435, **QUIET)
    z = np.full((100, 100), 1.5)
    assert np.allclose(N.stereo_noise(z, np.random.default_rng(1), cam, scale=0.0), z)
    s1 = N.stereo_noise(z, np.random.default_rng(1), cam, scale=1.0).std()
    s2 = N.stereo_noise(z, np.random.default_rng(1), cam, scale=2.0).std()
    assert s2 == pytest.approx(2 * s1, rel=0.01)


def test_quantization_steps():
    cam = replace(N.D435, **{**QUIET, "depth_unit_m": 0.001})
    out = N.stereo_noise(np.full((50, 50), 1.234), np.random.default_rng(2), cam)
    assert np.allclose(out * 1000, np.round(out * 1000))
    # disparity quantization alone: only a few distinct depths at a constant scene
    cam = replace(N.D435, **{**QUIET, "subpixel_step": 1 / 32})
    out = N.stereo_noise(np.full((50, 50), 1.0), np.random.default_rng(2), cam)
    assert len(np.unique(np.round(out, 9))) < 20


def test_correlated_noise_unit_variance_and_correlated():
    rng = np.random.default_rng(3)
    n = N.correlated_normal((300, 300), rng, 5)
    assert n.std() == pytest.approx(1.0, rel=1e-6)
    lag1 = np.corrcoef(n[:, :-1].ravel(), n[:, 1:].ravel())[0, 1]
    assert lag1 > 0.5
    iid = N.correlated_normal((300, 300), rng, 1)
    assert abs(np.corrcoef(iid[:, :-1].ravel(), iid[:, 1:].ravel())[0, 1]) < 0.05


def step_scene(H=64, W=64, near=0.6, far=1.2, col=32):
    z = np.full((H, W), far)
    z[:, col:] = near   # near object fills the right half
    return z


def test_edges_drop_and_fly_only_at_edges():
    z = step_scene()
    cam = replace(N.D435, **{**QUIET, "edge_drop_prob": 0.5, "edge_fly_prob": 0.5})
    out = N.stereo_noise(z, np.random.default_rng(4), cam, scale=0.0)
    changed = ~np.isclose(out, z)
    assert changed[:, 31:33].mean() > 0.9          # both sides of the step are edges
    assert not changed[:, :30].any() and not changed[:, 34:].any()
    fly = changed & (out > 0)
    assert ((out[fly] > 0.6) & (out[fly] < 1.2)).all()   # flying pixels lie between surfaces
    assert (out == 0).any()


def test_slanted_plane_is_not_an_edge():
    # a plane seen at a grazing angle: inverse depth linear in pixels, large per-pixel depth change
    H, W = 32, 64
    u = np.arange(W)[None, :].repeat(H, 0)
    z = 1.0 / (2.5 - 0.035 * u)        # 0.4 m to about 3.6 m across the row
    assert (np.abs(np.diff(z, axis=1)) / z[:, :-1]).max() > 0.05
    assert not N.edge_mask(z, 0.05).any()
    assert N.edge_mask(step_scene(), 0.05)[:, 31:33].all()


def test_occlusion_shadow_width_and_side():
    z = step_scene()
    fx, B = 100.0, 0.05
    d_near, d_far = fx * B / 0.6, fx * B / 1.2
    width = int(np.floor(d_near - d_far))
    hidden = N.occlusion_shadow(z, fx, B, side="left")
    row = hidden[10]
    assert row[32 - width:32].all()              # background just left of the near object
    assert not row[:32 - width].any()
    assert not row[32:].any()                    # the near object itself is visible
    # mirrored geometry with side="right" mirrors the shadow
    hr = N.occlusion_shadow(z[:, ::-1], fx, B, side="right")
    assert np.array_equal(hr, hidden[:, ::-1])
    # flat scene: no shadow
    assert not N.occlusion_shadow(np.full((8, 8), 1.0), fx, B).any()


def test_dark_and_bright_holes_need_rgb():
    z = np.full((100, 100), 1.0)
    rgb = np.full((100, 100, 3), 128, np.uint8)
    rgb[:, :50] = 5      # dark half
    cam = replace(N.D435, **{**QUIET, "dark_hole_prob": 1.0})
    out = N.stereo_noise(z, np.random.default_rng(5), cam, rgb=rgb, scale=0.0)
    assert (out[:, :50] == 0).all() and (out[:, 50:] > 0).all()
    out = N.stereo_noise(z, np.random.default_rng(5), cam, scale=0.0)
    assert (out > 0).all()


def test_range_limits_and_missing_input():
    z = np.array([[0.1, 0.5, 2.5, 4.0, 0.0]])
    cam = replace(N.D435, **QUIET)
    out = N.stereo_noise(z, np.random.default_rng(6), cam, scale=0.0)
    assert out[0, 0] == 0.0 and out[0, 3] == 0.0 and out[0, 4] == 0.0
    assert out[0, 1] == pytest.approx(0.5) and out[0, 2] == pytest.approx(2.5)


def test_random_hole_rate():
    cam = replace(N.D435, **{**QUIET, "hole_prob": 0.1})
    out = N.stereo_noise(np.full((200, 200), 1.0), np.random.default_rng(7), cam, scale=0.0)
    assert (out == 0).mean() == pytest.approx(0.1, abs=0.01)


def test_gaussian_level_and_sweep():
    z = np.full((200, 200), 1.0)
    z[0, 0] = 0.0
    rng = np.random.default_rng(8)
    g = N.apply_level("gauss_1cm", z, rng)
    assert g[0, 0] == 0.0
    assert g[1:].std() == pytest.approx(0.01, rel=0.05)
    assert np.array_equal(N.apply_level("clean", z, rng), z)
    for level in N.SWEEP:
        out = N.apply_level(level, step_scene(), np.random.default_rng(9),
                            rgb=np.full((64, 64, 3), 128, np.uint8), fx=200.0)
        assert out.shape == (64, 64) and np.isfinite(out).all() and (out >= 0).all()


def test_full_preset_runs_on_realistic_scene():
    rng = np.random.default_rng(10)
    z = 0.8 + 0.4 * rng.random((1, 1)) + np.zeros((128, 128))
    z[40:80, 40:80] = 0.5
    out = N.stereo_noise(z, rng, N.D435, rgb=np.full((128, 128, 3), 128, np.uint8), fx=110.0)
    valid = out > 0
    assert 0.8 < valid.mean() < 1.0
    interior = np.zeros_like(valid)
    interior[50:70, 50:70] = True
    err = np.abs(out - z)[interior & valid]
    assert np.median(err) < 0.01
